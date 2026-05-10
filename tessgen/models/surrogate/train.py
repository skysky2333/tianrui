from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .core import SurrogateConfig
from .lit_module import SurrogateLitModule, export_surrogate_pt
from .report import make_report_and_figures, predict_on_loader
from ...data import (
    TessellationRowDataset,
    collate_graph_batch,
    discover_graph_ids,
    rows_for_graph_ids,
    train_val_test_split_graph_ids,
)
from ...pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from ...pl_utils import lightning_device_from_arg
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...utils import device_from_arg, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train surrogate (Lightning): (graph, RD) -> metrics")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--target_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])
    p.add_argument("--use_rd", action=argparse.BooleanOptionalAction, default=True, help="Include RD as an input feature")
    p.add_argument(
        "--train_per_rd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train one surrogate per fixed RD slice; this automatically disables --use_rd.",
    )
    p.add_argument(
        "--rd_values",
        type=float,
        nargs="*",
        default=[],
        help="Optional RD values to keep when --train_per_rd is set. Defaults to all distinct RD values in the CSV.",
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit_train_batches", type=float, default=1.0, help="Lightning limit_train_batches (float<=1.0 or int)")
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--limit_test_batches", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    return p.parse_args()


def _fit_target_scaler(df: pd.DataFrame, rows: list[int], target_cols: list[str], log_cols: set[str]) -> StandardScaler:
    y = df.loc[rows, target_cols].to_numpy(dtype=np.float32)
    y_t = torch.from_numpy(y)
    y_t = apply_log_cols_torch(y_t, target_cols, log_cols)
    return StandardScaler.fit(y_t.numpy())


def _rows_for_rd(df: pd.DataFrame, rows: list[int], rd_value: float) -> list[int]:
    if not rows:
        return []
    rd = df.loc[rows, "RD"].to_numpy(dtype=np.float64)
    keep = np.isclose(rd, float(rd_value), atol=1e-8, rtol=0.0)
    return [int(row_idx) for row_idx, use_row in zip(rows, keep.tolist()) if use_row]


def _rd_tag(rd_value: float) -> str:
    return f"rd_{float(rd_value):g}".replace("-", "m").replace(".", "p")


def _mean_or_nan(values: list[float]) -> float:
    vals = [float(v) for v in values if isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _model_cfg_from_args(args: argparse.Namespace, *, use_rd: bool) -> SurrogateConfig:
    return SurrogateConfig(
        d_h=int(args.d_h),
        n_layers=int(args.n_layers),
        n_rbf=int(args.n_rbf),
        dropout=float(args.dropout),
        use_rd=bool(use_rd),
    )


def _summary_row_from_slice(summary: dict[str, Any], target_cols: list[str]) -> dict[str, Any]:
    regression = dict(summary["test"]["regression"])
    row: dict[str, Any] = {
        "rd": None if summary["rd_value"] is None else float(summary["rd_value"]),
        "train_rows": int(summary["rows"]["train"]),
        "val_rows": int(summary["rows"]["val"]),
        "test_rows": int(summary["rows"]["test"]),
        "test_mse_z": float(summary["test"]["mse_z"]),
        "mae_mean": float(regression["mae_mean"]),
        "rmse_mean": float(regression["rmse_mean"]),
        "mae_log_mean": float(regression["mae_log_mean"]),
        "rmse_log_mean": float(regression["rmse_log_mean"]),
        "r2_log_mean": float(regression["r2_log_mean"]),
        "elapsed_sec": float(summary["elapsed_sec"]),
        "run_dir": str(summary["run_dir"]),
    }
    per_col = dict(regression.get("per_col", {}))
    for col in target_cols:
        metrics = dict(per_col.get(col, {}))
        for metric_name, metric_value in metrics.items():
            row[f"{col}_{metric_name}"] = float(metric_value)
    return row


def _aggregate_per_rd(slice_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {
        "n_slices": int(len(slice_rows)),
        "mean_test_mse_z": _mean_or_nan([float(r["test_mse_z"]) for r in slice_rows]),
        "mean_mae_mean": _mean_or_nan([float(r["mae_mean"]) for r in slice_rows]),
        "mean_rmse_mean": _mean_or_nan([float(r["rmse_mean"]) for r in slice_rows]),
        "mean_mae_log_mean": _mean_or_nan([float(r["mae_log_mean"]) for r in slice_rows]),
        "mean_rmse_log_mean": _mean_or_nan([float(r["rmse_log_mean"]) for r in slice_rows]),
        "mean_r2_log_mean": _mean_or_nan([float(r["r2_log_mean"]) for r in slice_rows]),
    }

    by_rmse = [r for r in slice_rows if isfinite(float(r["rmse_mean"]))]
    if by_rmse:
        best_rmse = min(by_rmse, key=lambda r: float(r["rmse_mean"]))
        aggregate["best_rd_by_rmse_mean"] = {
            "rd": float(best_rmse["rd"]),
            "rmse_mean": float(best_rmse["rmse_mean"]),
            "run_dir": str(best_rmse["run_dir"]),
        }

    by_r2_log = [r for r in slice_rows if isfinite(float(r["r2_log_mean"]))]
    if by_r2_log:
        best_r2_log = max(by_r2_log, key=lambda r: float(r["r2_log_mean"]))
        aggregate["best_rd_by_r2_log_mean"] = {
            "rd": float(best_r2_log["rd"]),
            "r2_log_mean": float(best_r2_log["r2_log_mean"]),
            "run_dir": str(best_r2_log["run_dir"]),
        }
    return aggregate


def _write_per_rd_findings(
    *,
    path: Path,
    data_csv: str,
    target_cols: list[str],
    use_rd: bool,
    slice_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    lines = [
        "# Per-RD surrogate findings",
        "",
        f"- data_csv: `{data_csv}`",
        f"- target_cols: `{', '.join(target_cols)}`",
        f"- surrogate_input: `graph only`" if not use_rd else f"- surrogate_input: `graph + RD`",
        f"- slices_trained: `{len(slice_rows)}`",
        "",
        "| RD | train_rows | val_rows | test_rows | test_mse_z | rmse_mean | r2_log_mean | run_dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in slice_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['rd']):g}",
                    str(int(row["train_rows"])),
                    str(int(row["val_rows"])),
                    str(int(row["test_rows"])),
                    f"{float(row['test_mse_z']):.6g}",
                    f"{float(row['rmse_mean']):.6g}",
                    f"{float(row['r2_log_mean']):.6g}",
                    f"`{row['run_dir']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- mean_test_mse_z: `{float(aggregate['mean_test_mse_z']):.6g}`",
            f"- mean_rmse_mean: `{float(aggregate['mean_rmse_mean']):.6g}`",
            f"- mean_r2_log_mean: `{float(aggregate['mean_r2_log_mean']):.6g}`",
        ]
    )
    best_rmse = aggregate.get("best_rd_by_rmse_mean")
    if isinstance(best_rmse, dict):
        lines.append(
            f"- best_rd_by_rmse_mean: `RD={float(best_rmse['rd']):g}` with `rmse_mean={float(best_rmse['rmse_mean']):.6g}`"
        )
    best_r2_log = aggregate.get("best_rd_by_r2_log_mean")
    if isinstance(best_r2_log, dict):
        lines.append(
            f"- best_rd_by_r2_log_mean: `RD={float(best_r2_log['rd']):g}` with `r2_log_mean={float(best_r2_log['r2_log_mean']):.6g}`"
        )
    path.write_text("\n".join(lines) + "\n")


def _cleanup_device_cache(dev) -> None:
    if dev.accelerator == "gpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dev.accelerator == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _train_single_surrogate(
    *,
    args: argparse.Namespace,
    df: pd.DataFrame,
    dev,
    dl_kwargs: dict[str, Any],
    target_cols: list[str],
    log_cols: set[str],
    use_rd: bool,
    train_rows: list[int],
    val_rows: list[int],
    test_rows: list[int],
    run_dir: Path,
    run_name: str,
    base_dir: Path,
    torch_device: torch.device,
    rd_value: float | None,
    mode: str,
) -> dict[str, Any]:
    if not train_rows:
        raise RuntimeError(f"No training rows available for mode={mode!r}, rd_value={rd_value!r}")
    if not val_rows:
        raise RuntimeError(f"No validation rows available for mode={mode!r}, rd_value={rd_value!r}")
    if not test_rows:
        raise RuntimeError(f"No test rows available for mode={mode!r}, rd_value={rd_value!r}")

    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scaler = _fit_target_scaler(df, train_rows, target_cols, log_cols)

    ds_train = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=target_cols,
        row_indices=train_rows,
        cache_graphs=True,
    )
    ds_val = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=target_cols,
        row_indices=val_rows,
        cache_graphs=True,
    )
    ds_test = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=target_cols,
        row_indices=test_rows,
        cache_graphs=True,
    )

    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_graph_batch, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_graph_batch, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_graph_batch, **dl_kwargs)

    cfg = _model_cfg_from_args(args, use_rd=use_rd)
    lit = SurrogateLitModule(
        cfg=asdict(cfg),
        target_cols=target_cols,
        log_cols=sorted(log_cols),
        scaler_mean=scaler.mean.tolist(),
        scaler_std=scaler.std.tolist(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val/mse_z",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    history_path = str(run_dir / "history.jsonl")
    callbacks = [ckpt_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()]

    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        accelerator=dev.accelerator,
        devices=dev.devices,
        default_root_dir=str(run_dir),
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=int(args.log_every_n_steps),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)

    best_path = ckpt_cb.best_model_path
    if not best_path:
        raise RuntimeError("No best checkpoint was saved. Ensure validation ran and ModelCheckpoint is monitoring a metric.")
    best_lit = SurrogateLitModule.load_from_checkpoint(best_path)

    test_out = trainer.test(best_lit, dataloaders=dl_test, verbose=False)
    test_mse_z = float(test_out[0]["test/mse_z"])

    if ckpt_cb.best_model_score is None:
        raise RuntimeError("No best_model_score found on ModelCheckpoint callback.")
    export_surrogate_pt(
        lit=best_lit,
        out_path=str(run_dir / "surrogate.pt"),
        val_mse_z=float(ckpt_cb.best_model_score),
    )

    y_true, y_pred, mse_z_eval = predict_on_loader(lit=best_lit, dl=dl_test, device=torch_device)
    report = make_report_and_figures(
        run_dir=str(run_dir),
        history_path=history_path,
        target_cols=target_cols,
        log_cols=sorted(log_cols),
        y_true=y_true,
        y_pred=y_pred,
        test_mse_z=mse_z_eval,
    )

    elapsed_sec = float(time.time() - t0)
    config = {
        "task": "surrogate",
        "mode": mode,
        "rd_value": None if rd_value is None else float(rd_value),
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "target_cols": target_cols,
        "log_cols": sorted(log_cols),
        "use_rd": bool(use_rd),
        "model_cfg": asdict(cfg),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac)},
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "elapsed_sec": elapsed_sec,
        "best_ckpt": best_path,
        "test_mse_z_lightning": test_mse_z,
        "base_out_dir": str(base_dir),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
    }
    with open(str(run_dir / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    _cleanup_device_cache(dev)
    return {
        "task": "surrogate",
        "mode": mode,
        "rd_value": None if rd_value is None else float(rd_value),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
        "best_ckpt": best_path,
        "surrogate_pt": str(run_dir / "surrogate.pt"),
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "elapsed_sec": elapsed_sec,
        "test": report["test"],
    }


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    target_cols = list(args.target_cols)
    log_cols = set(args.log_cols or [])
    train_per_rd = bool(args.train_per_rd)
    use_rd = bool(args.use_rd)
    if train_per_rd and use_rd:
        print(json.dumps({"info": "--train_per_rd requested; forcing surrogate training to ignore RD inputs."}), flush=True)
        use_rd = False

    dev = lightning_device_from_arg(args.device)
    torch_device = device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}

    # Split by graph_id to avoid leakage across RD rows
    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g, test_g = train_val_test_split_graph_ids(
        graph_ids, val_frac=float(args.val_frac), test_frac=float(args.test_frac), seed=int(args.seed)
    )

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    test_rows = rows_for_graph_ids(len(df), test_g)

    t0 = time.time()
    common_payload = {
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "splits": {"train_graphs": len(train_g), "val_graphs": len(val_g), "test_graphs": len(test_g)},
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "use_rd": bool(use_rd),
        "train_per_rd": bool(train_per_rd),
    }

    if not train_per_rd:
        print(json.dumps(common_payload), flush=True)
        _train_single_surrogate(
            args=args,
            df=df,
            dev=dev,
            dl_kwargs=dl_kwargs,
            target_cols=target_cols,
            log_cols=log_cols,
            use_rd=use_rd,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            run_dir=run_dir,
            run_name=run_name,
            base_dir=base_dir,
            torch_device=torch_device,
            rd_value=None,
            mode="single",
        )
    else:
        requested_rd_values = [float(v) for v in list(args.rd_values)]
        if requested_rd_values:
            rd_values = sorted(dict.fromkeys(requested_rd_values))
        else:
            all_rows = sorted(set(train_rows + val_rows + test_rows))
            rd_values = sorted(float(v) for v in df.loc[all_rows, "RD"].dropna().unique().tolist())
        if not rd_values:
            raise RuntimeError("No RD values found for per-RD surrogate training.")

        print(json.dumps({**common_payload, "rd_values": rd_values}), flush=True)
        slice_summaries: list[dict[str, Any]] = []
        slice_rows_summary: list[dict[str, Any]] = []
        for rd_value in rd_values:
            rd_train_rows = _rows_for_rd(df, train_rows, rd_value)
            rd_val_rows = _rows_for_rd(df, val_rows, rd_value)
            rd_test_rows = _rows_for_rd(df, test_rows, rd_value)
            print(
                json.dumps(
                    {
                        "rd": float(rd_value),
                        "rows": {"train": len(rd_train_rows), "val": len(rd_val_rows), "test": len(rd_test_rows)},
                    }
                ),
                flush=True,
            )
            slice_dir = run_dir / _rd_tag(rd_value)
            summary = _train_single_surrogate(
                args=args,
                df=df,
                dev=dev,
                dl_kwargs=dl_kwargs,
                target_cols=target_cols,
                log_cols=log_cols,
                use_rd=False,
                train_rows=rd_train_rows,
                val_rows=rd_val_rows,
                test_rows=rd_test_rows,
                run_dir=slice_dir,
                run_name=slice_dir.name,
                base_dir=base_dir,
                torch_device=torch_device,
                rd_value=float(rd_value),
                mode="per_rd_slice",
            )
            slice_summaries.append(summary)
            slice_rows_summary.append(_summary_row_from_slice(summary, target_cols))

        aggregate = _aggregate_per_rd(slice_rows_summary)
        per_rd_report = {
            "task": "surrogate",
            "mode": "per_rd",
            "data_csv": args.data_csv,
            "tess_root": args.tess_root,
            "target_cols": target_cols,
            "log_cols": sorted(log_cols),
            "use_rd": False,
            "rd_values": rd_values,
            "aggregate": aggregate,
            "slices": slice_summaries,
        }
        with open(str(run_dir / "report.json"), "w") as f:
            json.dump(per_rd_report, f, indent=2)
        pd.DataFrame(slice_rows_summary).to_csv(str(run_dir / "rd_slice_summary.csv"), index=False)
        with open(str(run_dir / "rd_slice_summary.json"), "w") as f:
            json.dump({"rows": slice_rows_summary, "aggregate": aggregate}, f, indent=2)
        _write_per_rd_findings(
            path=run_dir / "findings.md",
            data_csv=args.data_csv,
            target_cols=target_cols,
            use_rd=False,
            slice_rows=slice_rows_summary,
            aggregate=aggregate,
        )

        config = {
            "task": "surrogate",
            "mode": "per_rd",
            "data_csv": args.data_csv,
            "tess_root": args.tess_root,
            "target_cols": target_cols,
            "log_cols": sorted(log_cols),
            "requested_use_rd": bool(args.use_rd),
            "use_rd": False,
            "train_per_rd": True,
            "rd_values": rd_values,
            "model_cfg": asdict(_model_cfg_from_args(args, use_rd=False)),
            "train": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
            },
            "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac)},
            "device": {"accelerator": dev.accelerator, "devices": dev.devices},
            "elapsed_sec": float(time.time() - t0),
            "base_out_dir": str(base_dir),
            "run_dir": str(run_dir),
            "run_name": str(run_name),
            "slice_dirs": [str(run_dir / _rd_tag(rd_value)) for rd_value in rd_values],
        }
        with open(str(run_dir / "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(f"saved_run_dir: {run_dir}")
    print(f"saved_base_dir: {base_dir} (latest files copied here)")


if __name__ == "__main__":
    main()
