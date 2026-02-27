from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .core import DiffusionConfig, NodeDenoiserConfig
from .cycle_callback import CycleEvalEveryEpochCallback, node_bundle_from_lit
from .lit_module import NodeDiffusionLitModule, export_node_diffusion_pt
from .report import eval_node_diffusion, make_report_and_figures
from ...ckpt import load_edge_model, load_n_prior, load_surrogate
from ...cycle_eval import run_cycle_eval
from ...data import (
    GraphStore,
    TessellationRowDataset,
    collate_first,
    discover_graph_ids,
    rows_for_graph_ids,
    train_val_test_split_graph_ids,
)
from ...pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from ...pl_utils import lightning_device_from_arg
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...utils import device_from_arg, ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train node diffusion (Lightning): (RD, logN, metrics) -> coords")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--cond_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--k_nn", type=int, default=12)

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--limit_test_batches", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--report_max_samples", type=int, default=0, help="0 = evaluate full test set in report")

    p.add_argument("--cycle_surrogate_ckpt", type=str, default="", help="If set, run end-to-end cycle eval on test rows")
    p.add_argument("--cycle_edge_ckpt", type=str, default="", help="Edge model ckpt used for cycle eval")
    p.add_argument("--cycle_k_best", type=int, default=8)
    p.add_argument("--cycle_deg_cap", type=int, default=12)
    p.add_argument("--cycle_min_n", type=int, default=64)
    p.add_argument("--cycle_max_n", type=int, default=5000)
    p.add_argument("--cycle_n_mode", type=str, default="true", help="N selection: true|fixed|candidates|prior")
    p.add_argument("--cycle_n_fixed", type=int, default=0)
    p.add_argument("--cycle_n_candidates", type=int, nargs="*", default=[])
    p.add_argument("--cycle_n_prior_ckpt", type=str, default="")
    p.add_argument("--cycle_n_prior_samples", type=int, default=12)
    p.add_argument("--cycle_max_rows", type=int, default=0, help="0 = all test rows")
    p.add_argument("--cycle_each_epoch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cycle_epoch_rows", type=int, default=100, help="-1 = all validation rows")
    p.add_argument("--cycle_epoch_save_row_figs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cycle_epoch_save_graph_files", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cycle_epoch_progress_every", type=int, default=1, help="0 disables cycle progress prints during training")
    p.add_argument("--cycle_save_row_figs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cycle_save_graph_files", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _fit_cond_scaler(
    df: pd.DataFrame,
    rows: list[int],
    *,
    tess_root: str,
    cond_cols: list[str],
    log_cols: set[str],
) -> StandardScaler:
    store = GraphStore(tess_root=tess_root)
    graph_ids = sorted({(int(r) // 5) + 1 for r in rows})
    logn_by_gid = {gid: float(np.log(float(store.get(gid).n_nodes))) for gid in graph_ids}

    base = df.loc[rows, ["RD"] + cond_cols].to_numpy(dtype=np.float32)
    logn = np.array([logn_by_gid[(int(r) // 5) + 1] for r in rows], dtype=np.float32).reshape(-1, 1)
    x = np.concatenate([base[:, 0:1], logn, base[:, 1:]], axis=1)
    x_t = torch.from_numpy(x)
    x_t[:, 2:] = apply_log_cols_torch(x_t[:, 2:], cond_cols, log_cols)
    return StandardScaler.fit(x_t.numpy())


def main() -> None:
    args = _parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    cond_cols = list(args.cond_cols)
    log_cols = set(args.log_cols or [])

    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g, test_g = train_val_test_split_graph_ids(
        graph_ids, val_frac=float(args.val_frac), test_frac=float(args.test_frac), seed=int(args.seed)
    )

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    test_rows = rows_for_graph_ids(len(df), test_g)

    cond_scaler = _fit_cond_scaler(df, train_rows, tess_root=args.tess_root, cond_cols=cond_cols, log_cols=log_cols)

    ds_train = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=["RS"],  # unused
        cond_cols=cond_cols,
        row_indices=train_rows,
        cache_graphs=True,
    )
    ds_val = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=["RS"],
        cond_cols=cond_cols,
        row_indices=val_rows,
        cache_graphs=True,
    )
    ds_test = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=["RS"],
        cond_cols=cond_cols,
        row_indices=test_rows,
        cache_graphs=True,
    )

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_first, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)

    den_cfg = NodeDenoiserConfig(
        cond_dim=2 + len(cond_cols),
        d_h=int(args.d_h),
        n_layers=int(args.n_layers),
        n_rbf=int(args.n_rbf),
        dropout=float(args.dropout),
    )
    schedule_cfg = DiffusionConfig(n_steps=int(args.steps), beta_start=float(args.beta_start), beta_end=float(args.beta_end))

    lit = NodeDiffusionLitModule(
        denoiser_cfg=asdict(den_cfg),
        schedule_cfg=asdict(schedule_cfg),
        cond_cols=cond_cols,
        log_cols=sorted(log_cols),
        cond_scaler_mean=cond_scaler.mean.tolist(),
        cond_scaler_std=cond_scaler.std.tolist(),
        k_nn=int(args.k_nn),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    cycle_enabled = bool(str(args.cycle_surrogate_ckpt) or str(args.cycle_edge_ckpt))
    cycle_each_epoch = bool(args.cycle_each_epoch) and cycle_enabled
    torch_device = device_from_arg(args.device)

    if cycle_enabled and (not str(args.cycle_surrogate_ckpt) or not str(args.cycle_edge_ckpt)):
        raise SystemExit("For cycle eval, provide both --cycle_surrogate_ckpt and --cycle_edge_ckpt.")

    cycle_cb = None
    if cycle_each_epoch:
        surrogate = load_surrogate(str(args.cycle_surrogate_ckpt), device=torch_device)
        edge_bundle = load_edge_model(str(args.cycle_edge_ckpt), device=torch_device)
        n_prior = None
        if str(args.cycle_n_mode) == "prior":
            if not str(args.cycle_n_prior_ckpt):
                raise SystemExit("For cycle_n_mode=prior, provide --cycle_n_prior_ckpt.")
            n_prior = load_n_prior(str(args.cycle_n_prior_ckpt), device=torch_device)
        cycle_cb = CycleEvalEveryEpochCallback(
            df=df,
            tess_root=args.tess_root,
            row_indices=val_rows,
            surrogate=surrogate,
            edge_bundle=edge_bundle,
            n_prior=n_prior,
            device=torch_device,
            out_dir_base=str(Path(args.out_dir) / "cycle_val"),
            epoch_rows=int(args.cycle_epoch_rows),
            k_best=int(args.cycle_k_best),
            deg_cap=int(args.cycle_deg_cap),
            min_n=int(args.cycle_min_n),
            max_n=int(args.cycle_max_n),
            n_mode=str(args.cycle_n_mode),
            n_fixed=int(args.cycle_n_fixed),
            n_candidates=[int(x) for x in list(args.cycle_n_candidates)],
            n_prior_samples=int(args.cycle_n_prior_samples),
            save_row_figs=bool(args.cycle_epoch_save_row_figs),
            save_graph_files=bool(args.cycle_epoch_save_graph_files),
            progress_every=int(args.cycle_epoch_progress_every),
        )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best",
        monitor="val/cycle_r_best" if cycle_each_epoch else "val/loss",
        mode="max" if cycle_each_epoch else "min",
        save_top_k=1,
        save_last=True,
    )
    history_path = str(Path(args.out_dir) / "history.jsonl")
    callbacks = [c for c in [cycle_cb, ckpt_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()] if c is not None]

    t0 = time.time()
    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        accelerator=dev.accelerator,
        devices=dev.devices,
        default_root_dir=args.out_dir,
        logger=False,
        callbacks=callbacks,
        log_every_n_steps=int(args.log_every_n_steps),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        num_sanity_val_steps=0 if cycle_each_epoch else 2,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    print(
        json.dumps(
            {
                "device": {"accelerator": dev.accelerator, "devices": dev.devices},
                "splits": {"train_graphs": len(train_g), "val_graphs": len(val_g), "test_graphs": len(test_g)},
                "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
            }
        ),
        flush=True,
    )

    trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)

    best_path = ckpt_cb.best_model_path
    if not best_path:
        raise RuntimeError("No best checkpoint was saved. Ensure validation ran and ModelCheckpoint is monitoring a metric.")
    best_lit = NodeDiffusionLitModule.load_from_checkpoint(best_path)
    trainer.test(best_lit, dataloaders=dl_test, verbose=False)

    if ckpt_cb.best_model_score is None:
        raise RuntimeError("No best_model_score found on ModelCheckpoint callback.")
    export_node_diffusion_pt(
        lit=best_lit,
        out_path=str(Path(args.out_dir) / "node_diffusion.pt"),
        monitor={
            "name": str(ckpt_cb.monitor),
            "mode": str(ckpt_cb.mode),
            "value": float(ckpt_cb.best_model_score),
        },
    )

    # Report + figures
    device = torch_device
    best_lit = best_lit.to(device)
    test_eval = eval_node_diffusion(lit=best_lit, dl=dl_test, max_samples=int(args.report_max_samples))
    cycle_eval = None
    if cycle_enabled:
        cycle_rows = test_rows if int(args.cycle_max_rows) == 0 else test_rows[: int(args.cycle_max_rows)]
        if cycle_cb is None:
            surrogate = load_surrogate(str(args.cycle_surrogate_ckpt), device=device)
            edge_bundle = load_edge_model(str(args.cycle_edge_ckpt), device=device)
            n_prior = None
            if str(args.cycle_n_mode) == "prior":
                if not str(args.cycle_n_prior_ckpt):
                    raise SystemExit("For cycle_n_mode=prior, provide --cycle_n_prior_ckpt.")
                n_prior = load_n_prior(str(args.cycle_n_prior_ckpt), device=device)
        else:
            surrogate = cycle_cb.surrogate
            edge_bundle = cycle_cb.edge_bundle
            n_prior = cycle_cb.n_prior
        node_bundle = node_bundle_from_lit(lit=best_lit)
        cycle_eval = run_cycle_eval(
            df=df,
            row_indices=[int(x) for x in cycle_rows],
            tess_root=args.tess_root,
            surrogate=surrogate,
            node_bundle=node_bundle,
            edge_bundle=edge_bundle,
            n_prior=n_prior,
            device=device,
            k_best=int(args.cycle_k_best),
            deg_cap=int(args.cycle_deg_cap),
            min_n=int(args.cycle_min_n),
            max_n=int(args.cycle_max_n),
            n_mode=str(args.cycle_n_mode),
            n_fixed=int(args.cycle_n_fixed),
            n_candidates=[int(x) for x in list(args.cycle_n_candidates)],
            n_prior_samples=int(args.cycle_n_prior_samples),
            out_dir=str(Path(args.out_dir) / "cycle"),
            save_row_figs=bool(args.cycle_save_row_figs),
            save_graph_files=bool(args.cycle_save_graph_files),
            progress_prefix="cycle/test",
        )
    make_report_and_figures(run_dir=args.out_dir, history_path=history_path, test_eval=test_eval, cycle_eval=cycle_eval)

    config = {
        "task": "node_diffusion",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "cond_cols": cond_cols,
        "log_cols": sorted(log_cols),
        "denoiser_cfg": asdict(den_cfg),
        "schedule_cfg": asdict(schedule_cfg),
        "train": {"epochs": int(args.epochs), "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "k_nn": int(args.k_nn),
        "cycle_eval": {
            "enabled": bool(str(args.cycle_surrogate_ckpt) or str(args.cycle_edge_ckpt)),
            "surrogate_ckpt": str(args.cycle_surrogate_ckpt) if str(args.cycle_surrogate_ckpt) else None,
            "edge_ckpt": str(args.cycle_edge_ckpt) if str(args.cycle_edge_ckpt) else None,
            "k_best": int(args.cycle_k_best),
            "deg_cap": int(args.cycle_deg_cap),
            "min_n": int(args.cycle_min_n),
            "max_n": int(args.cycle_max_n),
            "n_mode": str(args.cycle_n_mode),
            "n_fixed": int(args.cycle_n_fixed),
            "n_candidates": [int(x) for x in list(args.cycle_n_candidates)],
            "n_prior_ckpt": str(args.cycle_n_prior_ckpt) if str(args.cycle_n_prior_ckpt) else None,
            "n_prior_samples": int(args.cycle_n_prior_samples),
            "max_rows": int(args.cycle_max_rows),
            "each_epoch": bool(cycle_each_epoch),
            "epoch_rows": int(args.cycle_epoch_rows),
            "epoch_save_row_figs": bool(args.cycle_epoch_save_row_figs),
            "epoch_save_graph_files": bool(args.cycle_epoch_save_graph_files),
            "epoch_progress_every": int(args.cycle_epoch_progress_every),
            "save_row_figs": bool(args.cycle_save_row_figs),
            "save_graph_files": bool(args.cycle_save_graph_files),
        },
        "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac)},
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "elapsed_sec": float(time.time() - t0),
        "best_ckpt": best_path,
    }
    with open(str(Path(args.out_dir) / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
