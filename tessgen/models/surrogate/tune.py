from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .core import SurrogateConfig
from .lit_module import SurrogateLitModule
from ...data import (
    TessellationRowDataset,
    collate_graph_batch,
    discover_graph_ids,
    rows_for_graph_ids,
    train_val_split_graph_ids,
)
from ...pl_utils import lightning_device_from_arg
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...utils import ensure_dir, set_seed
from ...reporting import save_line_plot, write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for surrogate (Lightning)")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--target_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--limit_train_batches", type=float, default=0.2)
    p.add_argument("--limit_val_batches", type=float, default=1.0)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout_sec", type=int, default=0, help="0 = no timeout")
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def _fit_target_scaler(df: pd.DataFrame, rows: list[int], target_cols: list[str], log_cols: set[str]) -> StandardScaler:
    y = df.loc[rows, target_cols].to_numpy(dtype=np.float32)
    y_t = torch.from_numpy(y)
    y_t = apply_log_cols_torch(y_t, target_cols, log_cols)
    return StandardScaler.fit(y_t.numpy())


def main() -> None:
    args = _parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    target_cols = list(args.target_cols)
    log_cols = set(args.log_cols or [])

    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g = train_val_split_graph_ids(graph_ids, val_frac=float(args.val_frac), seed=int(args.seed))

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    scaler = _fit_target_scaler(df, train_rows, target_cols, log_cols)

    ds_train = TessellationRowDataset(
        data_csv=args.data_csv, tess_root=args.tess_root, target_cols=target_cols, row_indices=train_rows, cache_graphs=True
    )
    ds_val = TessellationRowDataset(
        data_csv=args.data_csv, tess_root=args.tess_root, target_cols=target_cols, row_indices=val_rows, cache_graphs=True
    )

    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_graph_batch, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_graph_batch, **dl_kwargs)

    def objective(trial: optuna.Trial) -> float:
        cfg = SurrogateConfig(
            d_h=trial.suggest_categorical("d_h", [64, 96, 128, 160, 192, 256]),
            n_layers=trial.suggest_int("n_layers", 2, 6),
            n_rbf=trial.suggest_categorical("n_rbf", [8, 16, 24]),
            dropout=trial.suggest_float("dropout", 0.0, 0.2),
        )
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)

        lit = SurrogateLitModule(
            cfg=asdict(cfg),
            target_cols=target_cols,
            log_cols=sorted(log_cols),
            scaler_mean=scaler.mean.tolist(),
            scaler_std=scaler.std.tolist(),
            lr=float(lr),
            weight_decay=float(wd),
        )
        trainer = pl.Trainer(
            max_epochs=int(args.max_epochs),
            accelerator=dev.accelerator,
            devices=dev.devices,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=float(args.limit_train_batches),
            limit_val_batches=float(args.limit_val_batches),
        )
        trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)
        val = trainer.callback_metrics["val/mse_z"]
        val_f = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)
        del trainer
        del lit
        if dev.accelerator == "gpu":
            torch.cuda.empty_cache()
        if dev.accelerator == "mps":
            torch.mps.empty_cache()
        return val_f

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(args.n_trials), timeout=None if args.timeout_sec == 0 else int(args.timeout_sec))

    # Save results
    df_trials = study.trials_dataframe()
    df_trials.to_csv(str(Path(args.out_dir) / "trials.csv"), index=False)
    write_json(str(Path(args.out_dir) / "best.json"), {"best_value": float(study.best_value), "best_params": study.best_params})

    # Simple plot: trial value over time
    values = [t.value for t in study.trials if t.value is not None]
    xs = list(range(1, len(values) + 1))
    save_line_plot(
        out_path=str(Path(args.out_dir) / "optuna_history.png"),
        x=xs,
        ys={"val/mse_z": [float(v) for v in values]},
        title="Optuna tuning history (surrogate)",
        xlabel="trial",
        ylabel="val MSE (z-space)",
    )

    cfg_out = {
        "task": "surrogate_tune",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "target_cols": target_cols,
        "log_cols": sorted(log_cols),
        "val_frac": float(args.val_frac),
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "n_trials": int(args.n_trials),
        "timeout_sec": int(args.timeout_sec),
        "max_epochs": int(args.max_epochs),
        "limit_train_batches": float(args.limit_train_batches),
        "limit_val_batches": float(args.limit_val_batches),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
    }
    write_json(str(Path(args.out_dir) / "config.json"), cfg_out)

    print(json.dumps({"best_value": float(study.best_value), "best_params": study.best_params}), flush=True)


if __name__ == "__main__":
    main()
