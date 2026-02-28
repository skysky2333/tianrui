from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .core import NPriorConfig
from .dataset import NPriorRowDataset
from .lit_module import NPriorLitModule
from ...data import discover_graph_ids, rows_for_graph_ids, train_val_split_graph_ids
from ...pl_utils import lightning_device_from_arg
from ...reporting import save_line_plot, write_json
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...utils import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for N prior (Lightning)")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--cond_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--sigma_min", type=float, nargs="+", default=[0.3, 1.2], help="Sigma_min spec: 1=fixed, 2=log-uniform range, >2=categorical")

    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--limit_train_batches", type=float, default=0.1)
    p.add_argument("--limit_val_batches", type=float, default=1.0)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout_sec", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def _fit_x_scaler(df: pd.DataFrame, rows: list[int], cond_cols: list[str], log_cols: set[str]) -> StandardScaler:
    x = df.loc[rows, ["RD"] + cond_cols].to_numpy(dtype=np.float32)
    x_t = torch.from_numpy(x)
    x_t[:, 1:] = apply_log_cols_torch(x_t[:, 1:], cond_cols, log_cols)
    return StandardScaler.fit(x_t.numpy())


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    cond_cols = list(args.cond_cols)
    log_cols = set(args.log_cols or [])
    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g = train_val_split_graph_ids(graph_ids, val_frac=float(args.val_frac), seed=int(args.seed))

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    scaler = _fit_x_scaler(df, train_rows, cond_cols, log_cols)

    ds_train = NPriorRowDataset(data_csv=args.data_csv, tess_root=args.tess_root, cond_cols=cond_cols, row_indices=train_rows)
    ds_val = NPriorRowDataset(data_csv=args.data_csv, tess_root=args.tess_root, cond_cols=cond_cols, row_indices=val_rows)
    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, **dl_kwargs)

    sigma_min_spec = [float(x) for x in list(args.sigma_min)]
    if not sigma_min_spec:
        raise SystemExit("--sigma_min must provide at least one value")

    def objective(trial: optuna.Trial) -> float:
        if len(sigma_min_spec) == 1:
            sigma_min = float(sigma_min_spec[0])
        elif len(sigma_min_spec) == 2:
            s0, s1 = float(sigma_min_spec[0]), float(sigma_min_spec[1])
            sigma_min = float(trial.suggest_float("sigma_min", min(s0, s1), max(s0, s1), log=True))
        else:
            sigma_min = float(trial.suggest_categorical("sigma_min", sigma_min_spec))

        cfg = NPriorConfig(
            d_h=trial.suggest_categorical("d_h", [64, 96, 128, 160, 192]),
            n_layers=trial.suggest_int("n_layers", 2, 5),
            dropout=trial.suggest_float("dropout", 0.0, 0.2),
            sigma_min=sigma_min,
        )
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)

        lit = NPriorLitModule(
            cfg=asdict(cfg),
            cond_cols=cond_cols,
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
            num_sanity_val_steps=0,
        )
        trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)
        val = trainer.callback_metrics["val/nll"]
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

    df_trials = study.trials_dataframe()
    df_trials.to_csv(str(run_dir / "trials.csv"), index=False)
    write_json(str(run_dir / "best.json"), {"best_value": float(study.best_value), "best_params": study.best_params})

    values = [t.value for t in study.trials if t.value is not None]
    xs = list(range(1, len(values) + 1))
    save_line_plot(
        out_path=str(run_dir / "optuna_history.png"),
        x=xs,
        ys={"val/nll": [float(v) for v in values]},
        title="Optuna tuning history (n prior)",
        xlabel="trial",
        ylabel="val nll",
    )

    cfg_out = {
        "task": "n_prior_tune",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "cond_cols": cond_cols,
        "log_cols": sorted(log_cols),
        "val_frac": float(args.val_frac),
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "search_space": {"sigma_min": sigma_min_spec},
        "n_trials": int(args.n_trials),
        "timeout_sec": int(args.timeout_sec),
        "max_epochs": int(args.max_epochs),
        "limit_train_batches": float(args.limit_train_batches),
        "limit_val_batches": float(args.limit_val_batches),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "base_out_dir": str(base_dir),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
    }
    write_json(str(run_dir / "config.json"), cfg_out)

    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "run_dir": str(run_dir),
                "base_out_dir": str(base_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
