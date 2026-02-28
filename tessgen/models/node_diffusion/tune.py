from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .core import DiffusionConfig, NodeDenoiserConfig
from .lit_module import NodeDiffusionLitModule
from ...data import (
    GraphStore,
    TessellationRowDataset,
    collate_first,
    discover_graph_ids,
    rows_for_graph_ids,
    train_val_split_graph_ids,
)
from ...pl_utils import lightning_device_from_arg
from ...reporting import save_line_plot, write_json
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...utils import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for node diffusion (Lightning)")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--cond_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--k_nn", type=int, nargs="+", default=[6, 12, 24], help="Candidate kNN values to tune over (one or more ints)")
    p.add_argument("--steps", type=int, nargs="+", default=[50, 100, 200], help="Candidate diffusion step counts to tune over")
    p.add_argument(
        "--beta_start",
        type=float,
        nargs="+",
        default=[1e-5, 5e-4],
        help="Beta start spec: 1 value=fixed, 2 values=log-uniform range, >2=categorical list",
    )
    p.add_argument(
        "--beta_end",
        type=float,
        nargs="+",
        default=[1e-3, 5e-2],
        help="Beta end spec: 1 value=fixed, 2 values=log-uniform range, >2=categorical list",
    )
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--limit_train_batches", type=float, default=0.1)
    p.add_argument("--limit_val_batches", type=float, default=0.2)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout_sec", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def _fit_cond_scaler(df: pd.DataFrame, rows: list[int], *, tess_root: str, cond_cols: list[str], log_cols: set[str]) -> StandardScaler:
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
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    cond_cols = list(args.cond_cols)
    log_cols = set(args.log_cols or [])

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g = train_val_split_graph_ids(graph_ids, val_frac=float(args.val_frac), seed=int(args.seed))

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    cond_scaler = _fit_cond_scaler(df, train_rows, tess_root=args.tess_root, cond_cols=cond_cols, log_cols=log_cols)

    ds_train = TessellationRowDataset(
        data_csv=args.data_csv,
        tess_root=args.tess_root,
        target_cols=["RS"],
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

    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}
    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_first, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)

    k_nn_choices = [int(x) for x in list(args.k_nn)]
    if not k_nn_choices:
        raise SystemExit("--k_nn must provide at least one value")
    steps_choices = [int(x) for x in list(args.steps)]
    if not steps_choices:
        raise SystemExit("--steps must provide at least one value")
    beta_start_spec = [float(x) for x in list(args.beta_start)]
    if not beta_start_spec:
        raise SystemExit("--beta_start must provide at least one value")
    beta_end_spec = [float(x) for x in list(args.beta_end)]
    if not beta_end_spec:
        raise SystemExit("--beta_end must provide at least one value")

    def objective(trial: optuna.Trial) -> float:
        k_nn = int(trial.suggest_categorical("k_nn", k_nn_choices))
        steps = int(trial.suggest_categorical("steps", steps_choices))

        if len(beta_start_spec) == 1:
            beta_start = float(beta_start_spec[0])
        elif len(beta_start_spec) == 2:
            b0, b1 = float(beta_start_spec[0]), float(beta_start_spec[1])
            beta_start = float(trial.suggest_float("beta_start", min(b0, b1), max(b0, b1), log=True))
        else:
            beta_start = float(trial.suggest_categorical("beta_start", beta_start_spec))

        if len(beta_end_spec) == 1:
            beta_end = float(beta_end_spec[0])
        elif len(beta_end_spec) == 2:
            e0, e1 = float(beta_end_spec[0]), float(beta_end_spec[1])
            end_low = float(min(e0, e1))
            end_high = float(max(e0, e1))
            end_low_eff = float(max(end_low, beta_start * 2.0))
            if end_low_eff >= end_high:
                raise optuna.TrialPruned()
            beta_end = float(trial.suggest_float("beta_end", end_low_eff, end_high, log=True))
        else:
            valid = sorted({float(x) for x in beta_end_spec if float(x) > beta_start})
            if not valid:
                raise optuna.TrialPruned()
            beta_end = float(trial.suggest_categorical("beta_end", valid))

        schedule_cfg = DiffusionConfig(n_steps=steps, beta_start=beta_start, beta_end=beta_end)
        den_cfg = NodeDenoiserConfig(
            cond_dim=2 + len(cond_cols),
            d_h=trial.suggest_categorical("d_h", [64, 96, 128, 160, 192]),
            n_layers=trial.suggest_int("n_layers", 2, 5),
            n_rbf=trial.suggest_categorical("n_rbf", [8, 16, 24]),
            dropout=trial.suggest_float("dropout", 0.0, 0.2),
        )
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)

        lit = NodeDiffusionLitModule(
            denoiser_cfg=asdict(den_cfg),
            schedule_cfg=asdict(schedule_cfg),
            cond_cols=cond_cols,
            log_cols=sorted(log_cols),
            cond_scaler_mean=cond_scaler.mean.tolist(),
            cond_scaler_std=cond_scaler.std.tolist(),
            k_nn=k_nn,
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
        val = trainer.callback_metrics["val/loss"]
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
        ys={"val/loss": [float(v) for v in values]},
        title="Optuna tuning history (node diffusion)",
        xlabel="trial",
        ylabel="val loss",
    )

    cfg_out = {
        "task": "node_diffusion_tune",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "cond_cols": cond_cols,
        "log_cols": sorted(log_cols),
        "val_frac": float(args.val_frac),
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "search_space": {"k_nn": k_nn_choices, "steps": steps_choices, "beta_start": beta_start_spec, "beta_end": beta_end_spec},
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
