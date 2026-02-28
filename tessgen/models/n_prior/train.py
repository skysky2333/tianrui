from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .core import NPriorConfig
from .dataset import NPriorRowDataset
from .lit_module import NPriorLitModule, export_n_prior_pt
from .report import make_report_and_figures, predict_on_loader
from ...data import discover_graph_ids, rows_for_graph_ids, train_val_test_split_graph_ids
from ...pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from ...pl_utils import lightning_device_from_arg
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...scaler import StandardScaler
from ...transforms import apply_log_cols_torch
from ...utils import device_from_arg, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train N prior (Lightning): (RD, metrics) -> log(N)")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--cond_cols", type=str, nargs="+", required=True)
    p.add_argument("--log_cols", type=str, nargs="*", default=["RS"])

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--sigma_min", type=float, default=0.7)

    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--limit_test_batches", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
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
    train_g, val_g, test_g = train_val_test_split_graph_ids(
        graph_ids,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    df = pd.read_csv(args.data_csv)
    train_rows = rows_for_graph_ids(len(df), train_g)
    val_rows = rows_for_graph_ids(len(df), val_g)
    test_rows = rows_for_graph_ids(len(df), test_g)

    scaler = _fit_x_scaler(df, train_rows, cond_cols, log_cols)

    ds_train = NPriorRowDataset(data_csv=args.data_csv, tess_root=args.tess_root, cond_cols=cond_cols, row_indices=train_rows)
    ds_val = NPriorRowDataset(data_csv=args.data_csv, tess_root=args.tess_root, cond_cols=cond_cols, row_indices=val_rows)
    ds_test = NPriorRowDataset(data_csv=args.data_csv, tess_root=args.tess_root, cond_cols=cond_cols, row_indices=test_rows)

    dl_train = DataLoader(ds_train, batch_size=int(args.batch_size), shuffle=True, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, **dl_kwargs)

    cfg = NPriorConfig(
        d_h=int(args.d_h),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        sigma_min=float(args.sigma_min),
    )
    lit = NPriorLitModule(
        cfg=asdict(cfg),
        cond_cols=cond_cols,
        log_cols=sorted(log_cols),
        scaler_mean=scaler.mean.tolist(),
        scaler_std=scaler.std.tolist(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    history_path = str(run_dir / "history.jsonl")
    callbacks = [ckpt_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()]

    t0 = time.time()
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
    best_lit = NPriorLitModule.load_from_checkpoint(best_path)
    trainer.test(best_lit, dataloaders=dl_test, verbose=False)

    if ckpt_cb.best_model_score is None:
        raise RuntimeError("No best_model_score found on ModelCheckpoint callback.")
    export_n_prior_pt(
        lit=best_lit,
        out_path=str(run_dir / "n_prior.pt"),
        val_nll=float(ckpt_cb.best_model_score),
    )

    torch_device = device_from_arg(args.device)
    logn_true, mu, log_sigma = predict_on_loader(lit=best_lit, dl=dl_test, device=torch_device)
    make_report_and_figures(run_dir=str(run_dir), history_path=history_path, logn_true=logn_true, mu=mu, log_sigma=log_sigma)

    config = {
        "task": "n_prior",
        "data_csv": args.data_csv,
        "tess_root": args.tess_root,
        "cond_cols": cond_cols,
        "log_cols": sorted(log_cols),
        "model_cfg": asdict(cfg),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "splits": {"val_frac": float(args.val_frac), "test_frac": float(args.test_frac)},
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "elapsed_sec": float(time.time() - t0),
        "best_ckpt": best_path,
        "base_out_dir": str(base_dir),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
    }
    with open(str(run_dir / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(f"saved_run_dir: {run_dir}")
    print(f"saved_base_dir: {base_dir} (latest files copied here)")


if __name__ == "__main__":
    main()
