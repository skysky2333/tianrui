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
from .lit_module import NodeDiffusionLitModule, export_node_diffusion_pt
from .report import eval_node_diffusion, make_report_and_figures
from ...data import (
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
from ...utils import ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train node diffusion (Lightning): (RD, metrics) -> coords")
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
    p.add_argument("--lambda_n", type=float, default=0.1)

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
    return p.parse_args()


def _fit_cond_scaler(df: pd.DataFrame, rows: list[int], cond_cols: list[str], log_cols: set[str]) -> StandardScaler:
    x = df.loc[rows, ["RD"] + cond_cols].to_numpy(dtype=np.float32)
    x_t = torch.from_numpy(x)
    x_t[:, 1:] = apply_log_cols_torch(x_t[:, 1:], cond_cols, log_cols)
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

    cond_scaler = _fit_cond_scaler(df, train_rows, cond_cols, log_cols)

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
        cond_dim=1 + len(cond_cols),
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
        lambda_n=float(args.lambda_n),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.out_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    history_path = str(Path(args.out_dir) / "history.jsonl")
    callbacks = [ckpt_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()]

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
        val_loss=float(ckpt_cb.best_model_score),
    )

    # Report + figures
    device = torch.device("cuda" if dev.accelerator == "gpu" else dev.accelerator)
    best_lit = best_lit.to(device)
    test_eval = eval_node_diffusion(lit=best_lit, dl=dl_test, max_samples=int(args.report_max_samples))
    make_report_and_figures(run_dir=args.out_dir, history_path=history_path, test_eval=test_eval)

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
        "lambda_n": float(args.lambda_n),
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
