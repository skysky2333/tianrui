from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .core import EdgeModelConfig
from .dataset import EdgeGraphDataset
from .lit_module import EdgeLitModule, export_edge_pt
from .preview_callback import EdgePreviewEveryEpochCallback
from .report import eval_edge_model, make_report_and_figures
from ...data import collate_first, discover_graph_ids, train_val_test_split_graph_ids
from ...pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from ...pl_utils import lightning_device_from_arg
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...utils import device_from_arg, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train edge model (Lightning): coords -> edges over kNN candidates")
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--k", type=int, default=48)
    p.add_argument("--neg_ratio", type=float, default=3.0)
    p.add_argument("--thr", type=float, default=0.5, help="Threshold for P(edge) in report metrics")
    p.add_argument("--max_pairs_report", type=int, default=2_000_000, help="Max candidate pairs to evaluate for curves")

    p.add_argument("--preview_each_epoch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preview_epoch_graphs", type=int, default=10)
    p.add_argument("--preview_edge_thr", type=float, default=0.5, help="Edge probability threshold used for preview sampling")
    p.add_argument("--preview_deg_cap", type=int, default=12)
    p.add_argument("--preview_save_svg", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches", type=float, default=1.0)
    p.add_argument("--limit_test_batches", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    if not (0.0 <= float(args.preview_edge_thr) <= 1.0):
        raise SystemExit(f"--preview_edge_thr must be in [0,1]; got {args.preview_edge_thr}")

    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g, test_g = train_val_test_split_graph_ids(
        graph_ids, val_frac=float(args.val_frac), test_frac=float(args.test_frac), seed=int(args.seed)
    )

    ds_train = EdgeGraphDataset(tess_root=args.tess_root, graph_ids=train_g)
    ds_val = EdgeGraphDataset(tess_root=args.tess_root, graph_ids=val_g)
    ds_test = EdgeGraphDataset(tess_root=args.tess_root, graph_ids=test_g)

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_first, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)

    cfg = EdgeModelConfig(d_h=int(args.d_h), n_layers=int(args.n_layers), n_rbf=int(args.n_rbf), dropout=float(args.dropout))
    lit = EdgeLitModule(cfg=asdict(cfg), k=int(args.k), neg_ratio=float(args.neg_ratio), lr=float(args.lr), weight_decay=float(args.weight_decay))

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val/bce",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    history_path = str(run_dir / "history.jsonl")
    preview_cb = None
    if bool(args.preview_each_epoch):
        preview_cb = EdgePreviewEveryEpochCallback(
            tess_root=str(args.tess_root),
            graph_ids=val_g,
            out_dir_base=str(run_dir / "preview_val"),
            epoch_graphs=int(args.preview_epoch_graphs),
            edge_thr=float(args.preview_edge_thr),
            deg_cap=int(args.preview_deg_cap),
            save_svg=bool(args.preview_save_svg),
        )
    callbacks = [c for c in [ckpt_cb, preview_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()] if c is not None]

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
            }
        ),
        flush=True,
    )

    trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)

    best_path = ckpt_cb.best_model_path
    if not best_path:
        raise RuntimeError("No best checkpoint was saved. Ensure validation ran and ModelCheckpoint is monitoring a metric.")
    best_lit = EdgeLitModule.load_from_checkpoint(best_path)
    trainer.test(best_lit, dataloaders=dl_test, verbose=False)

    if ckpt_cb.best_model_score is None:
        raise RuntimeError("No best_model_score found on ModelCheckpoint callback.")
    export_edge_pt(
        lit=best_lit,
        out_path=str(run_dir / "edge_model.pt"),
        val_bce=float(ckpt_cb.best_model_score),
    )

    # Comprehensive test report + figures
    torch_device = device_from_arg(args.device)
    test_eval = eval_edge_model(
        lit=best_lit.to(torch_device),
        dl=dl_test,
        thr=float(args.thr),
        max_pairs=int(args.max_pairs_report),
    )
    make_report_and_figures(run_dir=str(run_dir), history_path=history_path, test_eval=test_eval)

    config = {
        "task": "edge_model",
        "tess_root": args.tess_root,
        "model_cfg": asdict(cfg),
        "train": {"epochs": int(args.epochs), "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "k": int(args.k),
        "neg_ratio": float(args.neg_ratio),
        "thr": float(args.thr),
        "preview": {
            "each_epoch": bool(args.preview_each_epoch),
            "epoch_graphs": int(args.preview_epoch_graphs),
            "edge_thr": float(args.preview_edge_thr),
            "deg_cap": int(args.preview_deg_cap),
            "save_svg": bool(args.preview_save_svg),
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
