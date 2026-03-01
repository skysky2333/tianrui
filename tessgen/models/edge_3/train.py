from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .core import Edge3ModelConfig
from .lit_module import Edge3LitModule, export_edge3_pt
from .preview_callback import Edge3PreviewEveryEpochCallback
from .report import eval_edge3_model, make_report_and_figures
from ..edge.dataset import EdgeGraphDataset
from ..edge.surrogate_rs_callback import SurrogateRSEveryEpochCallback
from ...ckpt import load_surrogate
from ...data import collate_first, discover_graph_ids, rows_for_graph_ids, train_val_test_split_graph_ids
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from ...pl_utils import lightning_device_from_arg
from ...utils import device_from_arg, set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train edge_3 model (Lightning): coords -> edges with learned message graph")
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

    p.add_argument("--k", type=int, default=39, help="Candidate k (used when cand_mode=knn)")
    p.add_argument("--cand_mode", type=str, default="knn", help="Candidate set construction: knn|delaunay")
    p.add_argument("--k_msg", type=int, default=12, help="kNN for message graph in learned embedding space")
    p.add_argument("--neg_ratio", type=float, default=3.0)
    p.add_argument("--thr", type=float, default=0.5, help="Threshold for P(edge) in report metrics")
    p.add_argument("--max_pairs_report", type=int, default=2_000_000, help="Max candidate pairs to evaluate for curves")

    p.add_argument("--preview_each_epoch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preview_epoch_graphs", type=int, default=10)
    p.add_argument("--preview_edge_thr", type=float, default=0.5, help="Edge probability threshold used for preview sampling")
    p.add_argument("--preview_deg_cap", type=int, default=12)
    p.add_argument("--preview_save_svg", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--cycle_surrogate_ckpt", type=str, default="", help="If set, log surrogate RS correlation on edge-sampled graphs each epoch")
    p.add_argument("--cycle_data_csv", type=str, default="data/Data_2.csv")
    p.add_argument("--cycle_epoch_rows", type=int, default=50, help="How many csv rows to evaluate per epoch (-1 = all)")
    p.add_argument("--cycle_edge_thr", type=float, default=0.5)
    p.add_argument("--cycle_deg_cap", type=int, default=12)

    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--d_search", type=int, default=16)
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
    if str(args.cand_mode) not in ("knn", "delaunay"):
        raise SystemExit(f"--cand_mode must be 'knn' or 'delaunay'; got {args.cand_mode!r}")
    if int(args.k_msg) <= 0:
        raise SystemExit(f"--k_msg must be > 0; got {args.k_msg}")
    if int(args.d_search) <= 0:
        raise SystemExit(f"--d_search must be > 0; got {args.d_search}")

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

    cfg = Edge3ModelConfig(
        d_h=int(args.d_h),
        n_layers=int(args.n_layers),
        n_rbf=int(args.n_rbf),
        d_search=int(args.d_search),
        dropout=float(args.dropout),
    )
    lit = Edge3LitModule(
        cfg=asdict(cfg),
        k=int(args.k),
        cand_mode=str(args.cand_mode),
        k_msg=int(args.k_msg),
        neg_ratio=float(args.neg_ratio),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

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
        preview_cb = Edge3PreviewEveryEpochCallback(
            tess_root=str(args.tess_root),
            graph_ids=val_g,
            out_dir_base=str(run_dir / "preview_val"),
            epoch_graphs=int(args.preview_epoch_graphs),
            edge_thr=float(args.preview_edge_thr),
            deg_cap=int(args.preview_deg_cap),
            save_svg=bool(args.preview_save_svg),
        )

    torch_device = device_from_arg(args.device)
    cycle_cb = None
    if str(args.cycle_surrogate_ckpt):
        df = pd.read_csv(str(args.cycle_data_csv))
        val_rows = rows_for_graph_ids(len(df), val_g)
        surrogate = load_surrogate(str(args.cycle_surrogate_ckpt), device=torch_device)
        cycle_cb = SurrogateRSEveryEpochCallback(
            df=df,
            tess_root=str(args.tess_root),
            row_indices=val_rows,
            surrogate=surrogate,
            device=torch_device,
            out_dir_base=str(run_dir / "surrogate_val"),
            epoch_rows=int(args.cycle_epoch_rows),
            edge_thr=float(args.cycle_edge_thr),
            deg_cap=int(args.cycle_deg_cap),
        )

    callbacks = [c for c in [ckpt_cb, preview_cb, cycle_cb, JsonlMetricsCallback(history_path), EmptyCacheCallback()] if c is not None]

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
    best_lit = Edge3LitModule.load_from_checkpoint(best_path)
    trainer.test(best_lit, dataloaders=dl_test, verbose=False)

    if ckpt_cb.best_model_score is None:
        raise RuntimeError("No best_model_score found on ModelCheckpoint callback.")
    export_edge3_pt(
        lit=best_lit,
        out_path=str(run_dir / "edge_model.pt"),
        val_bce=float(ckpt_cb.best_model_score),
    )

    test_eval = eval_edge3_model(
        lit=best_lit.to(torch_device),
        dl=dl_test,
        thr=float(args.thr),
        max_pairs=int(args.max_pairs_report),
    )
    make_report_and_figures(run_dir=str(run_dir), history_path=history_path, test_eval=test_eval)

    config = {
        "task": "edge_3_model",
        "tess_root": args.tess_root,
        "model_cfg": asdict(cfg),
        "train": {"epochs": int(args.epochs), "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
        "k": int(args.k),
        "cand_mode": str(args.cand_mode),
        "k_msg": int(args.k_msg),
        "neg_ratio": float(args.neg_ratio),
        "thr": float(args.thr),
        "preview": {
            "each_epoch": bool(args.preview_each_epoch),
            "epoch_graphs": int(args.preview_epoch_graphs),
            "edge_thr": float(args.preview_edge_thr),
            "deg_cap": int(args.preview_deg_cap),
            "save_svg": bool(args.preview_save_svg),
        },
        "cycle": {
            "surrogate_ckpt": str(args.cycle_surrogate_ckpt) if str(args.cycle_surrogate_ckpt) else None,
            "data_csv": str(args.cycle_data_csv) if str(args.cycle_surrogate_ckpt) else None,
            "epoch_rows": int(args.cycle_epoch_rows) if str(args.cycle_surrogate_ckpt) else None,
            "edge_thr": float(args.cycle_edge_thr) if str(args.cycle_surrogate_ckpt) else None,
            "deg_cap": int(args.cycle_deg_cap) if str(args.cycle_surrogate_ckpt) else None,
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
