from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from tessgen.outdirs import finalize_out_dir, make_timestamped_run_dir
from tessgen.pl_callbacks import EmptyCacheCallback, JsonlMetricsCallback
from tessgen.pl_utils import lightning_device_from_arg
from tessgen.utils import set_seed

from ..checkpoint import save_approach_artifacts
from ..data import GraphDataset, GraphStore, collate_first, discover_graph_ids, read_seed_n_nodes, read_seed_txt, split_graph_ids
from ..models.common import DiffusionConfig, GraphModelConfig
from ..models.diffusion_critic import DiffusionCriticLitModule
from ..models.hybrid_gan import HybridGANLitModule
from ..report import make_training_report
from ..viz import save_graph_grid
from ..graph_ops import candidate_pairs_from_xyr


class ProgressPrintCallback(pl.Callback):
    def __init__(self, *, every_n_steps: int):
        super().__init__()
        self.every_n_steps = int(every_n_steps)
        self._t0 = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        self._t0 = time.time()

    def on_train_batch_start(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        if self.every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step % self.every_n_steps != 0:
            return
        n_nodes = int(batch.get("n_nodes", -1)) if isinstance(batch, dict) else -1
        n_edges = int(batch.get("n_edges", -1)) if isinstance(batch, dict) else -1
        print(
            json.dumps(
                {
                    "event": "train_batch_start",
                    "epoch": int(trainer.current_epoch),
                    "global_step": step,
                    "batch_idx": int(batch_idx),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                }
            ),
            flush=True,
        )

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.every_n_steps != 0:
            return
        elapsed = time.time() - self._t0
        metrics = trainer.callback_metrics
        names = ["train/g_loss", "train/d_loss", "train/seed_loss", "train/critic_loss", "train/diff_loss", "train/edge_loss"]
        parts = []
        for name in names:
            val = metrics.get(name)
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                parts.append(f"{name}={float(val.detach().cpu()):.4g}")
        print(
            json.dumps(
                {
                    "event": "train_progress",
                    "epoch": int(trainer.current_epoch),
                    "global_step": step,
                    "batch_idx": int(batch_idx),
                    "elapsed_sec": round(elapsed, 1),
                    "metrics": " ".join(parts),
                }
            ),
            flush=True,
        )

    def on_validation_batch_start(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.every_n_steps <= 0:
            return
        if batch_idx % self.every_n_steps != 0:
            return
        n_nodes = int(batch.get("n_nodes", -1)) if isinstance(batch, dict) else -1
        n_edges = int(batch.get("n_edges", -1)) if isinstance(batch, dict) else -1
        print(
            json.dumps(
                {
                    "event": "validation_batch_start",
                    "stage": "sanity_check" if trainer.sanity_checking else "validation",
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "batch_idx": int(batch_idx),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                }
            ),
            flush=True,
        )


class PreviewEveryNStepsCallback(pl.Callback):
    def __init__(
        self,
        *,
        data_root: str,
        graph_ids: list[int],
        out_dir: str | Path,
        every_n_steps: int,
        n_graphs: int,
        edge_threshold: float,
        sample_steps: int,
        show_edges: bool,
    ):
        super().__init__()
        self.store = GraphStore(data_root)
        self.graph_ids = list(map(int, graph_ids[: max(0, int(n_graphs))]))
        self.out_dir = Path(out_dir)
        self.every_n_steps = int(every_n_steps)
        self.edge_threshold = float(edge_threshold)
        self.sample_steps = int(sample_steps)
        self.show_edges = bool(show_edges)
        self._batches_seen = 0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        self._batches_seen = 0

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._batches_seen += 1
        if self.every_n_steps <= 0 or not self.graph_ids:
            return
        if self._batches_seen % self.every_n_steps != 0:
            return
        optimizer_step = int(trainer.global_step)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        was_training = bool(pl_module.training)
        pl_module.eval()
        graphs = []
        with torch.no_grad():
            for gid in self.graph_ids:
                real = self.store.get(gid)
                real_xyr = real.xyr.detach().cpu().numpy()
                if bool(getattr(pl_module.hparams, "learn_edges", False)):
                    real_edges = real.edges_undirected.detach().cpu().numpy()
                else:
                    real_edges = candidate_pairs_from_xyr(
                        real.xyr,
                        candidate_mode=str(pl_module.hparams.candidate_mode),
                        k=int(pl_module.hparams.k_edge),
                    )
                graphs.append((real_xyr, real_edges, f"real {gid}"))
                try:
                    fake_xyr, fake_edges = pl_module.sample_graph(
                        n_nodes=real.n_nodes,
                        edge_threshold=self.edge_threshold,
                        sample_steps=self.sample_steps,
                    )
                except TypeError:
                    fake_xyr, fake_edges = pl_module.sample_graph(
                        n_nodes=real.n_nodes,
                        edge_threshold=self.edge_threshold,
                    )
                graphs.append(
                    (
                        fake_xyr.detach().cpu().numpy(),
                        fake_edges.detach().cpu().numpy(),
                        f"generated iter {self._batches_seen}",
                    )
                )
        save_path = self.out_dir / f"iter_{self._batches_seen:09d}.png"
        save_graph_grid(graphs=graphs, out_png=str(save_path), cols=2, show_edges=self.show_edges)
        shutil.copy2(save_path, self.out_dir / "latest.png")
        meta = {
            "iteration": int(self._batches_seen),
            "optimizer_global_step": optimizer_step,
            "path": str(save_path),
            "show_edges": self.show_edges,
            "edge_source": "Connection_*.txt" if bool(getattr(pl_module.hparams, "learn_edges", False)) else "helper geometry kNN/candidates",
        }
        (self.out_dir / "latest.json").write_text(json.dumps(meta, indent=2) + "\n")
        if was_training:
            pl_module.train()


def _limit_value(s: str) -> int | float:
    if "." not in s and "e" not in s.lower():
        return int(s)
    return float(s)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train modular graph generators on Seed/Connection graph pairs")
    p.add_argument("--approach", choices=["hybrid_gan", "diffusion_critic"], default="hybrid_gan")
    p.add_argument("--data_root", type=str, default="data/data_for_gan")
    p.add_argument("--out_dir", type=str, default="runs/graph_gen")
    p.add_argument("--max_graphs", type=int, default=0, help="0 = use all matched graph IDs")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--max_nodes_per_graph",
        type=int,
        default=0,
        help="0 = full graphs. Set >0 to train on random induced subgraphs for CPU debugging only.",
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_h", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_rbf", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--k_msg", type=int, default=12)
    p.add_argument("--k_edge", type=int, default=24)
    p.add_argument("--candidate_mode", choices=["knn", "delaunay"], default="knn")

    p.add_argument("--lambda_stats", type=float, default=10.0)
    p.add_argument("--lambda_edge", type=float, default=0.0)
    p.add_argument("--edge_temperature", type=float, default=1.0)
    p.add_argument("--edge_threshold", type=float, default=0.5)
    p.add_argument(
        "--learn_edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat Connection_*.txt as a learned target. Default false because current connections are helper-only.",
    )
    p.add_argument("--g_steps_per_batch", type=int, default=1, help="Hybrid GAN: generator updates per graph batch")
    p.add_argument("--d_every_n_steps", type=int, default=1, help="Hybrid GAN: update discriminator every N graph batches")
    p.add_argument("--real_label_smooth", type=float, default=0.9, help="Hybrid GAN: real target label for discriminator")
    p.add_argument("--fake_label_smooth", type=float, default=0.0, help="Hybrid GAN: fake target label for discriminator")
    p.add_argument("--instance_noise", type=float, default=0.0, help="Hybrid GAN: Gaussian noise added to discriminator x/y/r inputs")
    p.add_argument("--lambda_adv", type=float, default=1.0, help="Hybrid GAN: adversarial loss weight after generator pretraining")
    p.add_argument("--lambda_seed", type=float, default=50.0, help="Hybrid GAN: seed point/radius distribution loss weight")
    p.add_argument("--seed_mmd_points", type=int, default=256, help="Hybrid GAN: sampled points per graph for seed MMD/stat loss")
    p.add_argument("--g_pretrain_steps", type=int, default=5000, help="Hybrid GAN: train generator with seed/stat losses before adversarial updates")
    p.add_argument("--d_loss_floor", type=float, default=0.05, help="Hybrid GAN: skip discriminator update when D loss is already below this")

    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)
    p.add_argument("--critic_sample_steps", type=int, default=8)
    p.add_argument("--sample_steps", type=int, default=0, help="Diffusion report generation steps; 0 = all diffusion steps")

    p.add_argument("--stats_graphs", type=int, default=512, help="Bounded count used to estimate radius scale; 0 = all train graphs")
    p.add_argument("--n_values_graphs", type=int, default=0, help="Bounded count of train graphs used for generation N distribution; 0 = all")
    p.add_argument("--report_max_graphs", type=int, default=16)
    p.add_argument("--limit_train_batches", type=_limit_value, default=1.0)
    p.add_argument("--limit_val_batches", type=_limit_value, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--progress_every_n_steps", type=int, default=0, help="Print JSON progress to stdout every N train steps; 0 disables")
    p.add_argument("--preview_every_n_steps", type=int, default=1000, help="Save real-vs-generated preview image every N training steps; 0 disables")
    p.add_argument("--preview_graphs", type=int, default=4, help="Number of fixed validation graphs to compare in each preview")
    p.add_argument("--preview_show_edges", action=argparse.BooleanOptionalAction, default=False, help="Draw helper/learned edges in preview images")
    return p.parse_args()


def _estimate_r_scale(data_root: str, ids: list[int], *, max_graphs: int) -> float:
    if int(max_graphs) > 0:
        ids = ids[: int(max_graphs)]
    root = Path(data_root)
    vals: list[float] = []
    for gid in ids:
        xyr = read_seed_txt(root / f"Seed_{int(gid)}.txt")
        vals.extend(xyr[:, 2].astype(np.float64).tolist())
    if not vals:
        return 0.05
    p99 = float(np.percentile(np.array(vals, dtype=np.float64), 99.0))
    return max(1e-4, min(0.25, 1.5 * p99))


def _sample_n_values(data_root: str, ids: list[int], *, max_graphs: int) -> list[int]:
    if int(max_graphs) > 0:
        ids = ids[: int(max_graphs)]
    root = Path(data_root)
    vals = [read_seed_n_nodes(root / f"Seed_{int(gid)}.txt") for gid in ids]
    if not vals:
        raise ValueError("No training node counts were available for generation")
    return vals


def _make_lit(args: argparse.Namespace, *, r_scale: float):
    model_cfg = GraphModelConfig(
        d_h=int(args.d_h),
        n_layers=int(args.n_layers),
        n_rbf=int(args.n_rbf),
        dropout=float(args.dropout),
        z_dim=int(args.z_dim),
    )
    model_cfg_d = asdict(model_cfg)
    if args.approach == "hybrid_gan":
        return HybridGANLitModule(
            model_cfg=model_cfg_d,
            k_msg=int(args.k_msg),
            k_edge=int(args.k_edge),
            candidate_mode=str(args.candidate_mode),
            r_scale=float(r_scale),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d),
            weight_decay=float(args.weight_decay),
            lambda_stats=float(args.lambda_stats),
            edge_temperature=float(args.edge_temperature),
            g_steps_per_batch=int(args.g_steps_per_batch),
            d_every_n_steps=int(args.d_every_n_steps),
            real_label_smooth=float(args.real_label_smooth),
            fake_label_smooth=float(args.fake_label_smooth),
            instance_noise=float(args.instance_noise),
            learn_edges=bool(args.learn_edges),
            lambda_adv=float(args.lambda_adv),
            lambda_seed=float(args.lambda_seed),
            seed_mmd_points=int(args.seed_mmd_points),
            g_pretrain_steps=int(args.g_pretrain_steps),
            d_loss_floor=float(args.d_loss_floor),
        )
    diffusion_cfg = DiffusionConfig(
        n_steps=int(args.diffusion_steps),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
    )
    return DiffusionCriticLitModule(
        model_cfg=model_cfg_d,
        diffusion_cfg=asdict(diffusion_cfg),
        k_msg=int(args.k_msg),
        k_edge=int(args.k_edge),
        candidate_mode=str(args.candidate_mode),
        r_scale=float(r_scale),
        lr_g=float(args.lr_g),
        lr_d=float(args.lr_d),
        weight_decay=float(args.weight_decay),
        lambda_edge=float(args.lambda_edge),
        critic_sample_steps=int(args.critic_sample_steps),
        learn_edges=bool(args.learn_edges),
    )


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    set_seed(int(args.seed))
    pl.seed_everything(int(args.seed), workers=True)

    print(json.dumps({"event": "discover_start", "data_root": args.data_root, "max_graphs": int(args.max_graphs)}), flush=True)
    ids = discover_graph_ids(args.data_root, max_graphs=int(args.max_graphs))
    print(json.dumps({"event": "discover_done", "matched_graphs": len(ids), "elapsed_sec": round(time.time() - t0, 3)}), flush=True)
    if len(ids) < 3:
        raise SystemExit(f"Need at least 3 matched graph pairs for train/val/test splitting; found {len(ids)}")
    train_ids, val_ids, test_ids = split_graph_ids(
        ids,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    if not train_ids or not val_ids:
        raise SystemExit(f"Split produced train={len(train_ids)} val={len(val_ids)}; increase --max_graphs or reduce split fractions")

    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    stats_count = len(train_ids) if int(args.stats_graphs) <= 0 else min(len(train_ids), int(args.stats_graphs))
    print(
        json.dumps(
            {
                "event": "stats_start",
                "stats_graphs": stats_count,
                "run_dir": str(run_dir),
            }
        ),
        flush=True,
    )
    r_scale = _estimate_r_scale(args.data_root, train_ids, max_graphs=int(args.stats_graphs))
    print(json.dumps({"event": "stats_done", "r_scale": r_scale, "elapsed_sec": round(time.time() - t0, 3)}), flush=True)
    lit = _make_lit(args, r_scale=r_scale)

    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": dev.accelerator == "gpu",
    }
    dl_train = DataLoader(
        GraphDataset(
            data_root=args.data_root,
            graph_ids=train_ids,
            max_nodes_per_graph=int(args.max_nodes_per_graph),
            read_connections=bool(args.learn_edges),
        ),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_first,
        **dl_kwargs,
    )
    dl_val = DataLoader(
        GraphDataset(
            data_root=args.data_root,
            graph_ids=val_ids,
            max_nodes_per_graph=int(args.max_nodes_per_graph),
            read_connections=bool(args.learn_edges),
        ),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_first,
        **dl_kwargs,
    )

    history_path = run_dir / "history.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_cb = ModelCheckpoint(dirpath=str(ckpt_dir), filename="best", monitor="val/loss", mode="min", save_top_k=1, save_last=True)
    callbacks = [
        ckpt_cb,
        JsonlMetricsCallback(str(history_path)),
        EmptyCacheCallback(),
        ProgressPrintCallback(every_n_steps=int(args.progress_every_n_steps)),
    ]
    if int(args.preview_every_n_steps) > 0 and int(args.preview_graphs) > 0:
        callbacks.append(
            PreviewEveryNStepsCallback(
                data_root=str(args.data_root),
                graph_ids=val_ids,
                out_dir=run_dir / "previews",
                every_n_steps=int(args.preview_every_n_steps),
                n_graphs=int(args.preview_graphs),
                edge_threshold=float(args.edge_threshold),
                sample_steps=int(args.sample_steps),
                show_edges=bool(args.preview_show_edges),
            )
        )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning", version="")
    trainer = pl.Trainer(
        max_epochs=int(args.epochs),
        accelerator=dev.accelerator,
        devices=dev.devices,
        default_root_dir=str(run_dir),
        logger=csv_logger,
        callbacks=callbacks,
        log_every_n_steps=int(args.log_every_n_steps),
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
    )

    config: dict[str, Any] = vars(args).copy()
    train_batches_per_epoch = len(train_ids)
    if isinstance(args.limit_train_batches, int):
        train_batches_per_epoch = min(train_batches_per_epoch, int(args.limit_train_batches))
    elif isinstance(args.limit_train_batches, float) and float(args.limit_train_batches) < 1.0:
        train_batches_per_epoch = max(1, int(train_batches_per_epoch * float(args.limit_train_batches)))
    config.update(
        {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "splits": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
            "r_scale": float(r_scale),
            "device": {"accelerator": dev.accelerator, "devices": dev.devices},
            "lightning_log_dir": str(csv_logger.log_dir),
            "checkpoint_dir": str(ckpt_dir),
            "max_nodes_per_graph": int(args.max_nodes_per_graph),
            "preview_dir": str(run_dir / "previews") if int(args.preview_every_n_steps) > 0 else None,
        }
    )
    print(
        json.dumps(
            {
                "event": "train_plan",
                "train_graphs": len(train_ids),
                "val_graphs": len(val_ids),
                "epochs": int(args.epochs),
                "estimated_train_batches_per_epoch": train_batches_per_epoch,
                "estimated_train_steps_total": train_batches_per_epoch * int(args.epochs),
                "max_nodes_per_graph": int(args.max_nodes_per_graph),
                "progress_every_n_steps": int(args.progress_every_n_steps),
                "preview_every_n_steps": int(args.preview_every_n_steps),
                "preview_graphs": int(args.preview_graphs),
                "preview_show_edges": bool(args.preview_show_edges),
                "learn_edges": bool(args.learn_edges),
                "lightning_log_dir": str(csv_logger.log_dir),
            }
        ),
        flush=True,
    )
    print(json.dumps({"run_dir": str(run_dir), "approach": args.approach, "splits": config["splits"], "r_scale": r_scale}), flush=True)

    trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)

    if ckpt_cb.best_model_path:
        shutil.copy2(ckpt_cb.best_model_path, run_dir / "best.ckpt")
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        shutil.copy2(last_ckpt, run_dir / "last.ckpt")

    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_score = None if ckpt_cb.best_model_score is None else float(ckpt_cb.best_model_score.detach().cpu())
    print(
        json.dumps(
            {
                "event": "export_start",
                "n_values_graphs": len(train_ids) if int(args.n_values_graphs) <= 0 else min(len(train_ids), int(args.n_values_graphs)),
            }
        ),
        flush=True,
    )
    n_values = _sample_n_values(args.data_root, train_ids, max_graphs=int(args.n_values_graphs))
    payload = lit.export_payload(config=config, n_values=n_values, best_val_loss=best_score)
    save_approach_artifacts(payload, run_dir=run_dir)
    report_ids = test_ids if test_ids else val_ids
    report = make_training_report(
        lit,
        run_dir=run_dir,
        data_root=args.data_root,
        real_graph_ids=report_ids,
        n_values=payload["n_values"],
        edge_threshold=float(args.edge_threshold),
        sample_steps=int(args.sample_steps),
        report_max_graphs=int(args.report_max_graphs),
        num_workers=0,
        read_connections=bool(args.learn_edges),
    )
    print(json.dumps({"report": report, "elapsed_sec": round(time.time() - t0, 3)}, indent=2), flush=True)
    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)


if __name__ == "__main__":
    main()
