from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from ...ckpt import EdgeBundle
from ...constants import COORD_MIN, COORD_RANGE
from ...data import GraphStore
from ...generation import sample_edges_from_coords
from ...reporting import write_json, write_jsonl
from ...viz import save_graph_figure
from .lit_module import Edge3LitModule


class Edge3PreviewEveryEpochCallback(pl.Callback):
    """
    Save qualitative edge_3 examples at the end of every validation epoch.

    Outputs under: out_dir_base/epoch_###/
    """

    def __init__(
        self,
        *,
        tess_root: str,
        graph_ids: list[int],
        out_dir_base: str,
        epoch_graphs: int = 10,
        edge_thr: float = 0.5,
        deg_cap: int = 12,
        save_svg: bool = True,
    ) -> None:
        super().__init__()
        self.tess_root = str(tess_root)
        self.graph_ids = [int(x) for x in graph_ids]
        self.out_dir_base = str(out_dir_base)
        self.epoch_graphs = int(epoch_graphs)
        self.edge_thr = float(edge_thr)
        self.deg_cap = int(deg_cap)
        self.save_svg = bool(save_svg)

        if not self.graph_ids:
            raise ValueError("graph_ids is empty")
        if self.epoch_graphs <= 0:
            raise ValueError("epoch_graphs must be > 0")
        if not (0.0 <= self.edge_thr <= 1.0):
            raise ValueError(f"edge_thr must be in [0,1]; got {self.edge_thr}")
        if self.deg_cap <= 0:
            raise ValueError("deg_cap must be > 0")

        self.preview_graph_ids = self.graph_ids[: min(self.epoch_graphs, len(self.graph_ids))]
        self.store = GraphStore(self.tess_root)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        if not isinstance(pl_module, Edge3LitModule):
            raise TypeError("Edge3PreviewEveryEpochCallback requires Edge3LitModule")

        epoch = int(trainer.current_epoch) + 1
        out_dir = Path(self.out_dir_base) / f"epoch_{epoch:03d}"
        graphs_true_dir = out_dir / "graphs_true"
        graphs_pred_dir = out_dir / "graphs_pred"
        rows_path = out_dir / "rows.jsonl"
        summary_path = out_dir / "preview_summary.json"

        rows: list[dict] = []
        deg_true_all: list[float] = []
        deg_pred_all: list[float] = []
        ratio_all: list[float] = []

        device = pl_module.device
        for gid in self.preview_graph_ids:
            g = self.store.get(int(gid))
            n_nodes = int(g.n_nodes)
            n_edges_true = int(g.n_edges)
            mean_deg_true = (2.0 * float(n_edges_true)) / float(max(1, n_nodes))

            coords = g.coords01.detach().cpu().numpy() * COORD_RANGE + COORD_MIN
            coords01 = g.coords01.to(device=device, dtype=torch.float32)

            edge_bundle = EdgeBundle(
                model=pl_module.model,
                variant="edge_3",
                cand_mode=str(pl_module.cand_mode),
                k=int(pl_module.k),
                k_msg=int(pl_module.k_msg),
            )
            edges_pred = sample_edges_from_coords(
                edge_bundle=edge_bundle,
                coords01=coords01,
                deg_cap=int(self.deg_cap),
                edge_thr=float(self.edge_thr),
                ensure_connected=True,
                device=device,
            )
            n_edges_pred = int(edges_pred.shape[0])
            mean_deg_pred = (2.0 * float(n_edges_pred)) / float(max(1, n_nodes))
            ratio = float(n_edges_pred) / float(max(1, n_edges_true))

            true_png = graphs_true_dir / f"graph_gid{gid}.png"
            true_svg = graphs_true_dir / f"graph_gid{gid}.svg"
            pred_png = graphs_pred_dir / f"graph_gid{gid}.png"
            pred_svg = graphs_pred_dir / f"graph_gid{gid}.svg"

            save_graph_figure(
                coords=coords,
                edges_uv=g.edges_undirected.detach().cpu().numpy(),
                out_png=str(true_png),
                out_svg=str(true_svg) if self.save_svg else "",
                title=f"true | gid={gid} N={n_nodes} E={n_edges_true} mean_deg={mean_deg_true:.3g}",
            )
            save_graph_figure(
                coords=coords,
                edges_uv=edges_pred,
                out_png=str(pred_png),
                out_svg=str(pred_svg) if self.save_svg else "",
                title=(
                    f"pred | gid={gid} thr={self.edge_thr:.3g} deg_cap={self.deg_cap} "
                    f"N={n_nodes} E={n_edges_pred} mean_deg={mean_deg_pred:.3g}"
                ),
            )

            row = {
                "epoch": int(epoch),
                "graph_id": int(gid),
                "n_nodes": int(n_nodes),
                "n_edges_true": int(n_edges_true),
                "n_edges_pred": int(n_edges_pred),
                "mean_deg_true": float(mean_deg_true),
                "mean_deg_pred": float(mean_deg_pred),
                "edge_ratio": float(ratio),
                "edge_thr": float(self.edge_thr),
                "deg_cap": int(self.deg_cap),
                "figures": {
                    "true_png": str(true_png),
                    "true_svg": str(true_svg) if self.save_svg else None,
                    "pred_png": str(pred_png),
                    "pred_svg": str(pred_svg) if self.save_svg else None,
                },
            }
            rows.append(row)
            deg_true_all.append(float(mean_deg_true))
            deg_pred_all.append(float(mean_deg_pred))
            ratio_all.append(float(ratio))

        write_jsonl(str(rows_path), rows)

        summary = {
            "task": "edge_3_preview",
            "epoch": int(epoch),
            "tess_root": self.tess_root,
            "graphs": {"n": int(len(rows)), "graph_ids": [int(x) for x in self.preview_graph_ids]},
            "params": {
                "edge_thr": float(self.edge_thr),
                "deg_cap": int(self.deg_cap),
                "k": int(pl_module.k),
                "k_msg": int(pl_module.k_msg),
                "cand_mode": str(pl_module.cand_mode),
            },
            "stats": {
                "mean_deg_true": {"mean": float(np.mean(deg_true_all)), "median": float(np.median(deg_true_all))},
                "mean_deg_pred": {"mean": float(np.mean(deg_pred_all)), "median": float(np.median(deg_pred_all))},
                "edge_ratio": {"mean": float(np.mean(ratio_all)), "median": float(np.median(ratio_all))},
            },
            "artifacts": {
                "rows_jsonl": str(rows_path),
                "graphs_true_dir": str(graphs_true_dir),
                "graphs_pred_dir": str(graphs_pred_dir),
            },
        }
        write_json(str(summary_path), summary)

        pl_module.log(
            "val/preview_mean_deg_true",
            float(summary["stats"]["mean_deg_true"]["mean"]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            batch_size=1,
        )
        pl_module.log(
            "val/preview_mean_deg_pred",
            float(summary["stats"]["mean_deg_pred"]["mean"]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            batch_size=1,
        )
        pl_module.log(
            "val/preview_edge_ratio_mean",
            float(summary["stats"]["edge_ratio"]["mean"]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            batch_size=1,
        )

        with open(str(Path(self.out_dir_base) / "preview_history.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "edge_thr": float(self.edge_thr),
                        "deg_cap": int(self.deg_cap),
                        "k": int(pl_module.k),
                        "k_msg": int(pl_module.k_msg),
                        "cand_mode": str(pl_module.cand_mode),
                        "val/preview_mean_deg_true": float(summary["stats"]["mean_deg_true"]["mean"]),
                        "val/preview_mean_deg_pred": float(summary["stats"]["mean_deg_pred"]["mean"]),
                        "val/preview_edge_ratio_mean": float(summary["stats"]["edge_ratio"]["mean"]),
                    }
                )
                + "\n"
            )
