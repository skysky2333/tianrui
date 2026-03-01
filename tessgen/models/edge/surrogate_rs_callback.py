from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .lit_module import EdgeLitModule
from ..edge_3.lit_module import Edge3LitModule
from ...ckpt import EdgeBundle, SurrogateBundle
from ...data import GraphStore, undirected_to_directed_edge_index
from ...generation import sample_edges_from_coords
from ...metrics import mae, pearson_r, rmse
from ...reporting import save_histogram_both, save_scatter_plot_both, write_json, write_jsonl
from ...transforms import invert_log_cols_torch
from ...utils import Batch


class SurrogateRSEveryEpochCallback(pl.Callback):
    """
    End-of-epoch (val) evaluation:
      coords -> sampled edges (edge model) -> surrogate -> RS

    Logs:
      - val/surrogate_rs_r: Pearson r(true RS, pred RS)
      - val/surrogate_rs_n: number of evaluated rows
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        tess_root: str,
        row_indices: list[int],
        surrogate: SurrogateBundle,
        device: torch.device,
        out_dir_base: str,
        epoch_rows: int,
        edge_thr: float,
        deg_cap: int,
    ) -> None:
        super().__init__()
        self.df = df
        self.tess_root = str(tess_root)
        self.row_indices = [int(x) for x in row_indices]
        self.surrogate = surrogate
        self.device = torch.device(device)
        self.out_dir_base = str(out_dir_base)
        self.epoch_rows = int(epoch_rows)
        self.edge_thr = float(edge_thr)
        self.deg_cap = int(deg_cap)

        if "RD" not in self.df.columns:
            raise ValueError("df is missing required column 'RD'")
        if not self.row_indices:
            raise ValueError("row_indices is empty")
        if self.epoch_rows == 0 or self.epoch_rows < -1:
            raise ValueError("epoch_rows must be -1 (all) or a positive integer")
        if not (0.0 <= self.edge_thr <= 1.0):
            raise ValueError(f"edge_thr must be in [0,1]; got {self.edge_thr}")
        if self.deg_cap <= 0:
            raise ValueError("deg_cap must be > 0")

        self.rs_col = "RS" if "RS" in self.surrogate.target_cols else self.surrogate.target_cols[0]
        self.rs_idx = int(self.surrogate.target_cols.index(self.rs_col))
        if self.rs_col not in self.df.columns:
            raise ValueError(f"df is missing required RS column {self.rs_col!r} (from surrogate target_cols)")

        self.store = GraphStore(self.tess_root)

    @staticmethod
    def _compute_rs_metrics(*, y_true: list[float], y_pred: list[float]) -> dict[str, float | int]:
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        if yt.size == 0 or yp.size == 0:
            return {
                "n": 0,
                "pearson_r": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias_mean": float("nan"),
                "bias_median": float("nan"),
                "frac_over": float("nan"),
            }
        diff = yp - yt
        return {
            "n": int(yt.size),
            "pearson_r": float(pearson_r(yt, yp)),
            "mae": float(mae(yt, yp)),
            "rmse": float(rmse(yt, yp)),
            "bias_mean": float(np.mean(diff)),
            "bias_median": float(np.median(diff)),
            "frac_over": float(np.mean(yp > yt)),
        }

    @staticmethod
    def _log10_safe(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.log10(np.clip(x, float(eps), None))

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        if isinstance(pl_module, EdgeLitModule):
            edge_bundle = EdgeBundle(
                model=pl_module.model,
                variant="edge",
                cand_mode=str(pl_module.cand_mode),
                k=int(pl_module.k),
                k_msg=None,
            )
        elif isinstance(pl_module, Edge3LitModule):
            edge_bundle = EdgeBundle(
                model=pl_module.model,
                variant="edge_3",
                cand_mode=str(pl_module.cand_mode),
                k=int(pl_module.k),
                k_msg=int(pl_module.k_msg),
            )
        else:
            raise TypeError("SurrogateRSEveryEpochCallback requires EdgeLitModule or Edge3LitModule")

        epoch = int(trainer.current_epoch) + 1
        rows = self.row_indices if self.epoch_rows == -1 else self.row_indices[: self.epoch_rows]

        self.surrogate.model.eval()
        pl_module.eval()

        y_true: list[float] = []
        y_pred_sampled: list[float] = []
        y_pred_true_edges: list[float] = []
        edges_cache: dict[int, np.ndarray] = {}
        row_records: list[dict] = []

        for row_idx in rows:
            if int(row_idx) < 0 or int(row_idx) >= int(len(self.df)):
                raise ValueError(f"Row index out of range: {row_idx} (df len={len(self.df)})")
            r = self.df.iloc[int(row_idx)]
            graph_id = (int(row_idx) // 5) + 1

            rd = float(r["RD"])
            rs_true = float(r[self.rs_col])

            g = self.store.get(int(graph_id))
            coords01_cpu = g.coords01.to(dtype=torch.float32)
            n_nodes = int(coords01_cpu.shape[0])
            n_edges_true = int(g.edges_undirected.shape[0])

            if int(graph_id) not in edges_cache:
                edges_uv = sample_edges_from_coords(
                    edge_bundle=edge_bundle,
                    coords01=coords01_cpu,
                    deg_cap=int(self.deg_cap),
                    edge_thr=float(self.edge_thr),
                    ensure_connected=True,
                    device=self.device,
                )
                edges_cache[int(graph_id)] = edges_uv
            else:
                edges_uv = edges_cache[int(graph_id)]

            coords01 = coords01_cpu.to(device=self.device)
            edge_index = undirected_to_directed_edge_index(edges_uv).to(self.device)
            batch = Batch(
                x=coords01,
                edge_index=edge_index,
                batch=torch.zeros((n_nodes,), device=self.device, dtype=torch.long),
                rd=torch.tensor([[float(rd)]], device=self.device, dtype=torch.float32),
                y=torch.zeros((1, int(len(self.surrogate.target_cols))), device=self.device, dtype=torch.float32),
                n_nodes=torch.tensor([int(n_nodes)], device=self.device, dtype=torch.long),
                n_edges=torch.tensor([int(edges_uv.shape[0])], device=self.device, dtype=torch.long),
            )

            pred_z = self.surrogate.model(batch)
            pred_t = self.surrogate.scaler.inverse_transform_torch(pred_z)
            pred_raw = invert_log_cols_torch(pred_t, self.surrogate.target_cols, self.surrogate.log_cols)
            rs_pred = float(pred_raw[0, int(self.rs_idx)].detach().cpu().item())

            y_true.append(float(rs_true))
            y_pred_sampled.append(float(rs_pred))

            # Baseline: surrogate on the true graph edges (no edge sampling)
            batch_true = Batch(
                x=coords01,
                edge_index=g.edge_index.to(device=self.device),
                batch=torch.zeros((n_nodes,), device=self.device, dtype=torch.long),
                rd=torch.tensor([[float(rd)]], device=self.device, dtype=torch.float32),
                y=torch.zeros((1, int(len(self.surrogate.target_cols))), device=self.device, dtype=torch.float32),
                n_nodes=torch.tensor([int(n_nodes)], device=self.device, dtype=torch.long),
                n_edges=torch.tensor([int(n_edges_true)], device=self.device, dtype=torch.long),
            )
            pred_z_true = self.surrogate.model(batch_true)
            pred_t_true = self.surrogate.scaler.inverse_transform_torch(pred_z_true)
            pred_raw_true = invert_log_cols_torch(pred_t_true, self.surrogate.target_cols, self.surrogate.log_cols)
            rs_pred_true = float(pred_raw_true[0, int(self.rs_idx)].detach().cpu().item())
            y_pred_true_edges.append(float(rs_pred_true))

            row_records.append(
                {
                    "row_idx": int(row_idx),
                    "graph_id": int(graph_id),
                    "rd": float(rd),
                    "rs_col": str(self.rs_col),
                    "rs_true": float(rs_true),
                    "rs_pred_sampled": float(rs_pred),
                    "rs_pred_true_edges": float(rs_pred_true),
                    "n_nodes": int(n_nodes),
                    "n_edges_true": int(n_edges_true),
                    "n_edges_sampled": int(edges_uv.shape[0]),
                }
            )

        # Metrics + artifacts
        metrics_sampled = self._compute_rs_metrics(y_true=y_true, y_pred=y_pred_sampled)
        metrics_true = self._compute_rs_metrics(y_true=y_true, y_pred=y_pred_true_edges)

        out_dir = Path(self.out_dir_base) / f"epoch_{epoch:03d}"
        rows_path = out_dir / "rows.jsonl"
        figs_dir = out_dir / "figures"
        write_jsonl(str(rows_path), row_records)

        yt = np.asarray(y_true, dtype=np.float64)
        yp_s = np.asarray(y_pred_sampled, dtype=np.float64)
        yp_t = np.asarray(y_pred_true_edges, dtype=np.float64)
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_sampled.png"),
            out_svg=str(figs_dir / "rs_scatter_sampled.svg"),
            x=yt.tolist(),
            y=yp_s.tolist(),
            title=f"{self.rs_col}: true vs predicted (edge-sampled)",
            xlabel=f"true {self.rs_col}",
            ylabel=f"pred {self.rs_col}",
        )
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_true_edges.png"),
            out_svg=str(figs_dir / "rs_scatter_true_edges.svg"),
            x=yt.tolist(),
            y=yp_t.tolist(),
            title=f"{self.rs_col}: true vs predicted (true edges baseline)",
            xlabel=f"true {self.rs_col}",
            ylabel=f"pred {self.rs_col}",
        )
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_sampled_log10.png"),
            out_svg=str(figs_dir / "rs_scatter_sampled_log10.svg"),
            x=self._log10_safe(yt).tolist(),
            y=self._log10_safe(yp_s).tolist(),
            title=f"log10({self.rs_col}): true vs predicted (edge-sampled)",
            xlabel=f"log10(true {self.rs_col})",
            ylabel=f"log10(pred {self.rs_col})",
        )
        save_scatter_plot_both(
            out_png=str(figs_dir / "rs_scatter_true_edges_log10.png"),
            out_svg=str(figs_dir / "rs_scatter_true_edges_log10.svg"),
            x=self._log10_safe(yt).tolist(),
            y=self._log10_safe(yp_t).tolist(),
            title=f"log10({self.rs_col}): true vs predicted (true edges baseline)",
            xlabel=f"log10(true {self.rs_col})",
            ylabel=f"log10(pred {self.rs_col})",
        )
        save_histogram_both(
            out_png=str(figs_dir / "rs_error_hist_sampled.png"),
            out_svg=str(figs_dir / "rs_error_hist_sampled.svg"),
            values=(yp_s - yt).tolist(),
            title=f"{self.rs_col} error (pred-true) (edge-sampled)",
            xlabel=f"{self.rs_col}_pred - {self.rs_col}_true",
            bins=80,
        )
        save_histogram_both(
            out_png=str(figs_dir / "rs_error_hist_true_edges.png"),
            out_svg=str(figs_dir / "rs_error_hist_true_edges.svg"),
            values=(yp_t - yt).tolist(),
            title=f"{self.rs_col} error (pred-true) (true edges baseline)",
            xlabel=f"{self.rs_col}_pred - {self.rs_col}_true",
            bins=80,
        )

        write_json(
            str(out_dir / "surrogate_rs_summary.json"),
            {
                "task": "surrogate_rs_eval",
                "epoch": int(epoch),
                "rs_col": str(self.rs_col),
                "n_rows": int(len(rows)),
                "pearson_r": float(metrics_sampled["pearson_r"]),
                "metrics": {
                    "sampled": metrics_sampled,
                    "true_edges": metrics_true,
                },
                "edge": {
                    "variant": str(edge_bundle.variant),
                    "cand_mode": str(edge_bundle.cand_mode),
                    "k": int(edge_bundle.k),
                    "k_msg": int(edge_bundle.k_msg) if edge_bundle.k_msg is not None else None,
                    "edge_thr": float(self.edge_thr),
                    "deg_cap": int(self.deg_cap),
                },
                "artifacts": {
                    "rows_jsonl": str(rows_path),
                    "figures_dir": str(figs_dir),
                },
            },
        )

        pl_module.log(
            "val/surrogate_rs_r",
            float(metrics_sampled["pearson_r"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            batch_size=1,
        )
        pl_module.log("val/surrogate_rs_n", float(len(rows)), on_step=False, on_epoch=True, prog_bar=False, logger=False, batch_size=1)
