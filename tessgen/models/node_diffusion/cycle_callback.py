from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .lit_module import NodeDiffusionLitModule
from ...ckpt import EdgeBundle, NPriorBundle, NodeDiffusionBundle, SurrogateBundle
from ...cycle_eval import run_cycle_eval
from ...reporting import write_json
from ...scaler import StandardScaler


def node_bundle_from_lit(*, lit: NodeDiffusionLitModule) -> NodeDiffusionBundle:
    cond_scaler = StandardScaler(
        mean=lit.cond_scaler_mean.detach().cpu().numpy().astype(np.float32, copy=False),
        std=lit.cond_scaler_std.detach().cpu().numpy().astype(np.float32, copy=False),
    )
    return NodeDiffusionBundle(
        denoiser=lit.denoiser,
        schedule=lit.schedule,
        cond_cols=list(lit.cond_cols),
        log_cols=set(lit.log_cols),
        cond_scaler=cond_scaler,
        k_nn=int(lit.k_nn),
    )


class CycleEvalEveryEpochCallback(pl.Callback):
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        tess_root: str,
        row_indices: list[int],
        surrogate: SurrogateBundle,
        edge_bundle: EdgeBundle,
        n_prior: NPriorBundle | None,
        device: torch.device,
        out_dir_base: str,
        epoch_rows: int,
        k_best: int,
        deg_cap: int,
        min_n: int,
        max_n: int,
        n_mode: str,
        n_fixed: int,
        n_candidates: list[int],
        n_prior_samples: int,
        save_row_figs: bool,
        save_graph_files: bool,
        progress_every: int,
    ):
        super().__init__()
        self.df = df
        self.tess_root = str(tess_root)
        self.row_indices = [int(x) for x in row_indices]
        self.surrogate = surrogate
        self.edge_bundle = edge_bundle
        self.n_prior = n_prior
        self.device = torch.device(device)
        self.out_dir_base = str(out_dir_base)
        self.epoch_rows = int(epoch_rows)
        self.k_best = int(k_best)
        self.deg_cap = int(deg_cap)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.n_mode = str(n_mode)
        self.n_fixed = int(n_fixed)
        self.n_candidates = [int(x) for x in n_candidates]
        self.n_prior_samples = int(n_prior_samples)
        self.save_row_figs = bool(save_row_figs)
        self.save_graph_files = bool(save_graph_files)
        self.progress_every = int(progress_every)

        if not self.row_indices:
            raise ValueError("row_indices is empty")
        if self.epoch_rows == 0 or self.epoch_rows < -1:
            raise ValueError("epoch_rows must be -1 (all) or a positive integer")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        if not isinstance(pl_module, NodeDiffusionLitModule):
            raise TypeError("CycleEvalEveryEpochCallback requires NodeDiffusionLitModule")

        epoch = int(trainer.current_epoch) + 1
        rows = self.row_indices if self.epoch_rows == -1 else self.row_indices[: self.epoch_rows]
        out_dir = str(Path(self.out_dir_base) / f"epoch_{epoch:03d}")

        node_bundle = node_bundle_from_lit(lit=pl_module)
        cycle = run_cycle_eval(
            df=self.df,
            row_indices=[int(x) for x in rows],
            tess_root=self.tess_root,
            surrogate=self.surrogate,
            node_bundle=node_bundle,
            edge_bundle=self.edge_bundle,
            n_prior=self.n_prior,
            device=self.device,
            k_best=int(self.k_best),
            deg_cap=int(self.deg_cap),
            min_n=int(self.min_n),
            max_n=int(self.max_n),
            n_mode=str(self.n_mode),
            n_fixed=int(self.n_fixed),
            n_candidates=[int(x) for x in self.n_candidates],
            n_prior_samples=int(self.n_prior_samples),
            out_dir=out_dir,
            save_row_figs=bool(self.save_row_figs),
            save_graph_files=bool(self.save_graph_files),
            progress_every=int(self.progress_every),
            progress_prefix=f"cycle/val e{epoch}",
        )
        write_json(str(Path(out_dir) / "cycle_summary.json"), cycle)

        m = cycle["metrics"]
        r_true = float(m["true_graph"]["pearson_r"])
        r_single = float(m["single"]["pearson_r"])
        r_best = float(m["best"]["pearson_r"])

        pl_module.log("val/cycle_r_true_graph", r_true, on_step=False, on_epoch=True, prog_bar=False, logger=False, batch_size=1)
        pl_module.log("val/cycle_r_single", r_single, on_step=False, on_epoch=True, prog_bar=False, logger=False, batch_size=1)
        pl_module.log("val/cycle_r_best", r_best, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=1)
