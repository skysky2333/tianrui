from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .core import DiffusionConfig, DiffusionSchedule, NodeDenoiser, NodeDenoiserConfig, NPredictor
from ...graph_utils import knn_candidate_pairs, pairs_to_edge_index
from ...transforms import apply_log_cols_torch


class NodeDiffusionLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        denoiser_cfg: dict[str, Any],
        schedule_cfg: dict[str, Any],
        cond_cols: list[str],
        log_cols: list[str],
        cond_scaler_mean: list[float],
        cond_scaler_std: list[float],
        k_nn: int,
        lambda_n: float,
        lr: float,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        den_cfg_obj = NodeDenoiserConfig(**denoiser_cfg)
        sched_cfg_obj = DiffusionConfig(**schedule_cfg)

        self.denoiser = NodeDenoiser(den_cfg_obj)
        self.n_pred = NPredictor(cond_dim=den_cfg_obj.cond_dim, d_h=den_cfg_obj.d_h, dropout=float(den_cfg_obj.dropout))
        self.schedule = DiffusionSchedule(sched_cfg_obj)

        self.cond_cols = list(cond_cols)
        self.log_cols = set(log_cols)
        self.k_nn = int(k_nn)
        self.lambda_n = float(lambda_n)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        mean = torch.tensor(np.array(cond_scaler_mean, dtype=np.float32))
        std = torch.tensor(np.array(cond_scaler_std, dtype=np.float32))
        self.register_buffer("cond_scaler_mean", mean)
        self.register_buffer("cond_scaler_std", std)

    @property
    def denoiser_cfg(self) -> NodeDenoiserConfig:
        return self.denoiser.cfg

    def transfer_batch_to_device(self, batch, device, dataloader_idx):  # type: ignore[override]
        # Keep the raw sample on CPU; we explicitly move tensors we need.
        return batch

    def _cond_to_z(self, sample: dict) -> torch.Tensor:
        rd = sample["rd"].view(1, 1).to(self.device)  # (1,1)
        cond = sample["cond"].view(1, -1).to(self.device)  # (1,Dc)
        cond = apply_log_cols_torch(cond, self.cond_cols, self.log_cols)
        full = torch.cat([rd, cond], dim=-1)  # (1, 1+Dc)
        return ((full - self.cond_scaler_mean) / self.cond_scaler_std).squeeze(0)

    def _step(self, sample: dict, stage: str) -> torch.Tensor:
        coords0_cpu = sample["coords01"]  # (N,2) on CPU
        if coords0_cpu.device.type != "cpu":
            raise RuntimeError(f"Expected coords01 on CPU, got {coords0_cpu.device}")
        N = int(coords0_cpu.shape[0])
        cond_z = self._cond_to_z(sample)

        logN = torch.log(torch.tensor(float(N), device=self.device))
        mu, log_sigma = self.n_pred(cond_z)
        sigma2 = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (logN - mu) ** 2 / sigma2 + log_sigma

        t = torch.randint(low=0, high=self.schedule.n_steps, size=(1,), device=self.device, dtype=torch.long)
        eps_cpu = torch.randn_like(coords0_cpu)

        ab = self.schedule.alpha_bars[t].detach().cpu().view(1, 1)
        x_t_cpu = torch.sqrt(ab) * coords0_cpu + torch.sqrt(1.0 - ab) * eps_cpu
        cand = knn_candidate_pairs(x_t_cpu.numpy(), k=self.k_nn)
        edge_index = pairs_to_edge_index(cand).to(self.device)
        x_t = x_t_cpu.to(self.device)
        eps = eps_cpu.to(self.device)

        eps_pred = self.denoiser(x_t=x_t, t=t, cond=cond_z, edge_index=edge_index)
        diff_loss = torch.mean((eps_pred - eps) ** 2)
        loss = diff_loss + self.lambda_n * nll.mean()

        if stage == "train":
            self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "train"), batch_size=1)
        self.log(f"{stage}/diff_mse", diff_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log(f"{stage}/nll", nll.mean(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        return self._step(batch, "test")

    def configure_optimizers(self):  # type: ignore[override]
        params = list(self.denoiser.parameters()) + list(self.n_pred.parameters())
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def export_node_diffusion_pt(*, lit: NodeDiffusionLitModule, out_path: str, val_loss: float | None = None) -> None:
    torch.save(
        {
            "denoiser_state": lit.denoiser.state_dict(),
            "n_pred_state": lit.n_pred.state_dict(),
            "denoiser_cfg": asdict(lit.denoiser_cfg),
            "schedule_cfg": dict(lit.hparams["schedule_cfg"]),
            "cond_cols": lit.cond_cols,
            "log_cols": sorted(lit.log_cols),
            "cond_scaler_mean": lit.cond_scaler_mean.detach().cpu().tolist(),
            "cond_scaler_std": lit.cond_scaler_std.detach().cpu().tolist(),
            "k_nn": int(lit.k_nn),
            "val_loss": float(val_loss) if val_loss is not None else None,
        },
        out_path,
    )
