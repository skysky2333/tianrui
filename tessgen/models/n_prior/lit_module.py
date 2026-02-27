from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from .core import NPriorConfig, NPriorModel
from ...transforms import apply_log_cols_torch


class NPriorLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        cond_cols: list[str],
        log_cols: list[str],
        scaler_mean: list[float],
        scaler_std: list[float],
        lr: float,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg_obj = NPriorConfig(**cfg)
        self.model = NPriorModel(x_dim=1 + len(cond_cols), cfg=cfg_obj)

        self.cond_cols = list(cond_cols)
        self.log_cols = set(log_cols)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        mean = torch.tensor(np.array(scaler_mean, dtype=np.float32))
        std = torch.tensor(np.array(scaler_std, dtype=np.float32))
        self.register_buffer("scaler_mean", mean)
        self.register_buffer("scaler_std", std)

    @property
    def cfg(self) -> NPriorConfig:
        return self.model.cfg

    def inputs_to_z(self, *, rd: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        rd = rd.view(rd.shape[0], 1)
        cond = cond.view(cond.shape[0], -1)
        cond_t = apply_log_cols_torch(cond, self.cond_cols, self.log_cols)
        x = torch.cat([rd, cond_t], dim=-1)
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, *, rd: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_z = self.inputs_to_z(rd=rd, cond=cond)
        return self.model(x_z)

    def _loss(self, batch: dict) -> torch.Tensor:
        rd = batch["rd"]
        cond = batch["cond"]
        logn = batch["logn"].view(-1)
        mu, log_sigma = self.forward(rd=rd, cond=cond)
        sigma2 = torch.exp(2.0 * log_sigma)
        nll = 0.5 * (logn - mu) ** 2 / sigma2 + log_sigma
        return torch.mean(nll)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch["rd"].shape[0])
        self.log("train/nll_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=bs)
        self.log("train/nll", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch["rd"].shape[0])
        self.log("val/nll", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch["rd"].shape[0])
        self.log("test/nll", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def export_n_prior_pt(*, lit: NPriorLitModule, out_path: str, val_nll: float | None = None) -> None:
    torch.save(
        {
            "model_state": lit.model.state_dict(),
            "cfg": asdict(lit.cfg),
            "cond_cols": lit.cond_cols,
            "log_cols": sorted(lit.log_cols),
            "scaler_mean": lit.scaler_mean.detach().cpu().tolist(),
            "scaler_std": lit.scaler_std.detach().cpu().tolist(),
            "val_nll": float(val_nll) if val_nll is not None else None,
        },
        out_path,
    )

