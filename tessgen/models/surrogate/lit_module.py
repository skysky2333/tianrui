from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .core import SurrogateConfig, SurrogateModel
from ...transforms import apply_log_cols_torch, invert_log_cols_torch
from ...utils import Batch


class SurrogateLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        target_cols: list[str],
        log_cols: list[str],
        scaler_mean: list[float],
        scaler_std: list[float],
        lr: float,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg_obj = SurrogateConfig(**cfg)
        self.model = SurrogateModel(y_dim=len(target_cols), cfg=cfg_obj)
        self.target_cols = list(target_cols)
        self.log_cols = set(log_cols)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        mean = torch.tensor(np.array(scaler_mean, dtype=np.float32))
        std = torch.tensor(np.array(scaler_std, dtype=np.float32))
        self.register_buffer("scaler_mean", mean)
        self.register_buffer("scaler_std", std)

    @property
    def cfg(self) -> SurrogateConfig:
        return self.model.cfg

    def targets_to_z(self, y: torch.Tensor) -> torch.Tensor:
        y_t = apply_log_cols_torch(y, self.target_cols, self.log_cols)
        return (y_t - self.scaler_mean) / self.scaler_std

    def z_to_targets(self, z: torch.Tensor) -> torch.Tensor:
        y_t = z * self.scaler_std + self.scaler_mean
        return invert_log_cols_torch(y_t, self.target_cols, self.log_cols)

    def forward(self, batch: Batch) -> torch.Tensor:
        # Returns prediction in z-space (scaled transformed targets).
        y_z = self.targets_to_z(batch.y)
        pred_z = self.model(
            Batch(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                rd=batch.rd,
                y=y_z,
                n_nodes=batch.n_nodes,
                n_edges=batch.n_edges,
            )
        )
        return pred_z

    def _loss(self, batch: Batch) -> torch.Tensor:
        y_z = self.targets_to_z(batch.y)
        pred_z = self.model(
            Batch(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                rd=batch.rd,
                y=y_z,
                n_nodes=batch.n_nodes,
                n_edges=batch.n_edges,
            )
        )
        return F.mse_loss(pred_z, y_z)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch.rd.shape[0])
        self.log("train/mse_z_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=bs)
        self.log("train/mse_z", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch.rd.shape[0])
        self.log("val/mse_z", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._loss(batch)
        bs = int(batch.rd.shape[0])
        self.log("test/mse_z", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def export_surrogate_pt(
    *,
    lit: SurrogateLitModule,
    out_path: str,
    val_mse_z: float | None = None,
) -> None:
    torch.save(
        {
            "model_state": lit.model.state_dict(),
            "cfg": asdict(lit.cfg),
            "target_cols": lit.target_cols,
            "log_cols": sorted(lit.log_cols),
            "scaler_mean": lit.scaler_mean.detach().cpu().tolist(),
            "scaler_std": lit.scaler_std.detach().cpu().tolist(),
            "val_mse": float(val_mse_z) if val_mse_z is not None else None,
        },
        out_path,
    )
