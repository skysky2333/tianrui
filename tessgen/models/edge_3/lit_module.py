from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .core import Edge3Model, Edge3ModelConfig, label_candidate_pairs
from ...graph_utils import candidate_pairs, knn_candidate_pairs, pairs_to_edge_index


class Edge3LitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        k: int,
        cand_mode: str,
        k_msg: int,
        neg_ratio: float,
        lr: float,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        cfg_obj = Edge3ModelConfig(**cfg)
        self.model = Edge3Model(cfg=cfg_obj)
        self.k = int(k)
        self.cand_mode = str(cand_mode)
        self.k_msg = int(k_msg)
        self.neg_ratio = float(neg_ratio)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def cfg(self) -> Edge3ModelConfig:
        return self.model.cfg

    def transfer_batch_to_device(self, batch, device, dataloader_idx):  # type: ignore[override]
        return batch

    def _graph_loss(self, sample: dict, *, neg_subsample: bool) -> torch.Tensor:
        coords01_cpu = sample["coords01"]
        if coords01_cpu.device.type != "cpu":
            raise RuntimeError(f"Expected coords01 on CPU, got {coords01_cpu.device}")
        true_edges = sample["edges_undirected"].numpy()
        if sample["edges_undirected"].device.type != "cpu":
            raise RuntimeError(f"Expected edges_undirected on CPU, got {sample['edges_undirected'].device}")

        cand = candidate_pairs(coords01_cpu.numpy(), cand_mode=self.cand_mode, k=self.k)
        if cand.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        y = label_candidate_pairs(cand, true_edges)

        if neg_subsample:
            rng = np.random.default_rng(int(self.global_step) + 123)
            pos_idx = np.where(y > 0.5)[0]
            neg_idx = np.where(y <= 0.5)[0]
            if len(pos_idx) == 0:
                return torch.tensor(0.0, device=self.device)
            n_neg = int(min(len(neg_idx), max(1, int(len(pos_idx) * self.neg_ratio))))
            neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg < len(neg_idx) else neg_idx
            sel = np.concatenate([pos_idx, neg_sel], axis=0)
            rng.shuffle(sel)
            cand_sel = cand[sel]
            y_sel = y[sel]
        else:
            cand_sel = cand
            y_sel = y

        coords01 = coords01_cpu.to(self.device)
        cand_t = torch.from_numpy(cand_sel).to(device=self.device, dtype=torch.long)
        y_t = torch.from_numpy(y_sel).to(device=self.device, dtype=torch.float32)

        h0 = self.model.node_in(coords01)
        s = self.model.search_proj(h0)
        msg_pairs = knn_candidate_pairs(s.detach().cpu().numpy(), k=int(self.k_msg))
        msg_edge_index = pairs_to_edge_index(msg_pairs).to(self.device)

        logits = self.model(coords01=coords01, msg_edge_index=msg_edge_index, cand_pairs_uv=cand_t, h0=h0)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._graph_loss(batch, neg_subsample=True)
        self.log("train/bce_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1)
        self.log("train/bce", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._graph_loss(batch, neg_subsample=False)
        self.log("val/bce", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss = self._graph_loss(batch, neg_subsample=False)
        self.log("test/bce", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def export_edge3_pt(*, lit: Edge3LitModule, out_path: str, val_bce: float | None = None) -> None:
    torch.save(
        {
            "model_state": lit.model.state_dict(),
            "cfg": asdict(lit.cfg),
            "k": int(lit.k),
            "cand_mode": str(lit.cand_mode),
            "k_msg": int(lit.k_msg),
            "variant": "edge_3",
            "val_bce": float(val_bce) if val_bce is not None else None,
        },
        out_path,
    )

