from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..graph_ops import candidate_pairs_from_xyr, pairs_to_edge_index, undirected_to_directed
from ..metrics import corrupt_graph
from .common import (
    DiffusionConfig,
    DiffusionGraphGenerator,
    GraphCritic,
    GraphModelConfig,
    denormalize_xyr,
    normalize_xyr,
)


def _label_candidate_pairs(cand_pairs_uv: np.ndarray, true_edges_uv: np.ndarray) -> np.ndarray:
    if cand_pairs_uv.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if true_edges_uv.size == 0:
        return np.zeros((cand_pairs_uv.shape[0],), dtype=np.float32)
    max_node = int(max(int(cand_pairs_uv.max()), int(true_edges_uv.max())) + 1)
    cand_code = cand_pairs_uv[:, 0].astype(np.int64) * max_node + cand_pairs_uv[:, 1].astype(np.int64)
    true_code = true_edges_uv[:, 0].astype(np.int64) * max_node + true_edges_uv[:, 1].astype(np.int64)
    return np.isin(cand_code, true_code).astype(np.float32)


class DiffusionCriticLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        diffusion_cfg: dict[str, Any],
        k_msg: int,
        k_edge: int,
        candidate_mode: str,
        r_scale: float,
        lr_g: float,
        lr_d: float,
        weight_decay: float,
        lambda_edge: float = 0.0,
        critic_sample_steps: int = 8,
        learn_edges: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        cfg = GraphModelConfig(**model_cfg)
        dcfg = DiffusionConfig(**diffusion_cfg)
        self.generator = DiffusionGraphGenerator(
            cfg,
            dcfg,
            k_msg=int(k_msg),
            k_edge=int(k_edge),
            candidate_mode=str(candidate_mode),
        )
        self.critic = GraphCritic(cfg)
        self.automatic_optimization = False

    @property
    def r_scale(self) -> float:
        return float(self.hparams.r_scale)

    def _real_logit(self, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
        edge_index, edge_weight = undirected_to_directed(edges_uv)
        return self.critic(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)

    def _helper_edges(self, xyr01: torch.Tensor, *, k: int | None = None) -> torch.Tensor:
        pairs = candidate_pairs_from_xyr(
            xyr01.detach(),
            candidate_mode=str(self.hparams.candidate_mode),
            k=int(self.hparams.k_msg if k is None else k),
        )
        return torch.as_tensor(pairs, dtype=torch.long, device=xyr01.device)

    def _fake_logit(self, xyr01: torch.Tensor, *, detach_generator_edges: bool = False) -> torch.Tensor:
        if not bool(self.hparams.learn_edges):
            pairs = self._helper_edges(xyr01)
            edge_index, edge_weight = undirected_to_directed(
                pairs,
                edge_weight=torch.ones((pairs.shape[0],), dtype=torch.float32, device=xyr01.device),
            )
            return self.critic(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)
        if detach_generator_edges:
            with torch.no_grad():
                edge_out = self.generator.edge_logits_for(xyr01)
            edge_out = {
                "pairs_uv": edge_out["pairs_uv"].detach(),
                "edge_probs": edge_out["edge_probs"].detach(),
            }
        else:
            edge_out = self.generator.edge_logits_for(xyr01)
        edge_index, edge_weight = undirected_to_directed(edge_out["pairs_uv"], edge_weight=edge_out["edge_probs"])
        return self.critic(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)

    def _corrupt_logit(self, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
        x_c, e_c = corrupt_graph(xyr01, edges_uv)
        edge_index, edge_weight = undirected_to_directed(e_c)
        return self.critic(xyr01=x_c, edge_index=edge_index, edge_weight=edge_weight)

    def _edge_loss(self, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
        pairs_np = candidate_pairs_from_xyr(
            xyr01.detach(),
            candidate_mode=str(self.hparams.candidate_mode),
            k=int(self.hparams.k_edge),
        )
        if pairs_np.size == 0:
            return xyr01.new_tensor(0.0)
        labels_np = _label_candidate_pairs(pairs_np, edges_uv.detach().cpu().numpy())
        labels = torch.as_tensor(labels_np, dtype=torch.float32, device=xyr01.device)
        pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=xyr01.device)
        msg_edge_index = pairs_to_edge_index(pairs_np, device=xyr01.device)
        logits = self.generator.edge_generator(xyr01=xyr01, msg_edge_index=msg_edge_index, cand_pairs_uv=pairs)
        pos = labels.sum()
        neg = labels.numel() - pos
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 50.0)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        opt_g, opt_c = self.optimizers()
        xyr01 = normalize_xyr(batch["xyr"].to(self.device), r_scale=self.r_scale)
        if bool(self.hparams.learn_edges):
            edges_uv = batch["edges_undirected"].to(self.device)
        else:
            edges_uv = self._helper_edges(xyr01)
        n_nodes = int(batch["n_nodes"])

        diff_loss = self.generator.diffusion_loss(xyr01)
        edge_loss = self._edge_loss(xyr01, edges_uv) if bool(self.hparams.learn_edges) else xyr01.new_tensor(0.0)
        g_loss = diff_loss + float(self.hparams.lambda_edge) * edge_loss
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        with torch.no_grad():
            fake_xyr = self.generator.sample_nodes(
                n_nodes=n_nodes,
                sample_steps=int(self.hparams.critic_sample_steps),
                device=self.device,
            )
        real_logit = self._real_logit(xyr01, edges_uv)
        fake_logit = self._fake_logit(fake_xyr.detach(), detach_generator_edges=True)
        corrupt_logit = self._corrupt_logit(xyr01, edges_uv)
        c_loss = (
            F.binary_cross_entropy_with_logits(real_logit.view(1), torch.ones(1, device=self.device))
            + F.binary_cross_entropy_with_logits(fake_logit.view(1), torch.zeros(1, device=self.device))
            + F.binary_cross_entropy_with_logits(corrupt_logit.view(1), torch.zeros(1, device=self.device))
        ) / 3.0
        opt_c.zero_grad()
        self.manual_backward(c_loss)
        opt_c.step()

        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/diff_loss", diff_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/edge_loss", edge_loss, on_step=True, on_epoch=True, batch_size=1)
        self.log("train/critic_loss", c_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/real_prob", torch.sigmoid(real_logit), on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/fake_prob", torch.sigmoid(fake_logit), on_step=True, on_epoch=True, prog_bar=True, batch_size=1)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        xyr01 = normalize_xyr(batch["xyr"].to(self.device), r_scale=self.r_scale)
        if bool(self.hparams.learn_edges):
            edges_uv = batch["edges_undirected"].to(self.device)
        else:
            edges_uv = self._helper_edges(xyr01)
        n_nodes = int(batch["n_nodes"])
        diff_loss = self.generator.diffusion_loss(xyr01)
        edge_loss = self._edge_loss(xyr01, edges_uv) if bool(self.hparams.learn_edges) else xyr01.new_tensor(0.0)
        fake_xyr = self.generator.sample_nodes(
            n_nodes=n_nodes,
            sample_steps=int(self.hparams.critic_sample_steps),
            device=self.device,
        )
        real_logit = self._real_logit(xyr01, edges_uv)
        fake_logit = self._fake_logit(fake_xyr.detach(), detach_generator_edges=True)
        corrupt_logit = self._corrupt_logit(xyr01, edges_uv)
        c_loss = (
            F.binary_cross_entropy_with_logits(real_logit.view(1), torch.ones(1, device=self.device))
            + F.binary_cross_entropy_with_logits(fake_logit.view(1), torch.zeros(1, device=self.device))
            + F.binary_cross_entropy_with_logits(corrupt_logit.view(1), torch.zeros(1, device=self.device))
        ) / 3.0
        val_loss = diff_loss + float(self.hparams.lambda_edge) * edge_loss + c_loss
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/diff_loss", diff_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/edge_loss", edge_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/critic_loss", c_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/real_prob", torch.sigmoid(real_logit), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/fake_prob", torch.sigmoid(fake_logit), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/corrupt_prob", torch.sigmoid(corrupt_logit), on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(self.hparams.lr_g),
            weight_decay=float(self.hparams.weight_decay),
            betas=(0.9, 0.99),
        )
        opt_c = torch.optim.AdamW(
            self.critic.parameters(),
            lr=float(self.hparams.lr_d),
            weight_decay=float(self.hparams.weight_decay),
            betas=(0.5, 0.95),
        )
        return [opt_g, opt_c]

    @torch.no_grad()
    def sample_graph(
        self,
        *,
        n_nodes: int,
        edge_threshold: float,
        sample_steps: int = 0,
        max_edges: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.hparams.learn_edges):
            xyr01, edges = self.generator.sample_hard(
                n_nodes=int(n_nodes),
                threshold=float(edge_threshold),
                sample_steps=int(sample_steps),
                max_edges=int(max_edges),
                device=self.device,
            )
        else:
            xyr01 = self.generator.sample_nodes(n_nodes=int(n_nodes), sample_steps=int(sample_steps), device=self.device)
            pairs = candidate_pairs_from_xyr(
                xyr01.detach(),
                candidate_mode=str(self.hparams.candidate_mode),
                k=int(self.hparams.k_edge),
            )
            edges = torch.as_tensor(pairs, dtype=torch.long, device=self.device)
        return denormalize_xyr(xyr01, r_scale=self.r_scale), edges

    def export_payload(self, *, config: dict[str, Any], n_values: list[int], best_val_loss: float | None) -> dict[str, Any]:
        model_cfg = GraphModelConfig(**self.hparams.model_cfg)
        diffusion_cfg = DiffusionConfig(**self.hparams.diffusion_cfg)
        return {
            "approach": "diffusion_critic",
            "config": config,
            "model_cfg": asdict(model_cfg),
            "diffusion_cfg": asdict(diffusion_cfg),
            "hparams": dict(self.hparams),
            "r_scale": self.r_scale,
            "n_values": list(map(int, n_values)),
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "generator_state_dict": self.generator.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }
