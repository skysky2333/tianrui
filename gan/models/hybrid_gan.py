from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..graph_ops import candidate_pairs_from_xyr, undirected_to_directed
from ..metrics import corrupt_graph, real_graph_stats_torch, soft_graph_stats_torch
from .common import GraphCritic, GraphModelConfig, SparseGraphGenerator, denormalize_xyr, normalize_xyr


class HybridGANLitModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        k_msg: int,
        k_edge: int,
        candidate_mode: str,
        r_scale: float,
        lr_g: float,
        lr_d: float,
        weight_decay: float,
        lambda_stats: float = 10.0,
        edge_temperature: float = 1.0,
        g_steps_per_batch: int = 1,
        d_every_n_steps: int = 1,
        real_label_smooth: float = 0.9,
        fake_label_smooth: float = 0.0,
        instance_noise: float = 0.0,
        learn_edges: bool = False,
        lambda_adv: float = 1.0,
        lambda_seed: float = 50.0,
        seed_mmd_points: int = 256,
        g_pretrain_steps: int = 5000,
        d_loss_floor: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        cfg = GraphModelConfig(**model_cfg)
        self.generator = SparseGraphGenerator(cfg, k_msg=int(k_msg), k_edge=int(k_edge), candidate_mode=str(candidate_mode))
        self.discriminator = GraphCritic(cfg)
        self.automatic_optimization = False
        self._batches_seen = 0

    @property
    def r_scale(self) -> float:
        return float(self.hparams.r_scale)

    def _real_logit(self, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
        edge_index, edge_weight = undirected_to_directed(edges_uv)
        return self.discriminator(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)

    def _helper_edges(self, xyr01: torch.Tensor, *, k: int | None = None) -> torch.Tensor:
        pairs = candidate_pairs_from_xyr(
            xyr01.detach(),
            candidate_mode=str(self.hparams.candidate_mode),
            k=int(self.hparams.k_msg if k is None else k),
        )
        return torch.as_tensor(pairs, dtype=torch.long, device=xyr01.device)

    def _generate_fake(self, *, n_nodes: int) -> dict[str, torch.Tensor]:
        if bool(self.hparams.learn_edges):
            return self.generator(n_nodes=n_nodes, edge_temperature=float(self.hparams.edge_temperature))
        xyr01 = self.generator.node_generator(n_nodes=int(n_nodes), device=self.device)
        pairs = self._helper_edges(xyr01)
        probs = torch.ones((pairs.shape[0],), dtype=torch.float32, device=xyr01.device)
        return {"xyr01": xyr01, "pairs_uv": pairs, "edge_logits": torch.zeros_like(probs), "edge_probs": probs}

    def _fake_logit(self, fake: dict[str, torch.Tensor], *, detach: bool) -> torch.Tensor:
        xyr = fake["xyr01"].detach() if detach else fake["xyr01"]
        pairs = fake["pairs_uv"].detach() if detach else fake["pairs_uv"]
        probs = fake["edge_probs"].detach() if detach else fake["edge_probs"]
        edge_index, edge_weight = undirected_to_directed(pairs, edge_weight=probs)
        return self.discriminator(xyr01=xyr, edge_index=edge_index, edge_weight=edge_weight)

    def _corrupt_logit(self, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
        x_c, e_c = corrupt_graph(xyr01, edges_uv)
        edge_index, edge_weight = undirected_to_directed(e_c)
        return self.discriminator(xyr01=x_c, edge_index=edge_index, edge_weight=edge_weight)

    def _disc_input(self, xyr01: torch.Tensor) -> torch.Tensor:
        sigma = float(self.hparams.instance_noise)
        if sigma <= 0.0 or not self.training:
            return xyr01
        x = xyr01 + sigma * torch.randn_like(xyr01)
        x[:, :2] = x[:, :2].clamp(0.0, 1.0)
        x[:, 2] = x[:, 2].clamp(0.0, 1.0)
        return x

    def _sample_seed_points(self, xyr01: torch.Tensor) -> torch.Tensor:
        max_points = int(self.hparams.seed_mmd_points)
        if max_points <= 0 or xyr01.shape[0] <= max_points:
            return xyr01
        idx = torch.randperm(xyr01.shape[0], device=xyr01.device)[:max_points]
        return xyr01[idx]

    @staticmethod
    def _rbf_mmd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aa = torch.cdist(a, a, p=2.0).pow(2)
        bb = torch.cdist(b, b, p=2.0).pow(2)
        ab = torch.cdist(a, b, p=2.0).pow(2)
        loss = a.new_tensor(0.0)
        for sigma in (0.02, 0.05, 0.1, 0.2, 0.5):
            gamma = 1.0 / (2.0 * sigma * sigma)
            loss = loss + torch.exp(-gamma * aa).mean() + torch.exp(-gamma * bb).mean() - 2.0 * torch.exp(-gamma * ab).mean()
        return loss

    def _seed_distribution_loss(self, real_xyr01: torch.Tensor, fake_xyr01: torch.Tensor) -> torch.Tensor:
        real = self._sample_seed_points(real_xyr01)
        fake = self._sample_seed_points(fake_xyr01)
        n = min(int(real.shape[0]), int(fake.shape[0]))
        if n <= 1:
            return F.l1_loss(fake.mean(dim=0), real.mean(dim=0))
        real = real[:n]
        fake = fake[:n]

        mean_loss = F.l1_loss(fake.mean(dim=0), real.mean(dim=0))
        std_loss = F.l1_loss(fake.std(dim=0, unbiased=False), real.std(dim=0, unbiased=False))
        mmd_loss = self._rbf_mmd(fake, real)
        r_loss = F.l1_loss(torch.sort(fake[:, 2]).values, torch.sort(real[:, 2]).values)
        real_pd = torch.sort(torch.pdist(real[:, :2], p=2.0)).values
        fake_pd = torch.sort(torch.pdist(fake[:, :2], p=2.0)).values
        m = min(int(real_pd.shape[0]), int(fake_pd.shape[0]))
        pd_loss = F.l1_loss(fake_pd[:m], real_pd[:m]) if m > 0 else fake.new_tensor(0.0)
        return mean_loss + std_loss + mmd_loss + r_loss + pd_loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._batches_seen += 1
        opt_g, opt_d = self.optimizers()
        xyr01 = normalize_xyr(batch["xyr"].to(self.device), r_scale=self.r_scale)
        if bool(self.hparams.learn_edges):
            edges_uv = batch["edges_undirected"].to(self.device)
        else:
            edges_uv = self._helper_edges(xyr01)
        n_nodes = int(batch["n_nodes"])

        real_logit = xyr01.new_tensor(0.0)
        fake_logit = xyr01.new_tensor(0.0)
        d_loss = xyr01.new_tensor(0.0)
        in_g_pretrain = self._batches_seen <= int(self.hparams.g_pretrain_steps)
        d_scheduled = (int(batch_idx) % max(1, int(self.hparams.d_every_n_steps))) == 0 and not in_g_pretrain
        d_updated = False
        if d_scheduled:
            with torch.no_grad():
                fake_d = self._generate_fake(n_nodes=n_nodes)
            fake_d = dict(fake_d)
            fake_d["xyr01"] = self._disc_input(fake_d["xyr01"].detach())
            real_logit = self._real_logit(self._disc_input(xyr01), edges_uv)
            fake_logit = self._fake_logit(fake_d, detach=True)
            corrupt_logit = self._corrupt_logit(self._disc_input(xyr01), edges_uv)
            real_target = torch.full((1,), float(self.hparams.real_label_smooth), device=self.device)
            fake_target = torch.full((1,), float(self.hparams.fake_label_smooth), device=self.device)
            d_loss = (
                F.binary_cross_entropy_with_logits(real_logit.view(1), real_target)
                + F.binary_cross_entropy_with_logits(fake_logit.view(1), fake_target)
                + F.binary_cross_entropy_with_logits(corrupt_logit.view(1), fake_target)
            ) / 3.0

            if float(d_loss.detach().cpu()) >= float(self.hparams.d_loss_floor):
                opt_d.zero_grad()
                self.manual_backward(d_loss)
                opt_d.step()
                d_updated = True

        real_stats = real_graph_stats_torch(xyr01, edges_uv).detach()
        g_losses = []
        g_advs = []
        stat_losses = []
        seed_losses = []
        fake_probs = []
        for _ in range(max(1, int(self.hparams.g_steps_per_batch))):
            fake_g = self._generate_fake(n_nodes=n_nodes)
            fake_g_logit = self._fake_logit(fake_g, detach=False)
            g_adv = F.binary_cross_entropy_with_logits(fake_g_logit.view(1), torch.ones(1, device=self.device))
            fake_stats = soft_graph_stats_torch(fake_g["xyr01"], fake_g["pairs_uv"], fake_g["edge_probs"])
            stats_loss = F.l1_loss(fake_stats, real_stats)
            seed_loss = self._seed_distribution_loss(xyr01, fake_g["xyr01"])
            adv_weight = 0.0 if in_g_pretrain else float(self.hparams.lambda_adv)
            g_loss = adv_weight * g_adv + float(self.hparams.lambda_stats) * stats_loss + float(self.hparams.lambda_seed) * seed_loss

            opt_g.zero_grad()
            self.manual_backward(g_loss)
            opt_g.step()
            g_losses.append(g_loss.detach())
            g_advs.append(g_adv.detach())
            stat_losses.append(stats_loss.detach())
            seed_losses.append(seed_loss.detach())
            fake_probs.append(torch.sigmoid(fake_g_logit.detach()))

        g_loss_log = torch.stack(g_losses).mean()
        g_adv_log = torch.stack(g_advs).mean()
        stats_loss_log = torch.stack(stat_losses).mean()
        seed_loss_log = torch.stack(seed_losses).mean()
        fake_prob_log = torch.stack(fake_probs).mean()

        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/g_loss", g_loss_log, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/g_adv", g_adv_log, on_step=True, on_epoch=True, batch_size=1)
        self.log("train/stats_loss", stats_loss_log, on_step=True, on_epoch=True, batch_size=1)
        self.log("train/seed_loss", seed_loss_log, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/real_prob", torch.sigmoid(real_logit), on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/fake_prob", fake_prob_log, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/d_updated", xyr01.new_tensor(1.0 if d_updated else 0.0), on_step=True, on_epoch=True, batch_size=1)
        self.log("train/g_pretrain", xyr01.new_tensor(1.0 if in_g_pretrain else 0.0), on_step=True, on_epoch=True, batch_size=1)

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        xyr01 = normalize_xyr(batch["xyr"].to(self.device), r_scale=self.r_scale)
        if bool(self.hparams.learn_edges):
            edges_uv = batch["edges_undirected"].to(self.device)
        else:
            edges_uv = self._helper_edges(xyr01)
        n_nodes = int(batch["n_nodes"])
        fake = self._generate_fake(n_nodes=n_nodes)
        real_logit = self._real_logit(xyr01, edges_uv)
        fake_logit = self._fake_logit(fake, detach=True)
        corrupt_logit = self._corrupt_logit(xyr01, edges_uv)
        d_loss = (
            F.binary_cross_entropy_with_logits(real_logit.view(1), torch.ones(1, device=self.device))
            + F.binary_cross_entropy_with_logits(fake_logit.view(1), torch.zeros(1, device=self.device))
            + F.binary_cross_entropy_with_logits(corrupt_logit.view(1), torch.zeros(1, device=self.device))
        ) / 3.0
        real_stats = real_graph_stats_torch(xyr01, edges_uv)
        fake_stats = soft_graph_stats_torch(fake["xyr01"], fake["pairs_uv"], fake["edge_probs"])
        stats_loss = F.l1_loss(fake_stats, real_stats)
        seed_loss = self._seed_distribution_loss(xyr01, fake["xyr01"])
        val_loss = d_loss + stats_loss + float(self.hparams.lambda_seed) * seed_loss
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/d_loss", d_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/stats_loss", stats_loss, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/seed_loss", seed_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/real_prob", torch.sigmoid(real_logit), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/fake_prob", torch.sigmoid(fake_logit), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("val/corrupt_prob", torch.sigmoid(corrupt_logit), on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=float(self.hparams.lr_g),
            weight_decay=float(self.hparams.weight_decay),
            betas=(0.5, 0.95),
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=float(self.hparams.lr_d),
            weight_decay=float(self.hparams.weight_decay),
            betas=(0.5, 0.95),
        )
        return [opt_g, opt_d]

    @torch.no_grad()
    def sample_graph(self, *, n_nodes: int, edge_threshold: float, max_edges: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.hparams.learn_edges):
            xyr01, edges = self.generator.sample_hard(
                n_nodes=int(n_nodes),
                threshold=float(edge_threshold),
                max_edges=int(max_edges),
                device=self.device,
            )
        else:
            xyr01 = self.generator.node_generator(n_nodes=int(n_nodes), device=self.device)
            pairs = candidate_pairs_from_xyr(
                xyr01.detach(),
                candidate_mode=str(self.hparams.candidate_mode),
                k=int(self.hparams.k_edge),
            )
            edges = torch.as_tensor(pairs, dtype=torch.long, device=self.device)
        return denormalize_xyr(xyr01, r_scale=self.r_scale), edges

    def export_payload(self, *, config: dict[str, Any], n_values: list[int], best_val_loss: float | None) -> dict[str, Any]:
        model_cfg = GraphModelConfig(**self.hparams.model_cfg)
        return {
            "approach": "hybrid_gan",
            "config": config,
            "model_cfg": asdict(model_cfg),
            "hparams": dict(self.hparams),
            "r_scale": self.r_scale,
            "n_values": list(map(int, n_values)),
            "best_val_loss": None if best_val_loss is None else float(best_val_loss),
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }
