from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tessgen.gnn import GaussianRBF, SinusoidalEmbedding, mlp, scatter_sum

from ..graph_ops import candidate_pairs_from_xyr, pairs_to_edge_index, threshold_edges


def normalize_xyr(xyr: torch.Tensor, *, r_scale: float) -> torch.Tensor:
    out = xyr.float().clone()
    out[:, :2] = out[:, :2].clamp(0.0, 1.0)
    out[:, 2] = (out[:, 2] / max(float(r_scale), 1e-8)).clamp(0.0, 1.0)
    return out


def denormalize_xyr(xyr01: torch.Tensor, *, r_scale: float) -> torch.Tensor:
    out = xyr01.float().clone()
    out[:, :2] = out[:, :2].clamp(0.0, 1.0)
    out[:, 2] = out[:, 2].clamp(0.0, 1.0) * float(r_scale)
    return out


@dataclass(frozen=True)
class GraphModelConfig:
    d_h: int = 128
    n_layers: int = 3
    n_rbf: int = 16
    dropout: float = 0.0
    z_dim: int = 64


class WeightedInvariantMPNNLayer(nn.Module):
    def __init__(self, d_h: int, *, n_rbf: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rbf = GaussianRBF(n_rbf=n_rbf, r_min=0.0, r_max=math.sqrt(2.0))
        self.phi_m = mlp([2 * d_h + n_rbf, d_h, d_h], dropout=dropout)
        self.phi_h = mlp([2 * d_h, d_h, d_h], dropout=dropout)
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            dx = x[src, :2] - x[dst, :2]
            r = torch.sqrt((dx**2).sum(dim=-1) + 1e-8)
            rbf = self.rbf(r)
            m_in = torch.cat([h[src], h[dst], rbf], dim=-1)
            m = self.phi_m(m_in)
            if edge_weight is not None:
                m = m * edge_weight.float().view(-1, 1)
            agg = scatter_sum(m, dst, dim_size=h.shape[0])
        h_up = self.phi_h(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_up)


def _global_pool(h: torch.Tensor) -> torch.Tensor:
    mean = h.mean(dim=0)
    maxv = h.max(dim=0).values if h.shape[0] > 0 else torch.zeros((h.shape[1],), device=h.device, dtype=h.dtype)
    return torch.cat([mean, maxv], dim=-1)


def _node_degree_feature(n_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((n_nodes, 1), dtype=torch.float32, device=edge_index.device)
    dst = edge_index[1]
    if edge_weight is None:
        w = torch.ones((edge_index.shape[1],), dtype=torch.float32, device=edge_index.device)
    else:
        w = edge_weight.float()
    deg = scatter_sum(w, dst, dim_size=n_nodes).view(-1, 1)
    return deg / deg.mean().clamp_min(1.0)


class GraphCritic(nn.Module):
    """
    Weighted sparse GNN graph-level realism classifier.

    The same module is used as the GAN discriminator and as the diffusion
    critic. It can score hard graphs with unit edge weights or soft generated
    candidate graphs with differentiable edge probabilities.
    """

    def __init__(self, cfg: GraphModelConfig):
        super().__init__()
        self.cfg = cfg
        self.node_in = mlp([4, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [WeightedInvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.head = mlp([2 * cfg.d_h + 5, cfg.d_h, 1], dropout=cfg.dropout)

    def forward(
        self,
        *,
        xyr01: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = int(xyr01.shape[0])
        deg = _node_degree_feature(n, edge_index, edge_weight)
        h = self.node_in(torch.cat([xyr01.float(), deg], dim=-1))
        for layer in self.layers:
            h = layer(h, xyr01, edge_index, edge_weight)
        pooled = _global_pool(h)
        if edge_index.numel() == 0:
            edge_mass = xyr01.new_tensor(0.0)
        elif edge_weight is None:
            edge_mass = xyr01.new_tensor(float(edge_index.shape[1]) / 2.0)
        else:
            edge_mass = edge_weight.float().sum() / 2.0
        graph_stats = torch.stack(
            [
                torch.log(xyr01.new_tensor(float(max(n, 1)))),
                edge_mass / max(float(n), 1.0),
                xyr01[:, 2].mean(),
                xyr01[:, 2].std(unbiased=False),
                xyr01[:, :2].std(unbiased=False),
            ]
        )
        return self.head(torch.cat([pooled, graph_stats], dim=-1)).view(())


class NodeGenerator(nn.Module):
    def __init__(self, cfg: GraphModelConfig, *, k_msg: int):
        super().__init__()
        self.cfg = cfg
        self.k_msg = int(k_msg)
        self.seed_mlp = mlp([cfg.z_dim + 5, cfg.d_h, cfg.d_h, 3], dropout=cfg.dropout)
        self.node_in = mlp([3 + cfg.z_dim + 2, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [WeightedInvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.out = mlp([cfg.d_h, cfg.d_h, 3], dropout=cfg.dropout)

    def forward(self, *, n_nodes: int, z: torch.Tensor | None = None, device: torch.device | None = None) -> torch.Tensor:
        n_nodes = int(n_nodes)
        if n_nodes <= 0:
            raise ValueError("n_nodes must be > 0")
        if z is None:
            if device is None:
                device = next(self.parameters()).device
            z = torch.randn((self.cfg.z_dim,), device=device)
        else:
            device = z.device
        node_noise = torch.randn((n_nodes, 4), device=device)
        rank = torch.linspace(0.0, 1.0, steps=n_nodes, device=device).view(-1, 1)
        z_expand = z.view(1, -1).expand(n_nodes, -1)
        seed_logits = self.seed_mlp(torch.cat([z_expand, node_noise, rank], dim=-1))
        xyr0 = torch.sigmoid(seed_logits)

        pairs = candidate_pairs_from_xyr(xyr0.detach(), candidate_mode="knn", k=max(1, self.k_msg))
        edge_index = pairs_to_edge_index(pairs, device=device)
        edge_weight = torch.ones((edge_index.shape[1],), device=device, dtype=torch.float32)
        h = self.node_in(torch.cat([xyr0, z_expand, rank, node_noise[:, :1]], dim=-1))
        for layer in self.layers:
            h = layer(h, xyr0, edge_index, edge_weight)
        delta = self.out(h)
        xyr = torch.sigmoid(torch.logit(xyr0.clamp(1e-4, 1.0 - 1e-4)) + delta)
        return xyr


class EdgeGenerator(nn.Module):
    def __init__(self, cfg: GraphModelConfig):
        super().__init__()
        self.cfg = cfg
        self.node_in = mlp([3, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [WeightedInvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(max(1, cfg.n_layers // 2))]
        )
        self.edge_head = mlp([2 * cfg.d_h + 4, cfg.d_h, 1], dropout=cfg.dropout)

    def forward(
        self,
        *,
        xyr01: torch.Tensor,
        msg_edge_index: torch.Tensor,
        cand_pairs_uv: torch.Tensor,
    ) -> torch.Tensor:
        if cand_pairs_uv.numel() == 0:
            return xyr01.new_zeros((0,))
        edge_weight = torch.ones((msg_edge_index.shape[1],), device=xyr01.device, dtype=torch.float32)
        h = self.node_in(xyr01.float())
        for layer in self.layers:
            h = layer(h, xyr01, msg_edge_index, edge_weight)
        u = cand_pairs_uv[:, 0].long()
        v = cand_pairs_uv[:, 1].long()
        dx = xyr01[u, :2] - xyr01[v, :2]
        dist = torch.sqrt((dx**2).sum(dim=-1, keepdim=True) + 1e-8)
        r_pair = torch.cat([xyr01[u, 2:3], xyr01[v, 2:3], dist, torch.abs(xyr01[u, 2:3] - xyr01[v, 2:3])], dim=-1)
        logits = self.edge_head(torch.cat([h[u], h[v], r_pair], dim=-1)).squeeze(-1)
        return logits


class SparseGraphGenerator(nn.Module):
    def __init__(self, cfg: GraphModelConfig, *, k_msg: int, k_edge: int, candidate_mode: str):
        super().__init__()
        self.cfg = cfg
        self.k_msg = int(k_msg)
        self.k_edge = int(k_edge)
        self.candidate_mode = str(candidate_mode)
        self.node_generator = NodeGenerator(cfg, k_msg=k_msg)
        self.edge_generator = EdgeGenerator(cfg)

    def forward(
        self,
        *,
        n_nodes: int,
        z: torch.Tensor | None = None,
        edge_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        xyr01 = self.node_generator(n_nodes=int(n_nodes), z=z)
        pairs_np = candidate_pairs_from_xyr(xyr01.detach(), candidate_mode=self.candidate_mode, k=self.k_edge)
        pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=xyr01.device)
        msg_edge_index = pairs_to_edge_index(pairs_np, device=xyr01.device)
        edge_logits = self.edge_generator(xyr01=xyr01, msg_edge_index=msg_edge_index, cand_pairs_uv=pairs)
        probs = torch.sigmoid(edge_logits / max(float(edge_temperature), 1e-6))
        return {"xyr01": xyr01, "pairs_uv": pairs, "edge_logits": edge_logits, "edge_probs": probs}

    @torch.no_grad()
    def sample_hard(
        self,
        *,
        n_nodes: int,
        threshold: float,
        max_edges: int = 0,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        out = self.forward(n_nodes=int(n_nodes), z=torch.randn((self.cfg.z_dim,), device=device))
        pairs_np = out["pairs_uv"].detach().cpu().numpy()
        edges_np = threshold_edges(pairs_np, out["edge_probs"], threshold=float(threshold), max_edges=int(max_edges))
        return out["xyr01"].detach(), torch.as_tensor(edges_np, dtype=torch.long, device=device)


@dataclass(frozen=True)
class DiffusionConfig:
    n_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DiffusionSchedule(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        betas = torch.linspace(float(cfg.beta_start), float(cfg.beta_end), steps=int(cfg.n_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def n_steps(self) -> int:
        return int(self.betas.shape[0])

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t].view(1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


class GraphDenoiser(nn.Module):
    def __init__(self, cfg: GraphModelConfig):
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalEmbedding(dim=cfg.d_h)
        self.time_mlp = mlp([cfg.d_h, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.node_in = mlp([3 + cfg.d_h, cfg.d_h, cfg.d_h], dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [WeightedInvariantMPNNLayer(cfg.d_h, n_rbf=cfg.n_rbf, dropout=cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.out = mlp([cfg.d_h, cfg.d_h, 3], dropout=cfg.dropout)

    def forward(self, *, x_t: torch.Tensor, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = int(x_t.shape[0])
        t_emb = self.time_mlp(self.time_emb(t.view(1)).squeeze(0))
        h = self.node_in(torch.cat([x_t, t_emb.view(1, -1).expand(n, -1)], dim=-1))
        edge_weight = torch.ones((edge_index.shape[1],), device=x_t.device, dtype=torch.float32)
        for layer in self.layers:
            h = layer(h, x_t, edge_index, edge_weight)
        return self.out(h)


class DiffusionGraphGenerator(nn.Module):
    def __init__(self, cfg: GraphModelConfig, diffusion_cfg: DiffusionConfig, *, k_msg: int, k_edge: int, candidate_mode: str):
        super().__init__()
        self.cfg = cfg
        self.diffusion_cfg = diffusion_cfg
        self.k_msg = int(k_msg)
        self.k_edge = int(k_edge)
        self.candidate_mode = str(candidate_mode)
        self.schedule = DiffusionSchedule(diffusion_cfg)
        self.denoiser = GraphDenoiser(cfg)
        self.edge_generator = EdgeGenerator(cfg)

    def diffusion_loss(self, xyr01: torch.Tensor) -> torch.Tensor:
        device = xyr01.device
        t = torch.randint(0, self.schedule.n_steps, (1,), device=device, dtype=torch.long)
        eps = torch.randn_like(xyr01)
        x_t = self.schedule.q_sample(xyr01, t, eps)
        pairs_np = candidate_pairs_from_xyr(x_t.detach(), candidate_mode="knn", k=self.k_msg)
        edge_index = pairs_to_edge_index(pairs_np, device=device)
        pred = self.denoiser(x_t=x_t, t=t, edge_index=edge_index)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample_nodes(self, *, n_nodes: int, sample_steps: int = 0, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        steps = self.schedule.n_steps if int(sample_steps) <= 0 else min(int(sample_steps), self.schedule.n_steps)
        x = torch.randn((int(n_nodes), 3), device=device)
        step_ids = torch.linspace(self.schedule.n_steps - 1, 0, steps=steps, device=device).long()
        for t_val in step_ids.tolist():
            t = torch.tensor([int(t_val)], device=device, dtype=torch.long)
            pairs_np = candidate_pairs_from_xyr(x.detach().sigmoid(), candidate_mode="knn", k=self.k_msg)
            edge_index = pairs_to_edge_index(pairs_np, device=device)
            eps = self.denoiser(x_t=x, t=t, edge_index=edge_index)
            beta = self.schedule.betas[t].view(1, 1)
            alpha = self.schedule.alphas[t].view(1, 1)
            alpha_bar = self.schedule.alpha_bars[t].view(1, 1)
            x = (x - beta / torch.sqrt(1.0 - alpha_bar).clamp_min(1e-6) * eps) / torch.sqrt(alpha).clamp_min(1e-6)
            if int(t_val) > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return torch.sigmoid(x)

    def edge_logits_for(self, xyr01: torch.Tensor) -> dict[str, torch.Tensor]:
        pairs_np = candidate_pairs_from_xyr(xyr01.detach(), candidate_mode=self.candidate_mode, k=self.k_edge)
        pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=xyr01.device)
        msg_edge_index = pairs_to_edge_index(pairs_np, device=xyr01.device)
        logits = self.edge_generator(xyr01=xyr01, msg_edge_index=msg_edge_index, cand_pairs_uv=pairs)
        return {"pairs_uv": pairs, "edge_logits": logits, "edge_probs": torch.sigmoid(logits)}

    @torch.no_grad()
    def sample_hard(
        self,
        *,
        n_nodes: int,
        threshold: float,
        sample_steps: int = 0,
        max_edges: int = 0,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xyr01 = self.sample_nodes(n_nodes=int(n_nodes), sample_steps=int(sample_steps), device=device)
        out = self.edge_logits_for(xyr01)
        pairs_np = out["pairs_uv"].detach().cpu().numpy()
        edges_np = threshold_edges(pairs_np, out["edge_probs"], threshold=float(threshold), max_edges=int(max_edges))
        return xyr01.detach(), torch.as_tensor(edges_np, dtype=torch.long, device=xyr01.device)
