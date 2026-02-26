from __future__ import annotations

import math

import numpy as np
import torch

from .edge_model import EdgeModel
from .graph_utils import (
    ensure_connected_by_candidates,
    enforce_degree_cap,
    knn_candidate_pairs,
    pairs_to_edge_index,
)
from .node_diffusion import DiffusionSchedule, NodeDenoiser, NPredictor


@torch.no_grad()
def sample_node_count(
    n_pred: NPredictor,
    cond_z: torch.Tensor,
    *,
    min_n: int = 64,
    max_n: int = 5000,
) -> int:
    mu, log_sigma = n_pred(cond_z)
    sigma = torch.exp(log_sigma)
    eps = torch.randn_like(mu)
    logn = mu + sigma * eps
    n = int(torch.round(torch.exp(logn)).clamp(min=float(min_n), max=float(max_n)).item())
    return n


@torch.no_grad()
def ddpm_sample_coords(
    *,
    schedule: DiffusionSchedule,
    denoiser: NodeDenoiser,
    cond_z: torch.Tensor,
    n_nodes: int,
    k_nn: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns coords01 in (N,2) roughly in [0,1].
    """
    x = torch.randn((n_nodes, 2), device=device, dtype=torch.float32)
    T = schedule.n_steps
    for t_int in reversed(range(T)):
        t = torch.tensor([t_int], device=device, dtype=torch.long)
        cand = knn_candidate_pairs(x.detach().cpu().numpy(), k=int(k_nn))
        edge_index = pairs_to_edge_index(cand).to(device)
        eps_pred = denoiser(x_t=x, t=t, cond=cond_z, edge_index=edge_index)

        beta = schedule.betas[t].view(1, 1)
        alpha = schedule.alphas[t].view(1, 1)
        alpha_bar = schedule.alpha_bars[t].view(1, 1)

        # DDPM mean prediction
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - alpha_bar)) * eps_pred)

        if t_int > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean

    return x.clamp(0.0, 1.0)


@torch.no_grad()
def sample_edges_from_coords(
    *,
    edge_model: EdgeModel,
    coords01: torch.Tensor,  # (N,2)
    k: int,
    deg_cap: int = 12,
    ensure_connected: bool = True,
    device: torch.device,
) -> np.ndarray:
    coords_np = coords01.detach().cpu().numpy()
    cand = knn_candidate_pairs(coords_np, k=int(k))
    if cand.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    cand_t = torch.from_numpy(cand).to(device=device, dtype=torch.long)
    msg_edge_index = pairs_to_edge_index(cand).to(device)
    logits = edge_model(coords01=coords01.to(device), msg_edge_index=msg_edge_index, cand_pairs_uv=cand_t)
    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    n_nodes = int(coords01.shape[0])
    edges = enforce_degree_cap(n_nodes, cand, probs, deg_cap=int(deg_cap))
    if ensure_connected:
        edges = ensure_connected_by_candidates(n_nodes, edges, cand, probs)
    edges = np.unique(edges, axis=0)
    return edges

