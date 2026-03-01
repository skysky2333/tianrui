from __future__ import annotations

import math

import numpy as np
import torch

from .ckpt import EdgeBundle
from .graph_utils import (
    candidate_pairs,
    ensure_connected_by_candidates,
    enforce_degree_cap,
    knn_candidate_pairs,
    pairs_to_edge_index,
)
from .node_diffusion import DiffusionSchedule, NodeDenoiser


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
    edge_bundle: EdgeBundle,
    coords01: torch.Tensor,  # (N,2)
    deg_cap: int = 12,
    edge_thr: float = 0.5,
    ensure_connected: bool = True,
    device: torch.device,
) -> np.ndarray:
    edge_thr = float(edge_thr)
    if not (0.0 <= edge_thr <= 1.0):
        raise ValueError(f"edge_thr must be in [0,1]; got {edge_thr}")

    coords_np = coords01.detach().cpu().numpy()
    cand = candidate_pairs(coords_np, cand_mode=edge_bundle.cand_mode, k=int(edge_bundle.k))
    if cand.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    cand_t = torch.from_numpy(cand).to(device=device, dtype=torch.long)

    coords01_device = coords01.to(device)
    if edge_bundle.variant == "edge":
        msg_edge_index = pairs_to_edge_index(cand).to(device)
        logits = edge_bundle.model(coords01=coords01_device, msg_edge_index=msg_edge_index, cand_pairs_uv=cand_t)
    elif edge_bundle.variant == "edge_3":
        if edge_bundle.k_msg is None:
            raise ValueError("edge_bundle.k_msg must be set for variant='edge_3'")
        h0 = edge_bundle.model.node_in(coords01_device)
        s = edge_bundle.model.search_proj(h0)
        msg_pairs = knn_candidate_pairs(s.detach().cpu().numpy(), k=int(edge_bundle.k_msg))
        msg_edge_index = pairs_to_edge_index(msg_pairs).to(device)
        logits = edge_bundle.model(coords01=coords01_device, msg_edge_index=msg_edge_index, cand_pairs_uv=cand_t, h0=h0)
    else:
        raise ValueError(f"Unsupported edge_bundle.variant={edge_bundle.variant!r}")

    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    n_nodes = int(coords01.shape[0])
    mask = probs >= edge_thr
    cand_f = cand[mask]
    probs_f = probs[mask]
    if cand_f.shape[0] == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    else:
        edges = enforce_degree_cap(n_nodes, cand_f, probs_f, deg_cap=int(deg_cap))
    if ensure_connected:
        edges = ensure_connected_by_candidates(n_nodes, edges, cand, probs)
    edges = np.unique(edges, axis=0)
    return edges
