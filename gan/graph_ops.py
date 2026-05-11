from __future__ import annotations

import numpy as np
import torch

from tessgen.graph_utils import candidate_pairs as tess_candidate_pairs


def candidate_pairs_from_xyr(
    xyr: torch.Tensor | np.ndarray,
    *,
    candidate_mode: str,
    k: int,
) -> np.ndarray:
    arr = xyr.detach().cpu().numpy() if isinstance(xyr, torch.Tensor) else np.asarray(xyr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"xyr/coords must be (N,D>=2), got {arr.shape}")
    coords = np.clip(arr[:, :2].astype(np.float64, copy=False), 0.0, 1.0)
    mode = str(candidate_mode)
    if mode == "delaunay" and coords.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.int64)
    if mode == "delaunay":
        try:
            return tess_candidate_pairs(coords, cand_mode="delaunay", k=int(k))
        except Exception:
            return tess_candidate_pairs(coords, cand_mode="knn", k=max(1, int(k)))
    if mode != "knn":
        raise ValueError(f"Unsupported candidate_mode={candidate_mode!r}")
    return tess_candidate_pairs(coords, cand_mode="knn", k=int(k))


def pairs_to_edge_index(pairs_uv: np.ndarray | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    pairs = pairs_uv.detach().cpu().numpy() if isinstance(pairs_uv, torch.Tensor) else np.asarray(pairs_uv)
    if pairs.size == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    u = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)
    v = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)
    return torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)


def undirected_to_directed(
    edges_uv: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edges_uv.numel() == 0:
        dev = edges_uv.device
        return torch.zeros((2, 0), dtype=torch.long, device=dev), torch.zeros((0,), dtype=torch.float32, device=dev)
    u = edges_uv[:, 0].long()
    v = edges_uv[:, 1].long()
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    if edge_weight is None:
        w = torch.ones((edges_uv.shape[0],), dtype=torch.float32, device=edges_uv.device)
    else:
        w = edge_weight.float()
    return edge_index, torch.cat([w, w], dim=0)


def threshold_edges(
    pairs_uv: np.ndarray,
    probs: torch.Tensor | np.ndarray,
    *,
    threshold: float,
    max_edges: int = 0,
) -> np.ndarray:
    if pairs_uv.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    p = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    keep = np.where(p >= float(threshold))[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(p))], dtype=np.int64)
    if int(max_edges) > 0 and keep.size > int(max_edges):
        order = keep[np.argsort(-p[keep])[: int(max_edges)]]
        keep = order
    edges = pairs_uv[keep].astype(np.int64, copy=False)
    return np.unique(edges, axis=0)


def sample_empirical_n(n_values: list[int], *, rng: np.random.Generator, n_samples: int) -> list[int]:
    if not n_values:
        raise ValueError("n_values is empty")
    idx = rng.integers(0, len(n_values), size=int(n_samples))
    return [int(n_values[int(i)]) for i in idx.tolist()]
