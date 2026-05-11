from __future__ import annotations

import numpy as np
import torch


def degree_numpy(n_nodes: int, edges_uv: np.ndarray) -> np.ndarray:
    deg = np.zeros((int(n_nodes),), dtype=np.float32)
    if edges_uv.size == 0:
        return deg
    np.add.at(deg, edges_uv[:, 0], 1.0)
    np.add.at(deg, edges_uv[:, 1], 1.0)
    return deg


def graph_stats_numpy(xyr: np.ndarray, edges_uv: np.ndarray) -> dict[str, float]:
    n = int(xyr.shape[0])
    e = int(edges_uv.shape[0])
    deg = degree_numpy(n, edges_uv)
    if e > 0:
        lengths = np.sqrt(((xyr[edges_uv[:, 0], :2] - xyr[edges_uv[:, 1], :2]) ** 2).sum(axis=1))
    else:
        lengths = np.zeros((0,), dtype=np.float32)
    return {
        "n_nodes": float(n),
        "n_edges": float(e),
        "edge_per_node": float(e / max(n, 1)),
        "degree_mean": float(deg.mean()) if n else 0.0,
        "degree_std": float(deg.std()) if n else 0.0,
        "degree_max": float(deg.max()) if n else 0.0,
        "r_mean": float(xyr[:, 2].mean()) if n else 0.0,
        "r_std": float(xyr[:, 2].std()) if n else 0.0,
        "edge_len_mean": float(lengths.mean()) if lengths.size else 0.0,
        "edge_len_std": float(lengths.std()) if lengths.size else 0.0,
    }


def soft_graph_stats_torch(xyr: torch.Tensor, pairs_uv: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    n = int(xyr.shape[0])
    if pairs_uv.numel() == 0:
        soft_e = xyr.new_tensor(0.0)
        deg = xyr.new_zeros((n,))
        mean_len = xyr.new_tensor(0.0)
        std_len = xyr.new_tensor(0.0)
    else:
        p = probs.float().clamp(0.0, 1.0)
        u = pairs_uv[:, 0].long()
        v = pairs_uv[:, 1].long()
        soft_e = p.sum()
        deg = xyr.new_zeros((n,))
        deg.index_add_(0, u, p)
        deg.index_add_(0, v, p)
        lengths = torch.sqrt(((xyr[u, :2] - xyr[v, :2]) ** 2).sum(dim=-1) + 1e-8)
        denom = p.sum().clamp_min(1e-6)
        mean_len = (p * lengths).sum() / denom
        std_len = torch.sqrt((p * (lengths - mean_len) ** 2).sum() / denom + 1e-8)
    r = xyr[:, 2]
    return torch.stack(
        [
            soft_e / max(float(n), 1.0),
            deg.mean(),
            deg.std(unbiased=False),
            r.mean(),
            r.std(unbiased=False),
            mean_len,
            std_len,
        ]
    )


def real_graph_stats_torch(xyr: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
    if edges_uv.numel() == 0:
        probs = xyr.new_zeros((0,))
    else:
        probs = xyr.new_ones((edges_uv.shape[0],))
    return soft_graph_stats_torch(xyr, edges_uv, probs)


def corrupt_graph(xyr: torch.Tensor, edges_uv: torch.Tensor, *, drop_frac: float = 0.35) -> tuple[torch.Tensor, torch.Tensor]:
    x = xyr.clone()
    if x.shape[0] > 1:
        perm = torch.randperm(x.shape[0], device=x.device)
        x[:, 2] = x[perm, 2]
        jitter = 0.03 * torch.randn_like(x[:, :2])
        x[:, :2] = (x[:, :2] + jitter).clamp(0.0, 1.0)
    if edges_uv.numel() == 0:
        return x, edges_uv
    keep = torch.rand((edges_uv.shape[0],), device=edges_uv.device) > float(drop_frac)
    if not bool(keep.any()):
        keep[torch.randint(0, edges_uv.shape[0], (1,), device=edges_uv.device)] = True
    return x, edges_uv[keep]


def mean_abs_stat_delta(real_stats: list[dict[str, float]], fake_stats: list[dict[str, float]]) -> dict[str, float]:
    if not real_stats or not fake_stats:
        return {}
    keys = sorted(set(real_stats[0]).intersection(fake_stats[0]))
    out: dict[str, float] = {}
    for key in keys:
        rv = np.array([float(r[key]) for r in real_stats], dtype=np.float64)
        fv = np.array([float(f[key]) for f in fake_stats], dtype=np.float64)
        out[f"{key}_real_mean"] = float(rv.mean())
        out[f"{key}_fake_mean"] = float(fv.mean())
        out[f"{key}_mean_abs_delta"] = float(abs(rv.mean() - fv.mean()))
    return out
