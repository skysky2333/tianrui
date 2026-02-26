from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


def knn_candidate_pairs(coords01: np.ndarray, k: int) -> np.ndarray:
    """
    Build an undirected candidate set of edges via kNN.

    Returns pairs (u,v) with u<v, shape (M,2).
    """
    if coords01.ndim != 2 or coords01.shape[1] != 2:
        raise ValueError("coords01 must be (N,2)")
    n = coords01.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    k = int(k)
    if k <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    k_eff = min(k + 1, n)  # +1 includes self
    tree = cKDTree(coords01)
    nn = tree.query(coords01, k=k_eff)[1]  # (N, k_eff)
    if k_eff <= 1:
        return np.zeros((0, 2), dtype=np.int64)

    src = np.repeat(np.arange(n, dtype=np.int64), k_eff - 1)
    dst = nn[:, 1:].reshape(-1).astype(np.int64, copy=False)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    pairs = np.stack([u, v], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs


def pairs_to_edge_index(pairs_undirected: np.ndarray) -> torch.Tensor:
    if pairs_undirected.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    u = torch.from_numpy(pairs_undirected[:, 0].astype(np.int64, copy=False))
    v = torch.from_numpy(pairs_undirected[:, 1].astype(np.int64, copy=False))
    return torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)


def edges_undirected_to_set(edges_uv: np.ndarray) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    for u, v in edges_uv.tolist():
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        s.add((a, b))
    return s


@dataclass
class UnionFind:
    parent: np.ndarray
    rank: np.ndarray

    @classmethod
    def make(cls, n: int) -> "UnionFind":
        parent = np.arange(n, dtype=np.int64)
        rank = np.zeros(n, dtype=np.int64)
        return cls(parent=parent, rank=rank)

    def find(self, x: int) -> int:
        # path compression
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(int(p))
        return int(self.parent[x])

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def is_connected(n_nodes: int, edges_undirected: np.ndarray) -> bool:
    if n_nodes <= 1:
        return True
    uf = UnionFind.make(n_nodes)
    for u, v in edges_undirected.tolist():
        uf.union(int(u), int(v))
    root = uf.find(0)
    for i in range(1, n_nodes):
        if uf.find(i) != root:
            return False
    return True


def enforce_degree_cap(
    n_nodes: int,
    pairs_uv: np.ndarray,
    probs: np.ndarray,
    *,
    deg_cap: int,
) -> np.ndarray:
    """
    Greedy keep edges in descending prob order, respecting max degree per node.
    """
    if pairs_uv.size == 0:
        return pairs_uv
    order = np.argsort(-probs)
    deg = np.zeros(n_nodes, dtype=np.int64)
    keep = []
    for idx in order.tolist():
        u, v = pairs_uv[idx]
        if deg[u] >= deg_cap or deg[v] >= deg_cap:
            continue
        keep.append((int(u), int(v)))
        deg[u] += 1
        deg[v] += 1
    if not keep:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(keep, dtype=np.int64)


def ensure_connected_by_candidates(
    n_nodes: int,
    edges_uv: np.ndarray,
    cand_uv: np.ndarray,
    cand_probs: np.ndarray,
) -> np.ndarray:
    """
    Add high-prob candidate edges until the graph is connected.
    """
    if n_nodes <= 1:
        return edges_uv
    uf = UnionFind.make(n_nodes)
    for u, v in edges_uv.tolist():
        uf.union(int(u), int(v))
    # If already connected, return early
    root0 = uf.find(0)
    if all(uf.find(i) == root0 for i in range(n_nodes)):
        return edges_uv

    chosen = set(tuple(map(int, e)) for e in edges_uv.tolist())
    order = np.argsort(-cand_probs)
    added = []
    for idx in order.tolist():
        u, v = cand_uv[idx]
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in chosen:
            continue
        if uf.union(a, b):
            chosen.add((a, b))
            added.append((a, b))
            root0 = uf.find(0)
            if all(uf.find(i) == root0 for i in range(n_nodes)):
                break

    if not added:
        return edges_uv
    return np.concatenate([edges_uv, np.array(added, dtype=np.int64)], axis=0)
