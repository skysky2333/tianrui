from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import COORD_MAX, COORD_MIN, COORD_RANGE
from .utils import Batch


_NODE_RE = re.compile(r"^Node_(\d+)\.txt$")
_CONN_RE = re.compile(r"^Connection_(\d+)\.txt$")


def normalize_coords(xy: np.ndarray) -> np.ndarray:
    # xy: (N, 2) in [COORD_MIN, COORD_MAX]
    return (xy - COORD_MIN) / COORD_RANGE


def denormalize_coords(xy01: np.ndarray) -> np.ndarray:
    return xy01 * COORD_RANGE + COORD_MIN


def _read_nodes_txt(path: str) -> np.ndarray:
    xs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            xs.append((node_id, x, y))
    xs.sort(key=lambda t: t[0])
    coords = np.array([(x, y) for _, x, y in xs], dtype=np.float32)
    return coords


def _read_connections_txt(path: str, n_nodes: int) -> np.ndarray:
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            u = int(parts[1]) - 1
            v = int(parts[2]) - 1
            if u == v:
                continue
            if not (0 <= u < n_nodes and 0 <= v < n_nodes):
                raise ValueError(f"Edge endpoint out of range in {path}: ({u},{v}) for n={n_nodes}")
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.array(edges, dtype=np.int64)
    # Deduplicate undirected edges
    edges = np.unique(edges, axis=0)
    return edges


def undirected_to_directed_edge_index(edges_undirected: np.ndarray) -> torch.Tensor:
    if edges_undirected.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    u = torch.from_numpy(edges_undirected[:, 0].astype(np.int64, copy=False))
    v = torch.from_numpy(edges_undirected[:, 1].astype(np.int64, copy=False))
    edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    return edge_index


@dataclass(frozen=True)
class Graph:
    coords01: torch.Tensor  # (N, 2) float32 in [0,1]
    edges_undirected: torch.Tensor  # (E, 2) int64 with u<v
    edge_index: torch.Tensor  # (2, 2E) directed

    @property
    def n_nodes(self) -> int:
        return int(self.coords01.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges_undirected.shape[0])


class GraphStore:
    def __init__(self, tess_root: str):
        self.tess_root = Path(tess_root)
        if not self.tess_root.exists():
            raise FileNotFoundError(self.tess_root)

    def node_path(self, graph_id: int) -> Path:
        return self.tess_root / f"Node_{graph_id}.txt"

    def conn_path(self, graph_id: int) -> Path:
        return self.tess_root / f"Connection_{graph_id}.txt"

    @lru_cache(maxsize=256)
    def get(self, graph_id: int) -> Graph:
        node_path = str(self.node_path(graph_id))
        conn_path = str(self.conn_path(graph_id))
        coords = _read_nodes_txt(node_path)
        coords01 = normalize_coords(coords)
        coords01 = np.clip(coords01, 0.0, 1.0)
        n_nodes = coords01.shape[0]
        edges_undirected = _read_connections_txt(conn_path, n_nodes=n_nodes)
        edge_index = undirected_to_directed_edge_index(edges_undirected)
        return Graph(
            coords01=torch.from_numpy(coords01),
            edges_undirected=torch.from_numpy(edges_undirected.astype(np.int64)),
            edge_index=edge_index,
        )


def discover_graph_ids(tess_root: str) -> list[int]:
    root = Path(tess_root)
    node_ids = set()
    conn_ids = set()
    for p in root.iterdir():
        if not p.is_file():
            continue
        m = _NODE_RE.match(p.name)
        if m:
            node_ids.add(int(m.group(1)))
            continue
        m = _CONN_RE.match(p.name)
        if m:
            conn_ids.add(int(m.group(1)))
    ids = sorted(node_ids & conn_ids)
    return ids


def train_val_split_graph_ids(
    graph_ids: Sequence[int],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0,1)")
    rng = np.random.default_rng(seed)
    graph_ids = np.array(list(graph_ids), dtype=np.int64)
    rng.shuffle(graph_ids)
    n_val = int(math.ceil(len(graph_ids) * val_frac))
    val = graph_ids[:n_val].tolist()
    train = graph_ids[n_val:].tolist()
    return train, val


def train_val_test_split_graph_ids(
    graph_ids: Sequence[int],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0,1)")
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be in (0,1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rng = np.random.default_rng(seed)
    graph_ids = np.array(list(graph_ids), dtype=np.int64)
    rng.shuffle(graph_ids)

    n_test = int(math.ceil(len(graph_ids) * test_frac))
    n_val = int(math.ceil(len(graph_ids) * val_frac))
    test = graph_ids[:n_test].tolist()
    val = graph_ids[n_test : n_test + n_val].tolist()
    train = graph_ids[n_test + n_val :].tolist()
    return train, val, test


class TessellationRowDataset(Dataset):
    """
    Each row in the CSV corresponds to (graph_id, RD) → y.

    CSV row i maps to graph_id = i//5 + 1 (dataset convention).
    """

    def __init__(
        self,
        *,
        data_csv: str,
        tess_root: str = "data/Tessellation_Dataset",
        target_cols: Sequence[str],
        cond_cols: Sequence[str] | None = None,
        row_indices: Sequence[int] | None = None,
        cache_graphs: bool = True,
    ):
        self.df = pd.read_csv(data_csv)
        self.data_csv = data_csv
        self.tess_root = tess_root
        self.target_cols = list(target_cols)
        self.cond_cols = list(cond_cols) if cond_cols is not None else list(target_cols)
        if "RD" not in self.df.columns:
            raise ValueError("CSV missing required column RD")
        for c in self.target_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV missing target column {c}")
        for c in self.cond_cols:
            if c not in self.df.columns:
                raise ValueError(f"CSV missing cond column {c}")

        self.graph_store = GraphStore(tess_root=tess_root)
        self._cache_graphs = cache_graphs
        if not cache_graphs:
            self.graph_store.get.cache_clear()  # type: ignore[attr-defined]

        if row_indices is None:
            self.row_indices = list(range(len(self.df)))
        else:
            self.row_indices = list(map(int, row_indices))

    def __len__(self) -> int:
        return len(self.row_indices)

    def graph_id_for_row(self, row_idx: int) -> int:
        return (row_idx // 5) + 1

    def __getitem__(self, i: int) -> dict:
        row_idx = self.row_indices[i]
        r = self.df.iloc[row_idx]
        graph_id = self.graph_id_for_row(row_idx)
        g = self.graph_store.get(graph_id)

        rd = float(r["RD"])
        y = np.array([float(r[c]) for c in self.target_cols], dtype=np.float32)
        cond = np.array([float(r[c]) for c in self.cond_cols], dtype=np.float32)
        logn = float(math.log(float(g.n_nodes)))

        return {
            "graph_id": graph_id,
            "coords01": g.coords01,  # torch float32
            "edge_index": g.edge_index,  # torch long (2, 2E)
            "edges_undirected": g.edges_undirected,  # torch long (E, 2)
            "rd": torch.tensor([rd], dtype=torch.float32),
            "logn": torch.tensor([logn], dtype=torch.float32),
            "y": torch.from_numpy(y),
            "cond": torch.from_numpy(cond),
        }


def collate_graph_batch(samples: Sequence[dict]) -> Batch:
    coords_list = [s["coords01"] for s in samples]
    edge_index_list = [s["edge_index"] for s in samples]
    rds = torch.stack([s["rd"] for s in samples], dim=0)  # (B, 1)
    ys = torch.stack([s["y"] for s in samples], dim=0)

    node_offsets = []
    n_total = 0
    for coords in coords_list:
        node_offsets.append(n_total)
        n_total += int(coords.shape[0])

    x = torch.cat(coords_list, dim=0)  # (N, 2)

    edge_indices = []
    n_edges = []
    for off, ei in zip(node_offsets, edge_index_list):
        edge_indices.append(ei + off)
        n_edges.append(int(ei.shape[1]) // 2)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)

    batch = torch.empty((n_total,), dtype=torch.long)
    for graph_idx, (off, coords) in enumerate(zip(node_offsets, coords_list)):
        n = int(coords.shape[0])
        batch[off : off + n] = graph_idx

    n_nodes = torch.tensor([int(c.shape[0]) for c in coords_list], dtype=torch.long)
    n_edges_t = torch.tensor(n_edges, dtype=torch.long)

    return Batch(
        x=x,
        edge_index=edge_index,
        batch=batch,
        rd=rds,
        y=ys,
        n_nodes=n_nodes,
        n_edges=n_edges_t,
    )


def collate_first(samples: Sequence[Any]) -> Any:
    if len(samples) != 1:
        raise ValueError(f"collate_first expects batch_size=1; got {len(samples)}")
    return samples[0]


def rows_for_graph_ids(df_len: int, graph_ids: Iterable[int]) -> list[int]:
    graph_ids = list(graph_ids)
    rows = []
    for gid in graph_ids:
        start = (gid - 1) * 5
        if start + 4 >= df_len:
            continue
        rows.extend(range(start, start + 5))
    return rows


def coord_stats(tess_root: str, graph_ids: Sequence[int] | None = None) -> dict:
    store = GraphStore(tess_root=tess_root)
    if graph_ids is None:
        graph_ids = discover_graph_ids(tess_root)
    mins = []
    maxs = []
    for gid in graph_ids:
        g = store.get(gid)
        mins.append(g.coords01.min(dim=0).values.numpy())
        maxs.append(g.coords01.max(dim=0).values.numpy())
    mins = np.stack(mins, axis=0)
    maxs = np.stack(maxs, axis=0)
    return {
        "coord01_min": mins.min(axis=0).tolist(),
        "coord01_max": maxs.max(axis=0).tolist(),
        "raw_coord_min": COORD_MIN,
        "raw_coord_max": COORD_MAX,
    }
