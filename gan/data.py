from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SEED_RE = re.compile(r"^Seed_(\d+)\.txt$")
CONN_RE = re.compile(r"^Connection_(\d+)\.txt$")


@dataclass(frozen=True)
class GraphData:
    graph_id: int
    xyr: torch.Tensor  # (N,3): x,y,r in source units, expected near [0,1]
    edges_undirected: torch.Tensor  # (E,2), 0-based, u<v

    @property
    def n_nodes(self) -> int:
        return int(self.xyr.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges_undirected.shape[0])


def _parse_float_row(path: Path, lineno: int, line: str) -> list[float]:
    parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
    if not parts:
        raise ValueError(f"{path}:{lineno}: empty comma-separated row")
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"{path}:{lineno}: failed to parse float row") from e


def _parse_int_row(path: Path, lineno: int, line: str) -> list[int]:
    parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
    if not parts:
        raise ValueError(f"{path}:{lineno}: empty comma-separated row")
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"{path}:{lineno}: failed to parse integer row") from e


def read_seed_txt(path: str | Path) -> np.ndarray:
    path = Path(path)
    rows: list[list[float]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(_parse_float_row(path, lineno, line))

    if len(rows) != 3:
        raise ValueError(f"{path}: expected exactly 3 non-empty rows (x,y,r), got {len(rows)}")
    n = len(rows[0])
    if n <= 0:
        raise ValueError(f"{path}: seed file has no nodes")
    if len(rows[1]) != n or len(rows[2]) != n:
        raise ValueError(f"{path}: x/y/r row lengths differ: {[len(r) for r in rows]}")
    xyr = np.stack(rows, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(xyr).all():
        raise ValueError(f"{path}: seed file contains non-finite values")
    return xyr


def read_seed_n_nodes(path: str | Path) -> int:
    path = Path(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return len([p for p in line.split(",") if p.strip() != ""])
    raise ValueError(f"{path}: seed file has no non-empty rows")


def read_connection_txt(path: str | Path, *, n_nodes: int) -> np.ndarray:
    path = Path(path)
    rows: list[list[int]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(_parse_int_row(path, lineno, line))

    if len(rows) != 2:
        raise ValueError(f"{path}: expected exactly 2 non-empty endpoint rows, got {len(rows)}")
    if len(rows[0]) != len(rows[1]):
        raise ValueError(f"{path}: endpoint row lengths differ: {[len(r) for r in rows]}")

    n_nodes = int(n_nodes)
    edges: list[tuple[int, int]] = []
    for idx, (u1, v1) in enumerate(zip(rows[0], rows[1]), start=1):
        if not (1 <= u1 <= n_nodes and 1 <= v1 <= n_nodes):
            raise ValueError(f"{path}: endpoint {idx} out of range ({u1},{v1}) for n_nodes={n_nodes}")
        if u1 == v1:
            continue
        u = int(u1) - 1
        v = int(v1) - 1
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.array(edges, dtype=np.int64), axis=0)


def write_seed_txt(path: str | Path, xyr: np.ndarray | torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = xyr.detach().cpu().numpy() if isinstance(xyr, torch.Tensor) else np.asarray(xyr)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"xyr must be (N,3), got {arr.shape}")
    with path.open("w") as f:
        for j in range(3):
            f.write(",".join(f"{float(v):.8g}" for v in arr[:, j].tolist()) + "\n")


def write_connection_txt(path: str | Path, edges_uv: np.ndarray | torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = edges_uv.detach().cpu().numpy() if isinstance(edges_uv, torch.Tensor) else np.asarray(edges_uv)
    if arr.size == 0:
        rows = [[], []]
    else:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"edges_uv must be (E,2), got {arr.shape}")
        rows = [(arr[:, 0] + 1).astype(np.int64).tolist(), (arr[:, 1] + 1).astype(np.int64).tolist()]
    with path.open("w") as f:
        f.write(",".join(str(int(v)) for v in rows[0]) + "\n")
        f.write(",".join(str(int(v)) for v in rows[1]) + "\n")


def discover_graph_ids(data_root: str | Path, *, max_graphs: int = 0) -> list[int]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(root)
    seed_ids: set[int] = set()
    conn_ids: set[int] = set()
    for p in root.iterdir():
        if not p.is_file():
            continue
        m = SEED_RE.match(p.name)
        if m:
            seed_ids.add(int(m.group(1)))
            continue
        m = CONN_RE.match(p.name)
        if m:
            conn_ids.add(int(m.group(1)))
    ids = sorted(seed_ids & conn_ids)
    if int(max_graphs) > 0:
        ids = ids[: int(max_graphs)]
    return ids


def split_graph_ids(
    graph_ids: Sequence[int],
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not (0.0 <= float(val_frac) < 1.0):
        raise ValueError("val_frac must be in [0,1)")
    if not (0.0 <= float(test_frac) < 1.0):
        raise ValueError("test_frac must be in [0,1)")
    if float(val_frac) + float(test_frac) >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    ids = np.array(list(map(int, graph_ids)), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(ids)
    n_test = int(math.ceil(len(ids) * float(test_frac))) if test_frac > 0 else 0
    n_val = int(math.ceil(len(ids) * float(val_frac))) if val_frac > 0 else 0
    test = ids[:n_test].tolist()
    val = ids[n_test : n_test + n_val].tolist()
    train = ids[n_test + n_val :].tolist()
    return train, val, test


class GraphStore:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(self.data_root)

    def seed_path(self, graph_id: int) -> Path:
        return self.data_root / f"Seed_{int(graph_id)}.txt"

    def conn_path(self, graph_id: int) -> Path:
        return self.data_root / f"Connection_{int(graph_id)}.txt"

    @lru_cache(maxsize=256)
    def get(self, graph_id: int) -> GraphData:
        graph_id = int(graph_id)
        xyr = read_seed_txt(self.seed_path(graph_id))
        edges = read_connection_txt(self.conn_path(graph_id), n_nodes=int(xyr.shape[0]))
        return GraphData(
            graph_id=graph_id,
            xyr=torch.from_numpy(xyr),
            edges_undirected=torch.from_numpy(edges.astype(np.int64, copy=False)),
        )


def _subsample_graph(g: GraphData, *, max_nodes: int) -> GraphData:
    max_nodes = int(max_nodes)
    if max_nodes <= 0 or g.n_nodes <= max_nodes:
        return g
    perm = torch.randperm(g.n_nodes)[:max_nodes]
    perm, _ = torch.sort(perm)
    old_to_new = torch.full((g.n_nodes,), -1, dtype=torch.long)
    old_to_new[perm] = torch.arange(max_nodes, dtype=torch.long)
    edges = g.edges_undirected
    if edges.numel() == 0:
        new_edges = edges
    else:
        keep = (old_to_new[edges[:, 0]] >= 0) & (old_to_new[edges[:, 1]] >= 0)
        kept = edges[keep]
        if kept.numel() == 0:
            new_edges = torch.zeros((0, 2), dtype=torch.long)
        else:
            new_edges = torch.stack([old_to_new[kept[:, 0]], old_to_new[kept[:, 1]]], dim=1)
    return GraphData(graph_id=g.graph_id, xyr=g.xyr[perm], edges_undirected=new_edges)


class GraphDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str | Path,
        graph_ids: Sequence[int],
        max_nodes_per_graph: int = 0,
        read_connections: bool = True,
    ):
        self.data_root = str(data_root)
        self.graph_ids = list(map(int, graph_ids))
        self.max_nodes_per_graph = int(max_nodes_per_graph)
        self.read_connections = bool(read_connections)
        self.store = GraphStore(data_root)

    def __len__(self) -> int:
        return len(self.graph_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        graph_id = int(self.graph_ids[idx])
        if self.read_connections:
            g = self.store.get(graph_id)
        else:
            xyr = torch.from_numpy(read_seed_txt(Path(self.data_root) / f"Seed_{graph_id}.txt"))
            g = GraphData(graph_id=graph_id, xyr=xyr, edges_undirected=torch.zeros((0, 2), dtype=torch.long))
        g = _subsample_graph(g, max_nodes=self.max_nodes_per_graph)
        return {
            "graph_id": g.graph_id,
            "xyr": g.xyr,
            "edges_undirected": g.edges_undirected,
            "n_nodes": g.n_nodes,
            "n_edges": g.n_edges,
        }


def collate_first(samples: Sequence[Any]) -> Any:
    if len(samples) != 1:
        raise ValueError(f"collate_first expects batch_size=1; got {len(samples)}")
    return samples[0]


def validate_graph_pair(data_root: str | Path, graph_id: int) -> dict[str, int | float]:
    store = GraphStore(data_root)
    g = store.get(int(graph_id))
    xyr = g.xyr.numpy()
    edges = g.edges_undirected.numpy()
    if xyr.ndim != 2 or xyr.shape[1] != 3:
        raise ValueError(f"Seed_{graph_id}.txt parsed to invalid shape {xyr.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Connection_{graph_id}.txt parsed to invalid shape {edges.shape}")
    if edges.size > 0:
        if int(edges.min()) < 0 or int(edges.max()) >= int(xyr.shape[0]):
            raise ValueError(f"Connection_{graph_id}.txt contains endpoint outside 0..N-1 after parsing")
    return {
        "graph_id": int(graph_id),
        "n_nodes": int(xyr.shape[0]),
        "n_edges": int(edges.shape[0]),
        "x_min": float(xyr[:, 0].min()),
        "x_max": float(xyr[:, 0].max()),
        "y_min": float(xyr[:, 1].min()),
        "y_max": float(xyr[:, 1].max()),
        "r_min": float(xyr[:, 2].min()),
        "r_max": float(xyr[:, 2].max()),
    }
