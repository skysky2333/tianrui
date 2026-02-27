from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reporting import mpl_setup


@dataclass(frozen=True)
class ParsedGraph:
    coords: np.ndarray  # (N,2) float32
    edges_uv: np.ndarray  # (E,2) int64 u<v, 0-based


def _parse_csv_line(path: str, lineno: int, line: str, *, expected_fields: int) -> list[str]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != expected_fields:
        raise ValueError(f"{path}:{lineno}: expected {expected_fields} comma-separated fields, got {len(parts)}: {line!r}")
    return parts


def read_nodes_txt(path: str) -> np.ndarray:
    rows: list[tuple[int, float, float]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = _parse_csv_line(path, lineno, line, expected_fields=3)
            node_id = int(parts[0])
            if node_id <= 0:
                raise ValueError(f"{path}:{lineno}: node_id must be positive; got {node_id}")
            x = float(parts[1])
            y = float(parts[2])
            rows.append((node_id, x, y))

    if not rows:
        raise ValueError(f"{path}: no nodes parsed")

    rows.sort(key=lambda t: t[0])
    ids = [r[0] for r in rows]
    if ids[0] != 1:
        raise ValueError(f"{path}: node ids must start at 1; got {ids[0]}")
    for expected, got in enumerate(ids, start=1):
        if got != expected:
            raise ValueError(f"{path}: node ids must be contiguous 1..N; missing {expected} (found {got})")

    coords = np.array([(x, y) for _, x, y in rows], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path}: expected coords shape (N,2), got {coords.shape}")
    return coords


def read_connections_txt(path: str, *, n_nodes: int) -> np.ndarray:
    n_nodes = int(n_nodes)
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be > 0; got {n_nodes}")

    edges: list[tuple[int, int]] = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = _parse_csv_line(path, lineno, line, expected_fields=3)
            u1 = int(parts[1])
            v1 = int(parts[2])
            if u1 == v1:
                continue
            if not (1 <= u1 <= n_nodes and 1 <= v1 <= n_nodes):
                raise ValueError(f"{path}:{lineno}: edge endpoint out of range: ({u1},{v1}) for n_nodes={n_nodes}")
            u = u1 - 1
            v = v1 - 1
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b))

    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    edges_uv = np.unique(np.array(edges, dtype=np.int64), axis=0)
    return edges_uv


def load_graph_from_files(*, node_path: str, conn_path: str) -> ParsedGraph:
    coords = read_nodes_txt(node_path)
    edges_uv = read_connections_txt(conn_path, n_nodes=int(coords.shape[0]))
    return ParsedGraph(coords=coords, edges_uv=edges_uv)


def save_graph_figure(
    *,
    coords: np.ndarray,
    edges_uv: np.ndarray,
    out_png: str,
    out_svg: str,
    title: str | None = None,
    dpi: int = 160,
    node_size: float = 8.0,
    edge_lw: float = 0.4,
    edge_alpha: float = 0.35,
    equal_axis: bool = True,
    hide_axes: bool = True,
) -> None:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N,2); got {coords.shape}")
    if edges_uv.ndim != 2 or edges_uv.shape[1] != 2:
        raise ValueError(f"edges_uv must be (E,2); got {edges_uv.shape}")

    out_png = str(out_png) if out_png else ""
    out_svg = str(out_svg) if out_svg else ""
    if not out_png and not out_svg:
        raise ValueError("At least one of out_png or out_svg must be provided")

    mpl_setup()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    if out_svg:
        Path(out_svg).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if edges_uv.size > 0:
        segs = coords[edges_uv]  # (E,2,2)
        lc = LineCollection(segs, colors="black", linewidths=float(edge_lw), alpha=float(edge_alpha))
        ax.add_collection(lc)

    ax.scatter(coords[:, 0], coords[:, 1], s=float(node_size), c="tab:blue", alpha=0.9, linewidths=0.0)

    if equal_axis:
        ax.set_aspect("equal", adjustable="box")

    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))
    ymin = float(np.min(coords[:, 1]))
    ymax = float(np.max(coords[:, 1]))
    dx = xmax - xmin
    dy = ymax - ymin
    pad = 0.05 * float(max(dx, dy, 1e-6))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    n_nodes = int(coords.shape[0])
    n_edges = int(edges_uv.shape[0])
    ax.text(0.01, 0.01, f"N={n_nodes}  E={n_edges}", transform=ax.transAxes, fontsize=10, ha="left", va="bottom")

    if title is not None:
        ax.set_title(str(title))

    if hide_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=int(dpi))
    if out_svg:
        fig.savefig(out_svg)
    plt.close(fig)
