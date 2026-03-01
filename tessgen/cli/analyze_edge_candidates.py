from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..data import GraphStore, discover_graph_ids
from ..graph_utils import delaunay_candidate_pairs, radius_candidate_pairs
from ..outdirs import finalize_out_dir, make_timestamped_run_dir
from ..reporting import save_line_plot, write_json


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze edge-candidate sets (coverage vs. size) on the tessellation dataset")
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--max_graphs", type=int, default=0, help="0 = all graphs")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max_k", type=int, default=128, help="Max k to consider when computing required kNN coverage")
    p.add_argument(
        "--k_list",
        type=int,
        nargs="*",
        default=[8, 12, 16, 24, 32, 39, 48],
        help="k values to report (coverage/ratio); uses symmetrized kNN candidates",
    )

    p.add_argument(
        "--radius_list",
        type=float,
        nargs="*",
        default=[],
        help="Radius values to report (coverage/ratio). If empty, auto-select quantiles of true-edge max distances.",
    )
    return p.parse_args()


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _safe_median(xs: list[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def _safe_min(xs: list[float]) -> float:
    return float(np.min(xs)) if xs else float("nan")


def _safe_max(xs: list[float]) -> float:
    return float(np.max(xs)) if xs else float("nan")


def _stats(xs: list[float]) -> dict[str, float]:
    return {"mean": _safe_mean(xs), "median": _safe_median(xs), "min": _safe_min(xs), "max": _safe_max(xs)}


def _coverage_and_ratio(*, n_nodes: int, true_edges_uv: np.ndarray, cand_pairs_uv: np.ndarray) -> dict[str, float]:
    e = int(true_edges_uv.shape[0])
    m = int(cand_pairs_uv.shape[0])
    if e == 0:
        return {
            "n_true": float(e),
            "n_cand": float(m),
            "n_hit": 0.0,
            "coverage": 1.0,
            "ratio_true_over_cand": 0.0,
        }

    max_node = int(n_nodes)
    true_code = true_edges_uv[:, 0].astype(np.int64, copy=False) * max_node + true_edges_uv[:, 1].astype(np.int64, copy=False)
    cand_code = cand_pairs_uv[:, 0].astype(np.int64, copy=False) * max_node + cand_pairs_uv[:, 1].astype(np.int64, copy=False)
    hit = int(np.isin(true_code, cand_code).sum())
    cov = float(hit) / float(max(1, e))
    ratio = float(e) / float(max(1, m))
    return {
        "n_true": float(e),
        "n_cand": float(m),
        "n_hit": float(hit),
        "coverage": float(cov),
        "ratio_true_over_cand": float(ratio),
    }


def _knn_neighbors(*, coords01: np.ndarray, max_k: int) -> np.ndarray:
    n = int(coords01.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    k_eff = min(int(max_k) + 1, n)  # +1 includes self
    tree = cKDTree(coords01)
    nn = tree.query(coords01, k=k_eff)[1]  # (N, k_eff)
    if k_eff <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    return nn[:, 1:].astype(np.int64, copy=False)  # (N, k_eff-1)


def _knn_edge_required_k(*, neighbors: np.ndarray, true_edges_uv: np.ndarray, max_k: int) -> np.ndarray:
    n = int(neighbors.shape[0])
    if int(true_edges_uv.shape[0]) == 0:
        return np.zeros((0,), dtype=np.int64)
    if n == 0:
        raise ValueError("neighbors is empty but true_edges_uv is non-empty")

    rank_maps: list[dict[int, int]] = []
    for u in range(n):
        d: dict[int, int] = {}
        for r_idx, v in enumerate(neighbors[u].tolist(), start=1):
            v_i = int(v)
            if v_i == u:
                continue
            if v_i not in d:
                d[v_i] = int(r_idx)
        rank_maps.append(d)

    inf = int(max_k) + 1
    req = np.full((int(true_edges_uv.shape[0]),), inf, dtype=np.int64)
    for i, (u, v) in enumerate(true_edges_uv.tolist()):
        u_i = int(u)
        v_i = int(v)
        ru = int(rank_maps[u_i].get(v_i, inf))
        rv = int(rank_maps[v_i].get(u_i, inf))
        req[i] = min(ru, rv)
    return req


def _knn_candidate_count(*, neighbors: np.ndarray, k: int) -> int:
    n = int(neighbors.shape[0])
    if n == 0:
        return 0
    k_eff = min(int(k), int(neighbors.shape[1]))
    if k_eff <= 0:
        return 0

    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = neighbors[:, :k_eff].reshape(-1).astype(np.int64, copy=False)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return 0
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    codes = u * n + v
    return int(np.unique(codes).size)


def _true_edge_max_radius(*, coords01: np.ndarray, true_edges_uv: np.ndarray) -> float:
    if int(true_edges_uv.shape[0]) == 0:
        return 0.0
    u = true_edges_uv[:, 0].astype(np.int64, copy=False)
    v = true_edges_uv[:, 1].astype(np.int64, copy=False)
    dx = coords01[u] - coords01[v]
    d = np.sqrt((dx**2).sum(axis=1))
    return float(np.max(d)) if d.size else 0.0


@dataclass(frozen=True)
class RadiusEval:
    radius: float
    coverage_mean: float
    coverage_min: float
    ratio_mean: float


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)

    graph_ids = discover_graph_ids(args.tess_root)
    if not graph_ids:
        raise SystemExit("No graphs found under --tess_root")

    if bool(args.shuffle):
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(graph_ids)

    if int(args.max_graphs) > 0:
        graph_ids = graph_ids[: int(args.max_graphs)]
    if not graph_ids:
        raise SystemExit("No graph_ids selected (check --max_graphs)")

    k_list = [int(k) for k in list(args.k_list)]
    if not k_list:
        raise SystemExit("--k_list must be non-empty")
    k_list = sorted(set(k_list))

    max_k = int(args.max_k)
    if max_k <= 0:
        raise SystemExit("--max_k must be > 0")
    if max(k_list) > max_k:
        raise SystemExit(f"--k_list includes k={max(k_list)} > --max_k={max_k}")

    store = GraphStore(args.tess_root)

    rows: list[dict] = []
    delaunay_coverages: list[float] = []
    delaunay_ratios: list[float] = []
    k_needed_all: list[int] = []
    k_needed_over_max: list[int] = []
    max_true_r_all: list[float] = []

    # Aggregates over k_list
    knn_cov_by_k: dict[int, list[float]] = {k: [] for k in k_list}
    knn_ratio_by_k: dict[int, list[float]] = {k: [] for k in k_list}

    for gid in graph_ids:
        g = store.get(int(gid))
        coords01 = g.coords01.numpy()
        true_edges = g.edges_undirected.numpy()
        n_nodes = int(coords01.shape[0])
        n_true = int(true_edges.shape[0])

        # Delaunay (parameter-free)
        cand_del = delaunay_candidate_pairs(coords01)
        del_stats = _coverage_and_ratio(n_nodes=n_nodes, true_edges_uv=true_edges, cand_pairs_uv=cand_del)
        delaunay_coverages.append(float(del_stats["coverage"]))
        delaunay_ratios.append(float(del_stats["ratio_true_over_cand"]))

        # kNN (symmetrized)
        neighbors = _knn_neighbors(coords01=coords01, max_k=max_k)
        req_k_per_edge = _knn_edge_required_k(neighbors=neighbors, true_edges_uv=true_edges, max_k=max_k)
        k_needed = int(req_k_per_edge.max()) if req_k_per_edge.size else 0
        k_needed_all.append(int(k_needed))
        if k_needed > max_k:
            k_needed_over_max.append(int(gid))

        # Coverage and ratio for requested k values
        for k in k_list:
            cov = float(np.mean(req_k_per_edge <= int(k))) if req_k_per_edge.size else 1.0
            m = _knn_candidate_count(neighbors=neighbors, k=int(k))
            ratio = float(n_true) / float(max(1, int(m)))
            knn_cov_by_k[int(k)].append(float(cov))
            knn_ratio_by_k[int(k)].append(float(ratio))

        # Radius needed to not miss any true edge (per-graph)
        r_true_max = _true_edge_max_radius(coords01=coords01, true_edges_uv=true_edges)
        max_true_r_all.append(float(r_true_max))

        rows.append(
            {
                "graph_id": int(gid),
                "n_nodes": int(n_nodes),
                "n_true_edges": int(n_true),
                "delaunay_n_cand": int(del_stats["n_cand"]),
                "delaunay_coverage": float(del_stats["coverage"]),
                "delaunay_ratio_true_over_cand": float(del_stats["ratio_true_over_cand"]),
                "knn_k_needed": int(k_needed),
                "true_edge_max_radius": float(r_true_max),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(str(run_dir / "per_graph.csv"), index=False)

    k_opt = int(max(k_needed_all)) if k_needed_all else 0
    r_opt = float(max(max_true_r_all)) if max_true_r_all else 0.0

    radius_list = [float(r) for r in list(args.radius_list)]
    if not radius_list:
        if max_true_r_all:
            qs = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            radius_list = [float(np.quantile(np.array(max_true_r_all, dtype=np.float32), q)) for q in qs]
        else:
            radius_list = [0.0]
    radius_list = sorted(set(float(r) for r in radius_list))

    # Evaluate radius candidate sets for requested radii (build pairs; needed for ratio)
    rad_evals: list[RadiusEval] = []
    for r in radius_list:
        coverages: list[float] = []
        ratios: list[float] = []
        for gid in graph_ids:
            g = store.get(int(gid))
            coords01 = g.coords01.numpy()
            true_edges = g.edges_undirected.numpy()
            n_nodes = int(coords01.shape[0])
            cand_rad = radius_candidate_pairs(coords01, radius=float(r))
            st = _coverage_and_ratio(n_nodes=n_nodes, true_edges_uv=true_edges, cand_pairs_uv=cand_rad)
            coverages.append(float(st["coverage"]))
            ratios.append(float(st["ratio_true_over_cand"]))
        rad_evals.append(
            RadiusEval(
                radius=float(r),
                coverage_mean=_safe_mean(coverages),
                coverage_min=_safe_min(coverages),
                ratio_mean=_safe_mean(ratios),
            )
        )

    # Plots for kNN coverage and ratio tradeoffs (for k_list)
    k_x = [int(k) for k in k_list]
    cov_mean = [_safe_mean(knn_cov_by_k[int(k)]) for k in k_x]
    cov_min = [_safe_min(knn_cov_by_k[int(k)]) for k in k_x]
    ratio_mean = [_safe_mean(knn_ratio_by_k[int(k)]) for k in k_x]
    save_line_plot(
        out_path=str(run_dir / "figures" / "knn_candidate_coverage.png"),
        x=k_x,
        ys={"mean_coverage": cov_mean, "min_coverage": cov_min},
        title="kNN candidate coverage vs k (symmetrized)",
        xlabel="k",
        ylabel="coverage (true edges covered)",
    )
    save_line_plot(
        out_path=str(run_dir / "figures" / "knn_true_over_cand_ratio.png"),
        x=k_x,
        ys={"mean(true_edges/candidates)": ratio_mean},
        title="kNN candidate density vs k (higher is better)",
        xlabel="k",
        ylabel="true_edges / candidate_edges",
    )

    rad_x = [float(r.radius) for r in rad_evals]
    rad_cov_mean = [float(r.coverage_mean) for r in rad_evals]
    rad_cov_min = [float(r.coverage_min) for r in rad_evals]
    rad_ratio_mean = [float(r.ratio_mean) for r in rad_evals]
    save_line_plot(
        out_path=str(run_dir / "figures" / "radius_candidate_coverage.png"),
        x=rad_x,
        ys={"mean_coverage": rad_cov_mean, "min_coverage": rad_cov_min},
        title="Radius candidate coverage vs r",
        xlabel="radius r",
        ylabel="coverage (true edges covered)",
    )
    save_line_plot(
        out_path=str(run_dir / "figures" / "radius_true_over_cand_ratio.png"),
        x=rad_x,
        ys={"mean(true_edges/candidates)": rad_ratio_mean},
        title="Radius candidate density vs r (higher is better)",
        xlabel="radius r",
        ylabel="true_edges / candidate_edges",
    )

    summary = {
        "task": "analyze_edge_candidates",
        "tess_root": str(args.tess_root),
        "n_graphs": int(len(graph_ids)),
        "params": {"max_k": int(max_k), "k_list": k_x, "radius_list": radius_list},
        "delaunay": {
            "coverage": _stats(delaunay_coverages),
            "ratio_true_over_cand": _stats(delaunay_ratios),
        },
        "knn": {
            "k_needed": {
                "stats": _stats([float(x) for x in k_needed_all]),
                "k_opt_global": int(k_opt),
                "graphs_over_max_k": [int(x) for x in k_needed_over_max],
            },
            "by_k": {
                str(k): {
                    "coverage": _stats(knn_cov_by_k[int(k)]),
                    "ratio_true_over_cand": _stats(knn_ratio_by_k[int(k)]),
                }
                for k in k_x
            },
        },
        "radius": {
            "true_edge_max_radius": _stats(max_true_r_all),
            "r_opt_global": float(r_opt),
            "by_r": [
                {"radius": float(r.radius), "coverage_mean": float(r.coverage_mean), "coverage_min": float(r.coverage_min), "ratio_mean": float(r.ratio_mean)}
                for r in rad_evals
            ],
        },
        "artifacts": {
            "per_graph_csv": str(run_dir / "per_graph.csv"),
            "figures_dir": str(run_dir / "figures"),
        },
        "base_out_dir": str(base_dir),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
    }
    write_json(str(run_dir / "summary.json"), summary)

    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(json.dumps({"run_dir": str(run_dir), "base_out_dir": str(base_dir), "k_opt_global": int(k_opt), "r_opt_global": float(r_opt)}))


if __name__ == "__main__":
    main()
