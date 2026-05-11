from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from tessgen.reporting import read_jsonl, save_line_plot

from .data import GraphDataset, collate_first, write_connection_txt, write_seed_txt
from .graph_ops import candidate_pairs_from_xyr, sample_empirical_n, undirected_to_directed
from .metrics import corrupt_graph, graph_stats_numpy, mean_abs_stat_delta
from .models.common import normalize_xyr
from .viz import save_graph_grid, save_score_histogram


def _realism_logit(lit, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
    if hasattr(lit, "discriminator"):
        edge_index, edge_weight = undirected_to_directed(edges_uv)
        return lit.discriminator(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)
    if hasattr(lit, "critic"):
        edge_index, edge_weight = undirected_to_directed(edges_uv)
        return lit.critic(xyr01=xyr01, edge_index=edge_index, edge_weight=edge_weight)
    raise ValueError("lit module has no discriminator or critic")


def _score_edges_for_lit(lit, xyr01: torch.Tensor, edges_uv: torch.Tensor) -> torch.Tensor:
    if bool(getattr(lit.hparams, "learn_edges", False)):
        return edges_uv
    pairs = candidate_pairs_from_xyr(
        xyr01.detach(),
        candidate_mode=str(lit.hparams.candidate_mode),
        k=int(lit.hparams.k_edge),
    )
    return torch.as_tensor(pairs, dtype=torch.long, device=xyr01.device)


@torch.no_grad()
def score_dataset(
    lit,
    *,
    data_root: str,
    graph_ids: list[int],
    max_graphs: int,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    ids = list(map(int, graph_ids))
    if int(max_graphs) > 0:
        ids = ids[: int(max_graphs)]
    read_connections = bool(getattr(lit.hparams, "learn_edges", False))
    dl = DataLoader(
        GraphDataset(data_root=data_root, graph_ids=ids, read_connections=read_connections),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_first,
        num_workers=int(num_workers),
    )
    rows: list[dict[str, Any]] = []
    for batch in dl:
        xyr01 = normalize_xyr(batch["xyr"].to(lit.device), r_scale=float(lit.r_scale))
        edges_uv = _score_edges_for_lit(lit, xyr01, batch["edges_undirected"].to(lit.device))
        logit = _realism_logit(lit, xyr01, edges_uv)
        rows.append(
            {
                "graph_id": int(batch["graph_id"]),
                "n_nodes": int(batch["n_nodes"]),
                "n_edges": int(batch["n_edges"]),
                "realism_logit": float(logit.detach().cpu()),
                "realism_prob": float(torch.sigmoid(logit).detach().cpu()),
            }
        )
    return rows


@torch.no_grad()
def make_training_report(
    lit,
    *,
    run_dir: str | Path,
    data_root: str,
    real_graph_ids: list[int],
    n_values: list[int],
    edge_threshold: float,
    sample_steps: int,
    report_max_graphs: int,
    num_workers: int = 0,
    read_connections: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    figures = run_dir / "figures"
    generated_dir = run_dir / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    ids = list(map(int, real_graph_ids))
    if int(report_max_graphs) > 0:
        ids = ids[: int(report_max_graphs)]
    if not ids:
        return {"n_eval_graphs": 0}

    rng = np.random.default_rng(0)
    sample_ns = sample_empirical_n(n_values, rng=rng, n_samples=len(ids))
    dl = DataLoader(
        GraphDataset(data_root=data_root, graph_ids=ids, read_connections=read_connections),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_first,
        num_workers=int(num_workers),
    )

    real_scores: list[float] = []
    fake_scores: list[float] = []
    corrupt_scores: list[float] = []
    real_stats: list[dict[str, float]] = []
    fake_stats: list[dict[str, float]] = []
    real_grid: list[tuple[np.ndarray, np.ndarray, str]] = []
    fake_grid: list[tuple[np.ndarray, np.ndarray, str]] = []

    for i, batch in enumerate(dl):
        xyr = batch["xyr"].to(lit.device)
        xyr01 = normalize_xyr(xyr, r_scale=float(lit.r_scale))
        edges_uv = _score_edges_for_lit(lit, xyr01, batch["edges_undirected"].to(lit.device))
        real_logit = _realism_logit(lit, xyr01, edges_uv)
        x_c, e_c = corrupt_graph(xyr01, edges_uv)
        corrupt_logit = _realism_logit(lit, x_c, e_c)
        fake_xyr, fake_edges = lit.sample_graph(
            n_nodes=int(sample_ns[i]),
            edge_threshold=float(edge_threshold),
            sample_steps=int(sample_steps),
        ) if lit.__class__.__name__.startswith("Diffusion") else lit.sample_graph(
            n_nodes=int(sample_ns[i]),
            edge_threshold=float(edge_threshold),
        )
        fake_xyr01 = normalize_xyr(fake_xyr.to(lit.device), r_scale=float(lit.r_scale))
        fake_logit = _realism_logit(lit, fake_xyr01, fake_edges.to(lit.device))

        real_scores.append(float(torch.sigmoid(real_logit).detach().cpu()))
        corrupt_scores.append(float(torch.sigmoid(corrupt_logit).detach().cpu()))
        fake_scores.append(float(torch.sigmoid(fake_logit).detach().cpu()))

        real_np = batch["xyr"].detach().cpu().numpy()
        real_edges_np = edges_uv.detach().cpu().numpy()
        fake_np = fake_xyr.detach().cpu().numpy()
        fake_edges_np = fake_edges.detach().cpu().numpy()
        real_stats.append(graph_stats_numpy(real_np, real_edges_np))
        fake_stats.append(graph_stats_numpy(fake_np, fake_edges_np))

        if i < 12:
            real_grid.append((real_np, real_edges_np, f"real {int(batch['graph_id'])}"))
            fake_grid.append((fake_np, fake_edges_np, f"synthetic {i}"))
            write_seed_txt(generated_dir / f"Seed_gen_{i}.txt", fake_np)
            write_connection_txt(generated_dir / f"Connection_gen_{i}.txt", fake_edges_np)

    y_true = [1] * len(real_scores) + [0] * len(fake_scores) + [0] * len(corrupt_scores)
    scores = real_scores + fake_scores + corrupt_scores
    y_pred = [1 if s >= 0.5 else 0 for s in scores]
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if y_true else float("nan")

    save_graph_grid(graphs=real_grid, out_png=str(figures / "real_grid.png"), cols=4)
    save_graph_grid(graphs=fake_grid, out_png=str(figures / "synthetic_grid.png"), cols=4)
    save_score_histogram(
        out_png=str(figures / "realism_score_hist.png"),
        real_scores=real_scores,
        fake_scores=fake_scores,
        corrupt_scores=corrupt_scores,
        title="Realism Scores",
    )
    _plot_history(run_dir)

    report = {
        "n_eval_graphs": len(ids),
        "score_auc_real_vs_fake_corrupt": auc,
        "score_balanced_accuracy_at_0_5": bal_acc,
        "real_score_mean": float(np.mean(real_scores)) if real_scores else None,
        "fake_score_mean": float(np.mean(fake_scores)) if fake_scores else None,
        "corrupt_score_mean": float(np.mean(corrupt_scores)) if corrupt_scores else None,
        "stat_deltas": mean_abs_stat_delta(real_stats, fake_stats),
        "figures": {
            "real_grid": str(figures / "real_grid.png"),
            "synthetic_grid": str(figures / "synthetic_grid.png"),
            "realism_score_hist": str(figures / "realism_score_hist.png"),
        },
        "generated_dir": str(generated_dir),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _plot_history(run_dir: Path) -> None:
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return
    rows = read_jsonl(str(history_path))
    if not rows:
        return
    epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
    series_groups: dict[str, dict[str, list[float]]] = {
        "loss": {},
        "prob": {},
    }
    for key in sorted({k for r in rows for k in r.keys()}):
        vals = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))]
        if not vals:
            continue
        if "loss" in key:
            series_groups["loss"][key] = vals
        if key.endswith("_prob") or key.endswith("/real_prob") or key.endswith("/fake_prob"):
            series_groups["prob"][key] = vals
    figures = run_dir / "figures"
    if series_groups["loss"]:
        save_line_plot(
            out_path=str(figures / "loss_curves.png"),
            x=epochs,
            ys=series_groups["loss"],
            title="Training Losses",
            xlabel="epoch",
            ylabel="loss",
            y_scale="symlog",
        )
    if series_groups["prob"]:
        save_line_plot(
            out_path=str(figures / "score_curves.png"),
            x=epochs,
            ys=series_groups["prob"],
            title="Realism Probabilities",
            xlabel="epoch",
            ylabel="probability",
        )
