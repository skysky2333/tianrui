from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve as roc_curve_fn  # type: ignore

from .lit_module import EdgeLitModule
from .core import label_candidate_pairs
from ...graph_utils import knn_candidate_pairs, pairs_to_edge_index
from ...reporting import mpl_setup, read_jsonl, save_histogram, save_line_plot, write_json


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


@torch.no_grad()
def eval_edge_model(
    *,
    lit: EdgeLitModule,
    dl,
    thr: float,
    max_pairs: int = 2_000_000,
) -> dict:
    lit.eval()
    probs_all = []
    y_all = []
    bces = []
    coverages = []

    for sample in dl:
        coords01_cpu = sample["coords01"]
        true_edges = sample["edges_undirected"].numpy()
        cand = knn_candidate_pairs(coords01_cpu.numpy(), k=lit.k)
        if cand.shape[0] == 0:
            continue
        y = label_candidate_pairs(cand, true_edges)
        coverage = float(np.sum(y)) / float(max(1, true_edges.shape[0]))

        coords01 = coords01_cpu.to(lit.device)
        cand_t = torch.from_numpy(cand).to(device=lit.device, dtype=torch.long)
        y_t = torch.from_numpy(y).to(device=lit.device, dtype=torch.float32)
        msg_edge_index = pairs_to_edge_index(cand).to(lit.device)
        logits = lit.model(coords01=coords01, msg_edge_index=msg_edge_index, cand_pairs_uv=cand_t)
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t).detach().cpu())
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        bces.append(bce)
        coverages.append(coverage)
        probs_all.append(probs)
        y_all.append(y)

        if sum(map(len, probs_all)) >= max_pairs:
            break

    if probs_all:
        probs_cat = np.concatenate(probs_all, axis=0)
        y_cat = np.concatenate(y_all, axis=0)
    else:
        probs_cat = np.zeros((0,), dtype=np.float32)
        y_cat = np.zeros((0,), dtype=np.float32)

    pred = probs_cat >= float(thr)
    yb = y_cat > 0.5
    tp = float(np.sum(pred & yb))
    fp = float(np.sum(pred & (~yb)))
    fn = float(np.sum((~pred) & yb))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-12)

    # Ranking metrics (may be nan if only one class)
    ap = float("nan")
    auc = float("nan")
    pr_curve = None
    roc_curve = None
    if len(probs_cat) and yb.any() and (~yb).any():
        ap = float(average_precision_score(y_cat, probs_cat))
        auc = float(roc_auc_score(y_cat, probs_cat))
        pr = precision_recall_curve(y_cat, probs_cat)
        rc = roc_curve_fn(y_cat, probs_cat)
        pr_curve = {"precision": pr[0].tolist(), "recall": pr[1].tolist(), "thresholds": pr[2].tolist()}
        roc_curve = {"fpr": rc[0].tolist(), "tpr": rc[1].tolist(), "thresholds": rc[2].tolist()}

    return {
        "pairs_evaluated": int(len(probs_cat)),
        "bce_mean": _safe_mean(bces),
        "candidate_coverage_mean": _safe_mean(coverages),
        "thr": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(ap),
        "auc": float(auc),
        "pr_curve": pr_curve,
        "roc_curve": roc_curve,
        "probs": probs_cat,
        "labels": y_cat,
    }


def make_report_and_figures(
    *,
    run_dir: str,
    history_path: str,
    test_eval: dict,
) -> None:
    run = Path(run_dir)
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Loss curves
    hist = read_jsonl(history_path)
    epochs = [int(r["epoch"]) for r in hist]
    train = [float(r["train/bce"]) for r in hist]
    val = [float(r["val/bce"]) for r in hist]
    save_line_plot(
        out_path=str(figs / "loss_bce.png"),
        x=epochs,
        ys={"train/bce": train, "val/bce": val},
        title="Edge model loss (BCE)",
        xlabel="epoch",
        ylabel="BCE",
    )
    save_line_plot(
        out_path=str(figs / "loss_bce_logy.png"),
        x=epochs,
        ys={"train/bce": train, "val/bce": val},
        title="Edge model loss (BCE, log y)",
        xlabel="epoch",
        ylabel="BCE",
        y_scale="log",
    )

    # Preview metrics (if enabled during training)
    if any("val/preview_mean_deg_pred" in r for r in hist):
        deg_true = [float(r.get("val/preview_mean_deg_true", float("nan"))) for r in hist]
        deg_pred = [float(r.get("val/preview_mean_deg_pred", float("nan"))) for r in hist]
        save_line_plot(
            out_path=str(figs / "preview_mean_degree.png"),
            x=epochs,
            ys={"val/preview_mean_deg_true": deg_true, "val/preview_mean_deg_pred": deg_pred},
            title="Edge preview: mean degree",
            xlabel="epoch",
            ylabel="mean degree",
        )
    if any("val/preview_edge_ratio_mean" in r for r in hist):
        ratio = [float(r.get("val/preview_edge_ratio_mean", float("nan"))) for r in hist]
        save_line_plot(
            out_path=str(figs / "preview_edge_ratio.png"),
            x=epochs,
            ys={"val/preview_edge_ratio_mean": ratio},
            title="Edge preview: edge count ratio (pred/true)",
            xlabel="epoch",
            ylabel="ratio",
        )

    probs = test_eval.pop("probs")
    labels = test_eval.pop("labels")

    # Prob hist
    if len(probs) > 0:
        pos = probs[labels > 0.5].tolist()
        neg = probs[labels <= 0.5].tolist()
        if pos:
            save_histogram(out_path=str(figs / "prob_hist_pos.png"), values=pos, title="P(edge) for positives", xlabel="p", bins=60)
        if neg:
            save_histogram(out_path=str(figs / "prob_hist_neg.png"), values=neg, title="P(edge) for negatives", xlabel="p", bins=60)

    # PR / ROC curves
    mpl_setup()
    import matplotlib.pyplot as plt

    if test_eval["pr_curve"] is not None:
        pr = test_eval["pr_curve"]
        plt.figure(figsize=(5, 4))
        plt.plot(pr["recall"], pr["precision"])
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("Precision-Recall curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(figs / "pr_curve.png"), dpi=160)
        plt.close()
    if test_eval["roc_curve"] is not None:
        rc = test_eval["roc_curve"]
        plt.figure(figsize=(5, 4))
        plt.plot(rc["fpr"], rc["tpr"])
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("ROC curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(figs / "roc_curve.png"), dpi=160)
        plt.close()

    report = {"task": "edge_model", "test": test_eval, "figures_dir": str(figs)}
    write_json(str(run / "report.json"), report)
