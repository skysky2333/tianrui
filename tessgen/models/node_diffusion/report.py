from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .lit_module import NodeDiffusionLitModule
from ...graph_utils import knn_candidate_pairs, pairs_to_edge_index
from ...reporting import read_jsonl, save_bar_plot_both, save_line_plot, write_json


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


@torch.no_grad()
def eval_node_diffusion(
    *,
    lit: NodeDiffusionLitModule,
    dl,
    max_samples: int = 0,
) -> dict:
    lit.eval()
    losses = []
    diffs = []
    n_all = []

    for i, sample in enumerate(dl):
        if max_samples and i >= int(max_samples):
            break
        coords0_cpu = sample["coords01"]
        N = int(coords0_cpu.shape[0])
        cond_z = lit._cond_to_z(sample)  # noqa: SLF001

        t = torch.randint(low=0, high=lit.schedule.n_steps, size=(1,), device=lit.device, dtype=torch.long)
        eps_cpu = torch.randn_like(coords0_cpu)
        ab = lit.schedule.alpha_bars[t].detach().cpu().view(1, 1)
        x_t_cpu = torch.sqrt(ab) * coords0_cpu + torch.sqrt(1.0 - ab) * eps_cpu
        cand = knn_candidate_pairs(x_t_cpu.numpy(), k=int(lit.k_nn))
        edge_index = pairs_to_edge_index(cand).to(lit.device)
        x_t = x_t_cpu.to(lit.device)
        eps = eps_cpu.to(lit.device)
        eps_pred = lit.denoiser(x_t=x_t, t=t, cond=cond_z, edge_index=edge_index)
        diff_loss = torch.mean((eps_pred - eps) ** 2)

        loss = diff_loss
        losses.append(float(loss.detach().cpu()))
        diffs.append(float(diff_loss.detach().cpu()))
        n_all.append(float(N))

    return {
        "samples": int(min(len(dl), int(max_samples)) if max_samples else len(dl)),
        "loss_mean": _safe_mean(losses),
        "diff_mse_mean": _safe_mean(diffs),
        "n_nodes": {
            "mean": float(np.mean(n_all)) if n_all else float("nan"),
            "min": float(np.min(n_all)) if n_all else float("nan"),
            "max": float(np.max(n_all)) if n_all else float("nan"),
        },
    }


def make_report_and_figures(*, run_dir: str, history_path: str, test_eval: dict, cycle_eval: dict | None = None) -> None:
    run = Path(run_dir)
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    hist = read_jsonl(history_path)
    epochs = [int(r["epoch"]) for r in hist]
    train = [float(r["train/loss"]) for r in hist]
    val = [float(r["val/loss"]) for r in hist]
    save_line_plot(
        out_path=str(figs / "loss.png"),
        x=epochs,
        ys={"train/loss": train, "val/loss": val},
        title="Node diffusion loss",
        xlabel="epoch",
        ylabel="loss",
    )
    save_line_plot(
        out_path=str(figs / "loss_symlog.png"),
        x=epochs,
        ys={"train/loss": train, "val/loss": val},
        title="Node diffusion loss (symlog y)",
        xlabel="epoch",
        ylabel="loss",
        y_scale="symlog",
    )

    train_d = [float(r["train/diff_mse"]) for r in hist]
    val_d = [float(r["val/diff_mse"]) for r in hist]
    save_line_plot(
        out_path=str(figs / "diff_mse.png"),
        x=epochs,
        ys={"train/diff_mse": train_d, "val/diff_mse": val_d},
        title="Diffusion epsilon MSE",
        xlabel="epoch",
        ylabel="MSE",
    )
    save_line_plot(
        out_path=str(figs / "diff_mse_logy.png"),
        x=epochs,
        ys={"train/diff_mse": train_d, "val/diff_mse": val_d},
        title="Diffusion epsilon MSE (log y)",
        xlabel="epoch",
        ylabel="MSE",
        y_scale="log",
    )

    cycle_series = {}
    for k in ["val/cycle_r_true_graph", "val/cycle_r_single", "val/cycle_r_best"]:
        if any(k in r for r in hist):
            cycle_series[k] = [float(r.get(k, float("nan"))) for r in hist]
    if cycle_series:
        save_line_plot(
            out_path=str(figs / "cycle_r_over_epoch.png"),
            x=epochs,
            ys=cycle_series,
            title="Cycle eval Pearson r (validation)",
            xlabel="epoch",
            ylabel="pearson_r",
        )

    if cycle_eval is not None:
        m = cycle_eval["metrics"]
        labels = ["true_graph", "single", "best"]
        rs = [float(m[k]["pearson_r"]) for k in labels]
        save_bar_plot_both(
            out_png=str(figs / "cycle_pearson_r.png"),
            out_svg=str(figs / "cycle_pearson_r.svg"),
            labels=labels,
            values=rs,
            title="Cycle eval Pearson r (RS)",
            xlabel="mode",
            ylabel="pearson_r",
            y_lim=(-1.0, 1.0),
        )

    report = {"task": "node_diffusion", "test": test_eval, "figures_dir": str(figs)}
    if cycle_eval is not None:
        report["test"]["cycle"] = cycle_eval
    write_json(str(run / "report.json"), report)
