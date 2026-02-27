from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .lit_module import NodeDiffusionLitModule
from ...graph_utils import knn_candidate_pairs, pairs_to_edge_index
from ...reporting import read_jsonl, save_histogram, save_line_plot, save_scatter_plot, write_json


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
    nlls = []
    logn_err = []
    n_abs_err = []
    n_true_all = []
    n_pred_all = []

    for i, sample in enumerate(dl):
        if max_samples and i >= int(max_samples):
            break
        coords0_cpu = sample["coords01"]
        N = int(coords0_cpu.shape[0])
        cond_z = lit._cond_to_z(sample)  # noqa: SLF001

        logN = float(np.log(max(1.0, float(N))))
        mu, _ = lit.n_pred(cond_z)
        mu_f = float(mu.detach().cpu().item())
        n_hat = int(round(float(np.exp(mu_f))))

        # Compute diffusion loss + NLL on a random timestep, matching training objective
        logN_t = torch.log(torch.tensor(float(N), device=lit.device))
        mu_t, log_sigma_t = lit.n_pred(cond_z)
        sigma2 = torch.exp(2.0 * log_sigma_t)
        nll = 0.5 * (logN_t - mu_t) ** 2 / sigma2 + log_sigma_t

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

        loss = diff_loss + lit.lambda_n * nll.mean()
        losses.append(float(loss.detach().cpu()))
        diffs.append(float(diff_loss.detach().cpu()))
        nlls.append(float(nll.mean().detach().cpu()))

        logn_err.append(mu_f - logN)
        n_abs_err.append(float(abs(n_hat - N)))
        n_true_all.append(float(N))
        n_pred_all.append(float(n_hat))

    return {
        "samples": int(min(len(dl), int(max_samples)) if max_samples else len(dl)),
        "loss_mean": _safe_mean(losses),
        "diff_mse_mean": _safe_mean(diffs),
        "nll_mean": _safe_mean(nlls),
        "n_pred": {
            "n_mae": _safe_mean(n_abs_err),
            "logn_err_mean": float(np.mean(logn_err)) if logn_err else float("nan"),
            "logn_err_std": float(np.std(logn_err)) if logn_err else float("nan"),
            "n_true": n_true_all,
            "n_pred": n_pred_all,
            "logn_err": logn_err,
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

    n_true = test_eval["n_pred"].pop("n_true")
    n_pred = test_eval["n_pred"].pop("n_pred")
    logn_err = test_eval["n_pred"].pop("logn_err")

    if n_true:
        save_scatter_plot(
            out_path=str(figs / "n_true_vs_pred.png"),
            x=[float(x) for x in n_true],
            y=[float(y) for y in n_pred],
            title="Node count: true vs predicted",
            xlabel="true N",
            ylabel="pred N",
        )
    if logn_err:
        save_histogram(
            out_path=str(figs / "logn_error_hist.png"),
            values=[float(x) for x in logn_err],
            title="log(N) prediction error",
            xlabel="mu - log(N)",
            bins=60,
        )

    report = {"task": "node_diffusion", "test": test_eval, "figures_dir": str(figs)}
    if cycle_eval is not None:
        report["test"]["cycle"] = cycle_eval
    write_json(str(run / "report.json"), report)
