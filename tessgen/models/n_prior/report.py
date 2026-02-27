from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .lit_module import NPriorLitModule
from ...metrics import pearson_r, rmse
from ...reporting import read_jsonl, save_histogram, save_line_plot, save_scatter_plot, write_json


@torch.no_grad()
def predict_on_loader(
    *,
    lit: NPriorLitModule,
    dl,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lit = lit.to(device)
    lit.eval()
    logn_true = []
    mu = []
    log_sigma = []
    for batch in dl:
        rd = batch["rd"].to(device)
        cond = batch["cond"].to(device)
        logn = batch["logn"].to(device).view(-1)
        mu_t, log_sigma_t = lit(rd=rd, cond=cond)
        logn_true.append(logn.detach().cpu().numpy())
        mu.append(mu_t.detach().cpu().numpy())
        log_sigma.append(log_sigma_t.detach().cpu().numpy())
    t = np.concatenate(logn_true, axis=0) if logn_true else np.zeros((0,), dtype=np.float32)
    m = np.concatenate(mu, axis=0) if mu else np.zeros((0,), dtype=np.float32)
    ls = np.concatenate(log_sigma, axis=0) if log_sigma else np.zeros((0,), dtype=np.float32)
    return t.astype(np.float64, copy=False), m.astype(np.float64, copy=False), ls.astype(np.float64, copy=False)


def make_report_and_figures(
    *,
    run_dir: str,
    history_path: str,
    logn_true: np.ndarray,
    mu: np.ndarray,
    log_sigma: np.ndarray,
) -> None:
    run = Path(run_dir)
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    hist = read_jsonl(history_path)
    epochs = [int(r["epoch"]) for r in hist]
    train = [float(r["train/nll"]) for r in hist]
    val = [float(r["val/nll"]) for r in hist]
    save_line_plot(
        out_path=str(figs / "nll.png"),
        x=epochs,
        ys={"train/nll": train, "val/nll": val},
        title="NPrior NLL",
        xlabel="epoch",
        ylabel="nll",
    )
    save_line_plot(
        out_path=str(figs / "nll_symlog.png"),
        x=epochs,
        ys={"train/nll": train, "val/nll": val},
        title="NPrior NLL (symlog y)",
        xlabel="epoch",
        ylabel="nll",
        y_scale="symlog",
    )

    if logn_true.size:
        save_scatter_plot(
            out_path=str(figs / "logn_true_vs_mu.png"),
            x=logn_true.tolist(),
            y=mu.tolist(),
            title="log(N): true vs predicted mean",
            xlabel="true logN",
            ylabel="pred mu",
        )
        save_histogram(
            out_path=str(figs / "logn_error_hist.png"),
            values=(mu - logn_true).tolist(),
            title="log(N) error (mu - true)",
            xlabel="mu - logN_true",
            bins=80,
        )
        save_histogram(
            out_path=str(figs / "sigma_hist.png"),
            values=np.exp(log_sigma).tolist(),
            title="Predicted sigma distribution",
            xlabel="sigma",
            bins=80,
        )

    n_true = np.exp(logn_true)
    n_pred = np.exp(mu)
    metrics = {
        "n": int(logn_true.size),
        "pearson_r_logn": pearson_r(logn_true, mu),
        "pearson_r_n": pearson_r(n_true, n_pred),
        "rmse_logn": rmse(logn_true, mu),
        "rmse_n": rmse(n_true, n_pred),
    }

    write_json(str(run / "report.json"), {"task": "n_prior", "metrics": metrics, "figures_dir": str(figs)})

