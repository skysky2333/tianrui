from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .lit_module import SurrogateLitModule
from ...reporting import read_jsonl, save_histogram, save_line_plot, save_scatter_plot, write_json


def _r2(t: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    return float("nan") if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, cols: list[str], *, log_cols: set[str], eps: float = 1e-12) -> dict:
    per_col = {}
    log_maes = []
    log_rmses = []
    log_r2s = []
    for i, c in enumerate(cols):
        t = y_true[:, i].astype(np.float64, copy=False)
        p = y_pred[:, i].astype(np.float64, copy=False)
        mae = float(np.mean(np.abs(p - t)))
        mse = float(np.mean((p - t) ** 2))
        rmse = float(np.sqrt(mse))
        r2 = _r2(t, p)
        row: dict[str, float] = {"mae": mae, "rmse": rmse, "r2": r2}

        if c in log_cols:
            t_log = np.log(np.clip(t, eps, None))
            p_log = np.log(np.clip(p, eps, None))
            mae_log = float(np.mean(np.abs(p_log - t_log)))
            mse_log = float(np.mean((p_log - t_log) ** 2))
            rmse_log = float(np.sqrt(mse_log))
            r2_log = _r2(t_log, p_log)
            row.update({"mae_log": mae_log, "rmse_log": rmse_log, "r2_log": r2_log})
            log_maes.append(mae_log)
            log_rmses.append(rmse_log)
            log_r2s.append(r2_log)

        per_col[c] = row
    return {
        "per_col": per_col,
        "mae_mean": float(np.mean([v["mae"] for v in per_col.values()])) if per_col else float("nan"),
        "rmse_mean": float(np.mean([v["rmse"] for v in per_col.values()])) if per_col else float("nan"),
        "mae_log_mean": float(np.mean(log_maes)) if log_maes else float("nan"),
        "rmse_log_mean": float(np.mean(log_rmses)) if log_rmses else float("nan"),
        "r2_log_mean": float(np.mean(log_r2s)) if log_r2s else float("nan"),
    }


@torch.no_grad()
def predict_on_loader(
    *,
    lit: SurrogateLitModule,
    dl,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    lit = lit.to(device)
    lit.eval()
    y_true = []
    y_pred = []
    sse_z = 0.0
    n_z = 0
    for batch in dl:
        batch = batch
        batch = type(batch)(
            x=batch.x.to(device),
            edge_index=batch.edge_index.to(device),
            batch=batch.batch.to(device),
            rd=batch.rd.to(device),
            y=batch.y.to(device),
            n_nodes=batch.n_nodes.to(device),
            n_edges=batch.n_edges.to(device),
        )
        y_z = lit.targets_to_z(batch.y)
        pred_z = lit.model(
            type(batch)(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                rd=batch.rd,
                y=y_z,
                n_nodes=batch.n_nodes,
                n_edges=batch.n_edges,
            )
        )
        err = pred_z - y_z
        sse_z += float(torch.sum(err**2).detach().cpu())
        n_z += int(err.numel())

        pred = lit.z_to_targets(pred_z).detach().cpu().numpy()
        y_true.append(batch.y.detach().cpu().numpy())
        y_pred.append(pred)
    y_true_np = np.concatenate(y_true, axis=0) if y_true else np.zeros((0, len(lit.target_cols)), dtype=np.float32)
    y_pred_np = np.concatenate(y_pred, axis=0) if y_pred else np.zeros((0, len(lit.target_cols)), dtype=np.float32)
    mse_z = float("nan") if n_z == 0 else float(sse_z / float(n_z))
    return y_true_np, y_pred_np, mse_z


def make_report_and_figures(
    *,
    run_dir: str,
    history_path: str,
    target_cols: list[str],
    log_cols: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_mse_z: float,
) -> None:
    run = Path(run_dir)
    figs = run / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Loss curves
    hist = read_jsonl(history_path)
    epochs = [int(r["epoch"]) for r in hist]
    train = [float(r["train/mse_z"]) for r in hist]
    val = [float(r["val/mse_z"]) for r in hist]
    save_line_plot(
        out_path=str(figs / "loss_mse_z.png"),
        x=epochs,
        ys={"train/mse_z": train, "val/mse_z": val},
        title="Surrogate loss (z-space MSE)",
        xlabel="epoch",
        ylabel="MSE",
    )
    save_line_plot(
        out_path=str(figs / "loss_mse_z_logy.png"),
        x=epochs,
        ys={"train/mse_z": train, "val/mse_z": val},
        title="Surrogate loss (z-space MSE, log y)",
        xlabel="epoch",
        ylabel="MSE",
        y_scale="log",
    )

    # Scatter + error histograms
    log_cols_set = set(log_cols)
    for i, c in enumerate(target_cols):
        t = y_true[:, i].tolist()
        p = y_pred[:, i].tolist()
        if len(t) == 0:
            continue
        save_scatter_plot(
            out_path=str(figs / f"scatter_true_vs_pred_{c}.png"),
            x=t,
            y=p,
            title=f"True vs Pred: {c}",
            xlabel=f"true {c}",
            ylabel=f"pred {c}",
        )
        err = (y_pred[:, i] - y_true[:, i]).tolist()
        save_histogram(
            out_path=str(figs / f"error_hist_{c}.png"),
            values=err,
            title=f"Error histogram: {c}",
            xlabel=f"pred - true ({c})",
        )
        if c in log_cols_set:
            t_log = np.log(np.clip(y_true[:, i], 1e-12, None)).tolist()
            p_log = np.log(np.clip(y_pred[:, i], 1e-12, None)).tolist()
            save_scatter_plot(
                out_path=str(figs / f"scatter_log_true_vs_pred_{c}.png"),
                x=t_log,
                y=p_log,
                title=f"Log True vs Pred: {c}",
                xlabel=f"log(true {c})",
                ylabel=f"log(pred {c})",
            )
            err_log = (np.log(np.clip(y_pred[:, i], 1e-12, None)) - np.log(np.clip(y_true[:, i], 1e-12, None))).tolist()
            save_histogram(
                out_path=str(figs / f"error_hist_log_{c}.png"),
                values=err_log,
                title=f"Log-error histogram: {c}",
                xlabel=f"log(pred) - log(true) ({c})",
            )

    report = {
        "task": "surrogate",
        "test": {"mse_z": float(test_mse_z), "regression": regression_report(y_true, y_pred, target_cols, log_cols=log_cols_set)},
        "figures_dir": str(figs),
    }
    write_json(str(run / "report.json"), report)
