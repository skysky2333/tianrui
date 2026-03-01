from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .core import Edge3ModelConfig
from .lit_module import Edge3LitModule
from ..edge.dataset import EdgeGraphDataset
from ...data import collate_first, discover_graph_ids, train_val_split_graph_ids
from ...outdirs import finalize_out_dir, make_timestamped_run_dir
from ...pl_utils import lightning_device_from_arg
from ...reporting import save_line_plot, write_json
from ...utils import set_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for edge_3 model (Lightning)")
    p.add_argument("--tess_root", type=str, default="data/Tessellation_Dataset")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument(
        "--cand_mode",
        type=str,
        nargs="+",
        default=["knn", "delaunay"],
        help="Candidate set construction modes to tune over (one or more values): knn, delaunay",
    )
    p.add_argument("--k", type=int, nargs="+", default=[24, 32, 39, 48], help="Candidate k values (used when cand_mode=knn)")
    p.add_argument("--k_msg", type=int, nargs="+", default=[8, 12, 16, 24], help="Message-graph k values to tune over")
    p.add_argument("--d_search", type=int, nargs="+", default=[8, 16, 32], help="Embedding dim used for learned-kNN message graph")
    p.add_argument(
        "--neg_ratio",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0],
        help="Candidate negative subsampling ratios to tune over (one or more floats)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--limit_train_batches", type=float, default=0.2)
    p.add_argument("--limit_val_batches", type=float, default=1.0)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout_sec", type=int, default=0, help="0 = no timeout")
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir, run_dir, run_name = make_timestamped_run_dir(args.out_dir)
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    graph_ids = discover_graph_ids(args.tess_root)
    train_g, val_g = train_val_split_graph_ids(graph_ids, val_frac=float(args.val_frac), seed=int(args.seed))

    ds_train = EdgeGraphDataset(tess_root=args.tess_root, graph_ids=train_g)
    ds_val = EdgeGraphDataset(tess_root=args.tess_root, graph_ids=val_g)
    dev = lightning_device_from_arg(args.device)
    num_workers = int(args.num_workers)
    dl_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0, "pin_memory": dev.accelerator == "gpu"}
    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_first, **dl_kwargs)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate_first, **dl_kwargs)

    cand_mode_choices = [str(x) for x in list(args.cand_mode)]
    if not cand_mode_choices:
        raise SystemExit("--cand_mode must provide at least one value")
    for m in cand_mode_choices:
        if m not in ("knn", "delaunay"):
            raise SystemExit(f"Invalid cand_mode {m!r}; expected knn or delaunay")

    k_choices = [int(x) for x in list(args.k)]
    if not k_choices:
        raise SystemExit("--k must provide at least one value")
    k_msg_choices = [int(x) for x in list(args.k_msg)]
    if not k_msg_choices:
        raise SystemExit("--k_msg must provide at least one value")
    d_search_choices = [int(x) for x in list(args.d_search)]
    if not d_search_choices:
        raise SystemExit("--d_search must provide at least one value")
    neg_ratio_choices = [float(x) for x in list(args.neg_ratio)]
    if not neg_ratio_choices:
        raise SystemExit("--neg_ratio must provide at least one value")

    def objective(trial: optuna.Trial) -> float:
        cand_mode = str(trial.suggest_categorical("cand_mode", cand_mode_choices))
        if cand_mode == "knn":
            k = int(trial.suggest_categorical("k", k_choices))
        else:
            k = 0
        k_msg = int(trial.suggest_categorical("k_msg", k_msg_choices))
        d_search = int(trial.suggest_categorical("d_search", d_search_choices))
        neg_ratio = float(trial.suggest_categorical("neg_ratio", neg_ratio_choices))

        cfg = Edge3ModelConfig(
            d_h=trial.suggest_categorical("d_h", [64, 96, 128, 160, 192, 256]),
            n_layers=trial.suggest_int("n_layers", 2, 5),
            n_rbf=trial.suggest_categorical("n_rbf", [8, 16, 24]),
            d_search=d_search,
            dropout=trial.suggest_float("dropout", 0.0, 0.2),
        )
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)

        lit = Edge3LitModule(
            cfg=asdict(cfg),
            k=k,
            cand_mode=cand_mode,
            k_msg=k_msg,
            neg_ratio=neg_ratio,
            lr=float(lr),
            weight_decay=float(wd),
        )
        trainer = pl.Trainer(
            max_epochs=int(args.max_epochs),
            accelerator=dev.accelerator,
            devices=dev.devices,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=float(args.limit_train_batches),
            limit_val_batches=float(args.limit_val_batches),
            num_sanity_val_steps=0,
        )
        trainer.fit(lit, train_dataloaders=dl_train, val_dataloaders=dl_val)
        val = trainer.callback_metrics["val/bce"]
        val_f = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)
        del trainer
        del lit
        if dev.accelerator == "gpu":
            torch.cuda.empty_cache()
        if dev.accelerator == "mps":
            torch.mps.empty_cache()
        return val_f

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(args.n_trials), timeout=None if args.timeout_sec == 0 else int(args.timeout_sec))

    df_trials = study.trials_dataframe()
    df_trials.to_csv(str(run_dir / "trials.csv"), index=False)
    write_json(str(run_dir / "best.json"), {"best_value": float(study.best_value), "best_params": study.best_params})

    values = [t.value for t in study.trials if t.value is not None]
    xs = list(range(1, len(values) + 1))
    save_line_plot(
        out_path=str(run_dir / "optuna_history.png"),
        x=xs,
        ys={"val/bce": [float(v) for v in values]},
        title="Optuna tuning history (edge_3 model)",
        xlabel="trial",
        ylabel="val BCE",
    )

    cfg_out = {
        "task": "edge_3_tune",
        "tess_root": args.tess_root,
        "val_frac": float(args.val_frac),
        "device": {"accelerator": dev.accelerator, "devices": dev.devices},
        "search_space": {
            "cand_mode": cand_mode_choices,
            "k": k_choices,
            "k_msg": k_msg_choices,
            "d_search": d_search_choices,
            "neg_ratio": neg_ratio_choices,
        },
        "n_trials": int(args.n_trials),
        "timeout_sec": int(args.timeout_sec),
        "max_epochs": int(args.max_epochs),
        "limit_train_batches": float(args.limit_train_batches),
        "limit_val_batches": float(args.limit_val_batches),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "base_out_dir": str(base_dir),
        "run_dir": str(run_dir),
        "run_name": str(run_name),
    }
    write_json(str(run_dir / "config.json"), cfg_out)

    finalize_out_dir(base_dir=base_dir, run_dir=run_dir, run_name=run_name, argv=sys.argv)
    print(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "run_dir": str(run_dir),
                "base_out_dir": str(base_dir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

