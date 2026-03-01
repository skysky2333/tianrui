# Tessellation inverse design (metrics → graph)

This repo contains a small deep-learning framework to **generate tessellation-style geometric graphs** (nodes + connections) conditioned on:

- `RD` (either user-specified, or inferred by searching over a candidate list), and
- target simulation metrics `y` (at minimum `RS`, optionally more columns from `data/Data_1.csv` or `data/Data_2.csv`).

The node diffusion model is **explicitly conditioned on node count** via `logN`, so generation can control `N` directly.
If the user does not provide `N`, a separate **NPrior** model can propose a broad distribution of plausible `N` values
conditioned on `(RD, metrics)`.

The dataset lives in:

- `data/Tessellation_Dataset/Node_<n>.txt`
- `data/Tessellation_Dataset/Connection_<n>.txt`
- `data/Data_1.csv`, `data/Data_2.csv` (5 rows per `<n>` with `RD ∈ {0.01, 0.05, 0.1, 0.15, 0.2}`)

## Environment

The code expects a conda env with PyTorch + common scientific Python packages.

Training and Optuna tuning default to `--num_workers 4` for faster input loading.

Dependencies used by the training framework:
- PyTorch Lightning (`pytorch_lightning`)
- Optuna (`optuna`) for optional hyperparameter tuning
- Matplotlib (`matplotlib`) for report figures

## Visualization

Visualize an existing `Node_*.txt` + `Connection_*.txt` pair:

```bash
conda run -n tianrui python -m tessgen.cli.visualize_graph \
  --node_path data/Tessellation_Dataset/Node_1.txt \
  --conn_path data/Tessellation_Dataset/Connection_1.txt \
  --out_png out/graph_1.png \
  --out_svg out/graph_1.svg
```

`tessgen.cli.generate` writes `graph_gen_<i>.png` and `graph_gen_<i>.svg` under `--out_dir/<timestamp>/figures/` for each saved sample.

## Quickstart

## Output directories

All commands in this repo that accept `--out_dir` treat it as a **base directory**:

- Full outputs for each run are written under `--out_dir/<timestamp>/`
- The most recent run is recorded in `--out_dir/latest_run.json`
- Key top-level files from the latest run (e.g. `*.pt`, `best.ckpt`, `config.json`, `trials.csv`) are copied into `--out_dir/`
  so stable paths like `runs/surrogate/surrogate.pt` keep working.

1) Sanity-check parsing:

```bash
conda run -n tianrui python -m tessgen.cli.check_data --data_csv data/Data_2.csv
```

2) Train a surrogate (graph + RD → metrics):

```bash
conda run -n tianrui python -m tessgen.cli.train_surrogate \
  --data_csv data/Data_2.csv \
  --target_cols RS \
  --epochs 32 \
  --batch_size 64 \
  --out_dir runs/surrogate \
  --device cpu
```

Outputs in `runs/surrogate/<timestamp>/` (full run):
- `surrogate.pt` (inference artifact used by `tessgen.cli.generate`)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curves + regression plots; includes `loss_mse_z_logy.png`)

Latest top-level artifacts are also copied into `runs/surrogate/` (see `runs/surrogate/latest_run.json`).

3) Train an edge model (coords → edges):

```bash
conda run -n tianrui python -m tessgen.cli.train_edge \
  --epochs 32 \
  --out_dir runs/edge \
  --device cpu \
  --cand_mode delaunay \
  --cycle_surrogate_ckpt runs/surrogate/surrogate.pt
```

tessgen supports different **candidate edge sets** via `--cand_mode`:
- `knn` (default): local kNN candidates (uses `--k`)
- `delaunay`: 2D Delaunay triangulation candidates (`--k` ignored; can miss true edges on some graphs)

tuning candidates (to find a `k` that covers all true edges in your dataset):
```bash
conda run -n tianrui python -m tessgen.cli.analyze_edge_candidates \
  --tess_root data/Tessellation_Dataset \
  --out_dir out/candidate_analysis
```


Outputs in `runs/edge/<timestamp>/` (full run):
- `edge_model.pt` (inference artifact)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curve, PR/ROC curves, probability histograms; includes `loss_bce_logy.png`)
- `preview_val/epoch_###/` (optional per-epoch qualitative previews; disable with `--no-preview_each_epoch`)

Latest top-level artifacts are also copied into `runs/edge/` (see `runs/edge/latest_run.json`).

3b) Train an `edge_3` model (IDGL-lite; learned kNN message graph):

This variant keeps the same candidate-set idea (`--cand_mode` + `--k`), but uses a **learned embedding-space kNN**
(`--k_msg`) for message passing.

```bash
conda run -n tianrui python -m tessgen.cli.train_edge_3 \
  --epochs 8 \
  --out_dir runs/edge_3 \
  --device cpu \
  --cand_mode delaunay

```


4) Train an N prior (RD + metrics → log(N)):

```bash
conda run -n tianrui python -m tessgen.cli.train_n_prior \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --epochs 16 \
  --out_dir runs/n_prior \
  --device cpu
```

Outputs in `runs/n_prior/<timestamp>/` (full run):
- `n_prior.pt` (inference artifact used by `tessgen.cli.generate` when `--n_nodes` is omitted)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (NLL curves + log(N) scatter/hist)

Latest top-level artifacts are also copied into `runs/n_prior/` (see `runs/n_prior/latest_run.json`).

5) Train a node diffusion model (RD + logN + metrics → coords):

```bash
conda run -n tianrui python -m tessgen.cli.train_node_diffusion \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --epochs 16 \
  --out_dir runs/node_diffusion \
  --device cpu
```

Optional: run an end-to-end cycle benchmark (metrics → graph → surrogate → metrics) on the diffusion **test split** at the
end of training. By default, when cycle checkpoints are provided, an additional **validation cycle eval** also runs at the
end of **every epoch** and `best.ckpt` is selected by `val/cycle_r_best`:

```bash
conda run -n tianrui python -m tessgen.cli.train_node_diffusion \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --epochs 32 \
  --out_dir runs/node_diffusion \
  --device cpu \
  --cycle_surrogate_ckpt runs/surrogate/surrogate.pt \
  --cycle_edge_ckpt runs/edge/edge_model.pt \
  --cycle_k_best 8 \
  --cycle_edge_thr 0.5 \
  --cycle_epoch_rows 10 \
  --report_max_samples 20 \
  --cycle_k_best 16
```

To disable the per-epoch validation cycle eval (and monitor `val/loss` instead), add `--no-cycle_each_epoch`.

Outputs in `runs/node_diffusion/<timestamp>/` (full run):
- `node_diffusion.pt` (inference artifact)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curves; includes `loss_symlog.png` and `diff_mse_logy.png`)

Latest top-level artifacts are also copied into `runs/node_diffusion/` (see `runs/node_diffusion/latest_run.json`).

If cycle eval is enabled, additional outputs are written in `runs/node_diffusion/<timestamp>/cycle/` and the diffusion `report.json`
includes `test.cycle.metrics.*` (Pearson/Spearman, MAE/RMSE, R²) for both single-sample and best-of-k.
The diffusion run also includes `figures/cycle_pearson_r.(png|svg)`.

If per-epoch cycle eval is enabled (default when cycle ckpts are provided), outputs are written under:
- `runs/node_diffusion/<timestamp>/cycle_val/epoch_###/`
and `figures/cycle_r_over_epoch.png` is generated from `history.jsonl`.

6) Generate candidate graphs for a target `(RD, RS)`:

```bash
conda run -n tianrui python -m tessgen.cli.generate \
  --rd 0.10 \
  --cond RS=0.01 \
  --n_nodes 1024 \
  --k 4 \
  --edge_thr 0.5 \
  --surrogate_ckpt runs/surrogate/surrogate.pt \
  --node_ckpt runs/node_diffusion/node_diffusion.pt \
  --edge_ckpt runs/edge/edge_model.pt \
  --out_dir out/generated \
  --device cpu
```

Notes:
- The generator is **stochastic**: use `--k` to sample multiple candidates and select by surrogate score.
- `--edge_thr` controls graph sparsity by filtering low-confidence edges before applying `--deg_cap` (a hard maximum degree).
- Conditioning on **more than RS** will generally produce tighter, more identifiable generations.
- If you omit `--n_nodes`, provide either `--n_candidates` (grid search) or `--n_prior_ckpt` (sample candidate N values).

7) (Optional) Infer `RD` given only metrics (e.g. only `RS`):

If you omit `--rd`, the generator will search over `--rd_candidates` (defaults to `0.01 0.05 0.1 0.15 0.2`) and pick
the best sample by surrogate score. When searching, `--k` is interpreted as **samples per (RD, N) combination**.
The chosen `RD` is written to `meta_*.json` and printed as `best_rd`.

```bash
conda run -n tianrui python -m tessgen.cli.generate \
  --cond RS=0.01 \
  --rd_candidates 0.01 0.05 0.1 0.15 0.2 \
  --n_prior_ckpt runs/n_prior/n_prior.pt \
  --n_prior_samples 12 \
  --k 2 \
  --edge_thr 0.5 \
  --surrogate_ckpt runs/surrogate/surrogate.pt \
  --node_ckpt runs/node_diffusion/node_diffusion.pt \
  --edge_ckpt runs/edge/edge_model.pt \
  --out_dir out/generated_infer_rd \
  --device cpu

```

8) End-to-end benchmark on the test split (metrics → graph → surrogate → metrics):

This evaluates the full pipeline by taking test-set metrics (e.g. `RS`) + true `RD`, generating a graph with the
node diffusion + edge model, then predicting metrics with the surrogate and benchmarking correlation/error vs the
original test metrics.

```bash
conda run -n tianrui python -m tessgen.cli.benchmark_cycle \
  --data_csv data/Data_2.csv \
  --surrogate_ckpt runs/surrogate/surrogate.pt \
  --node_ckpt runs/node_diffusion/node_diffusion.pt \
  --edge_ckpt runs/edge/edge_model.pt \
  --edge_thr 0.5 \
  --out_dir out/bench_cycle \
  --device cpu
```

Outputs in `out/bench_cycle/<timestamp>/` (full run):
- `report.json` (Pearson/Spearman, MAE/RMSE, R² for both single-sample and best-of-k)
- `rows.jsonl` (per-row predictions/errors)
- `figures/` (summary scatter + error hist, PNG+SVG)
- `graphs_true/`, `graphs_single/`, `graphs_best/` (per-row graph visualizations, PNG+SVG; titles include RS_true/RS_pred)


## Hyperparameter tuning (Optuna)

Each model has an optional Optuna tuner that searches hyperparameters against the validation split.
Tuners default to `--max_epochs 1` per trial for speed; increase it for more reliable ranking.

Notes on default search spaces:
- `tune_edge` tunes `k` and `neg_ratio` (pass a single value to fix either)
- `tune_node_diffusion` tunes `k_nn`, `steps`, `beta_start`, and `beta_end` (pass a single value to fix any of them)
- `tune_n_prior` tunes `sigma_min` (pass a single value to fix it)

Surrogate:

```bash
conda run -n tianrui python -m tessgen.cli.tune_surrogate \
  --data_csv data/Data_2.csv \
  --target_cols RS \
  --out_dir runs/tune_surrogate
```

Edge model:

```bash
conda run -n tianrui python -m tessgen.cli.tune_edge \
  --out_dir runs/tune_edge \
  --device cpu \
  --n_trials 50
```

```bash
conda run -n tianrui python -m tessgen.cli.tune_edge_3 \
  --n_trials 50 \
  --max_epochs 1 \
  --out_dir runs/tune_edge_3 \
  --device cpu
```

Node diffusion:

```bash
conda run -n tianrui python -m tessgen.cli.tune_node_diffusion \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --out_dir runs/tune_node_diffusion
```

N prior:

```bash
conda run -n tianrui python -m tessgen.cli.tune_n_prior \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --out_dir runs/tune_n_prior
```

Each tuning run writes `trials.csv`, `best.json`, `optuna_history.png`, and `config.json`.
Each tuning run writes its full outputs under `--out_dir/<timestamp>/` and copies key files into `--out_dir/` for convenience.

## Code layout

- `tessgen/models/surrogate/`: surrogate model + Lightning trainer + Optuna tuner + reports
- `tessgen/models/edge/`: edge model + Lightning trainer + Optuna tuner + reports
- `tessgen/models/node_diffusion/`: node diffusion model + Lightning trainer + Optuna tuner + reports
- `tessgen/models/n_prior/`: N prior model + Lightning trainer + Optuna tuner + reports

## Acknowledgement

This repo is created with assistance by LLMs from OpenAI and/or Antropic.
