# Tessellation inverse design (metrics → graph)

This repo contains a small deep-learning framework to **generate tessellation-style geometric graphs** (nodes + connections) conditioned on:

- `RD` (either user-specified, or inferred by searching over a candidate list), and
- target simulation metrics `y` (at minimum `RS`, optionally more columns from `data/Data_1.csv` or `data/Data_2.csv`).

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

`tessgen.cli.generate` also writes `--out_dir/figures/graph_gen_<i>.png` and `graph_gen_<i>.svg` for each saved sample.

## Quickstart

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

Outputs in `runs/surrogate/`:
- `surrogate.pt` (inference artifact used by `tessgen.cli.generate`)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curves + regression plots; includes `loss_mse_z_logy.png`)

3) Train an edge model (coords → edges):

```bash
conda run -n tianrui python -m tessgen.cli.train_edge \
  --epochs 8 \
  --out_dir runs/edge \
  --device cpu
```

Outputs in `runs/edge/`:
- `edge_model.pt` (inference artifact)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curve, PR/ROC curves, probability histograms; includes `loss_bce_logy.png`)

4) Train a node diffusion model (condition → coords):

```bash
conda run -n tianrui python -m tessgen.cli.train_node_diffusion \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --epochs 8 \
  --out_dir runs/node_diffusion \
  --device cpu
```

Outputs in `runs/node_diffusion/`:
- `node_diffusion.pt` (inference artifact)
- `best.ckpt` / `last.ckpt` (Lightning checkpoints)
- `config.json`, `history.jsonl`, `report.json`
- `figures/` (loss curves + node-count prediction plots; includes `loss_symlog.png` and `diff_mse_logy.png`)

5) Generate K candidate graphs for a target `(RD, RS)`:

```bash
conda run -n tianrui python -m tessgen.cli.generate \
  --rd 0.10 \
  --cond RS=0.01 \
  --k 8 \
  --surrogate_ckpt runs/surrogate/surrogate.pt \
  --node_ckpt runs/node_diffusion/node_diffusion.pt \
  --edge_ckpt runs/edge/edge_model.pt \
  --out_dir out/generated
```

Notes:
- The generator is **stochastic**: use `--k` to sample multiple candidates and select by surrogate score.
- Conditioning on **more than RS** will generally produce tighter, more identifiable generations.

6) (Optional) Infer `RD` given only metrics (e.g. only `RS`):

If you omit `--rd`, the generator will search over `--rd_candidates` (defaults to `0.01 0.05 0.1 0.15 0.2`) and pick
the best sample by surrogate score. When searching, `--k` is interpreted as **samples per RD** (total samples =
`len(rd_candidates) * k`). The chosen `RD` is written to `meta_*.json` and printed as `best_rd`.

```bash
conda run -n tianrui python -m tessgen.cli.generate \
  --cond RS=0.01 \
  --rd_candidates 0.01 0.05 0.1 0.15 0.2 \
  --k 2 \
  --surrogate_ckpt runs/surrogate/surrogate.pt \
  --node_ckpt runs/node_diffusion/node_diffusion.pt \
  --edge_ckpt runs/edge/edge_model.pt \
  --out_dir out/generated_infer_rd
```

## Hyperparameter tuning (Optuna)

Each model has an optional Optuna tuner that searches hyperparameters against the validation split.

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
  --out_dir runs/tune_edge
```

Node diffusion:

```bash
conda run -n tianrui python -m tessgen.cli.tune_node_diffusion \
  --data_csv data/Data_2.csv \
  --cond_cols RS \
  --out_dir runs/tune_node_diffusion
```

Each tuning run writes `trials.csv`, `best.json`, `optuna_history.png`, and `config.json`.

## Code layout

- `tessgen/models/surrogate/`: surrogate model + Lightning trainer + Optuna tuner + reports
- `tessgen/models/edge/`: edge model + Lightning trainer + Optuna tuner + reports
- `tessgen/models/node_diffusion/`: node diffusion model + Lightning trainer + Optuna tuner + reports

## Acknowledgement

This repo is created with assistance by LLMs from OpenAI and/or Antropic.
