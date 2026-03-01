from __future__ import annotations

import argparse

import torch

from ..data import TessellationRowDataset, collate_graph_batch
from ..edge_model import EdgeModel, EdgeModelConfig
from ..graph_utils import knn_candidate_pairs, pairs_to_edge_index
from ..models.edge_3 import Edge3Model, Edge3ModelConfig
from ..node_diffusion import DiffusionConfig, DiffusionSchedule, NodeDenoiser, NodeDenoiserConfig
from ..surrogate import SurrogateConfig, SurrogateModel
from ..utils import device_from_arg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick smoke test for model forward passes")
    p.add_argument("--data_csv", type=str, default="data/Data_2.csv")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = device_from_arg(args.device)
    ds = TessellationRowDataset(data_csv=args.data_csv, target_cols=["RS"])

    items = [ds[0], ds[5]]
    batch = collate_graph_batch(items)

    surrogate = SurrogateModel(y_dim=1, cfg=SurrogateConfig(d_h=64, n_layers=2)).to(device)
    batch_dev = type(batch)(
        x=batch.x.to(device),
        edge_index=batch.edge_index.to(device),
        batch=batch.batch.to(device),
        rd=batch.rd.to(device),
        y=batch.y.to(device),
        n_nodes=batch.n_nodes.to(device),
        n_edges=batch.n_edges.to(device),
    )
    out = surrogate(batch_dev)
    print("surrogate_out", tuple(out.shape))

    coords = items[0]["coords01"]
    cand = knn_candidate_pairs(coords.numpy(), k=12)
    edge_model = EdgeModel(cfg=EdgeModelConfig(d_h=64, n_layers=2)).to(device)
    logits = edge_model(
        coords01=coords.to(device),
        msg_edge_index=pairs_to_edge_index(cand).to(device),
        cand_pairs_uv=torch.from_numpy(cand).to(device),
    )
    print("edge_logits", tuple(logits.shape))

    edge3_model = Edge3Model(cfg=Edge3ModelConfig(d_h=64, n_layers=2, d_search=8)).to(device)
    h0 = edge3_model.node_in(coords.to(device))
    s = edge3_model.search_proj(h0)
    msg_pairs = knn_candidate_pairs(s.detach().cpu().numpy(), k=8)
    logits3 = edge3_model(
        coords01=coords.to(device),
        msg_edge_index=pairs_to_edge_index(msg_pairs).to(device),
        cand_pairs_uv=torch.from_numpy(cand).to(device),
        h0=h0,
    )
    print("edge3_logits", tuple(logits3.shape))

    schedule = DiffusionSchedule(DiffusionConfig(n_steps=10)).to(device)
    denoiser = NodeDenoiser(NodeDenoiserConfig(cond_dim=2, d_h=64, n_layers=2)).to(device)
    t = torch.tensor([3], dtype=torch.long)
    eps = torch.randn_like(coords).to(device)
    x_t = schedule.q_sample(coords.to(device), t=t.to(device), eps=eps)
    cand = knn_candidate_pairs(x_t.detach().cpu().numpy(), k=12)
    eps_pred = denoiser(
        x_t=x_t,
        t=t.to(device),
        cond=torch.tensor([0.1, 0.0], device=device),
        edge_index=pairs_to_edge_index(cand).to(device),
    )
    print("eps_pred", tuple(eps_pred.shape))


if __name__ == "__main__":
    main()
