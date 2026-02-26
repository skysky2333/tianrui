from __future__ import annotations

from torch.utils.data import Dataset

from ...data import GraphStore


class EdgeGraphDataset(Dataset):
    def __init__(self, *, tess_root: str, graph_ids: list[int]):
        self.tess_root = tess_root
        self.graph_ids = list(map(int, graph_ids))
        self.store = GraphStore(tess_root)

    def __len__(self) -> int:
        return len(self.graph_ids)

    def __getitem__(self, i: int) -> dict:
        gid = int(self.graph_ids[i])
        g = self.store.get(gid)
        return {
            "graph_id": gid,
            "coords01": g.coords01,
            "edges_undirected": g.edges_undirected,
            "n_nodes": g.n_nodes,
            "n_edges": g.n_edges,
        }

