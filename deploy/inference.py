"""
inference.py — Load model + run embedding inference on a graph.

No dependency on train.py. Uses standalone model.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from model import ShapeEncoder


# ─── device ──────────────────────────────────────────────────────────

def pick_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ─── load model ──────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> Tuple[ShapeEncoder, Dict[str, Any]]:
    """Load checkpoint and reconstruct ShapeEncoder."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    saved_args = ckpt.get("args", {})
    sd = ckpt["model_state_dict"]
    in_dim = int(sd["node_enc.0.weight"].shape[1])

    model = ShapeEncoder(
        in_dim=in_dim,
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        embed_dim=int(saved_args.get("embed_dim", 256)),
        heads=int(saved_args.get("heads", 4)),
        num_layers=int(saved_args.get("gat_layers", 4)),
        dropout=float(saved_args.get("dropout", 0.1)),
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ckpt


# ─── graph -> PyG Data ───────────────────────────────────────────────

def _compute_curvature(nodes: np.ndarray, adj: Dict[int, List[int]]) -> np.ndarray:
    """Menger curvature at each node."""
    n = len(nodes)
    curvature = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        neighbours = adj[idx]
        if len(neighbours) != 2:
            continue
        a, c = nodes[neighbours[0]], nodes[neighbours[1]]
        b = nodes[idx]
        ba, bc = a - b, c - b
        nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nba < 1e-12 or nbc < 1e-12:
            continue
        cos_angle = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        ac = np.linalg.norm(a - c)
        if ac < 1e-12:
            continue
        curvature[idx] = 2.0 * np.sin(angle) / ac
    return curvature


def _make_undirected(edges: List[List[int]]) -> List[List[int]]:
    seen = set()
    out = []
    for u, v in edges:
        if (u, v) not in seen:
            seen.add((u, v))
            seen.add((v, u))
            out.append([u, v])
            out.append([v, u])
    return out


def graph_to_pyg(nodes: np.ndarray, edges: List[List[int]]) -> Data:
    """
    Convert raw nodes + edges into PyG Data.
    x=[curvature, degree], pos=centred coords, bidirectional edges.
    """
    n = len(nodes)
    nodes = nodes - nodes.mean(axis=0)

    edges_bidir = _make_undirected(edges)

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges_bidir:
        adj[u].append(v)
    for k in adj:
        adj[k] = list(set(adj[k]))

    curvature = _compute_curvature(nodes, adj)
    degree = np.zeros(n, dtype=np.float32)
    for u, _ in edges_bidir:
        degree[u] += 1

    x = torch.tensor(np.column_stack([curvature, degree]), dtype=torch.float)
    pos = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges_bidir, dtype=torch.long).t().contiguous()

    return Data(x=x, pos=pos, edge_index=edge_index)


@torch.no_grad()
def embed_one(model: ShapeEncoder, data: Data, device: torch.device) -> np.ndarray:
    """Run a single graph through the encoder, return (embed_dim,) numpy."""
    data = data.to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    z = model(data)
    return z[0].cpu().numpy()
