#!/usr/bin/env python3
"""
infer.py
────────
Run embedding inference with a trained ShapeEncoder.

Loads a checkpoint, builds a PyG Data graph from a JSON file,
and outputs the 256-dim L2-normalised embedding.

Input JSON format:
    {
      "nodes": [[x, y, z], ...],
      "edges": [[i, j], ...]
    }

Usage
─────
    python infer.py --input my_graph.json
    python infer.py --input my_graph.json --ckpt processed/encoder.pt --device cuda:0
    python infer.py --input my_graph.json --out embedding.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# import model class from train.py (single source of truth)
from train import ShapeEncoder


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedding inference with trained ShapeEncoder")
    p.add_argument("--input", type=str, required=True,
                   help="Path to input JSON graph")
    p.add_argument("--ckpt", type=str, default="processed/encoder.pt",
                   help="Path to saved checkpoint")
    p.add_argument("--device", type=str, default="auto",
                   help="cpu | cuda:0 | mps | auto")
    p.add_argument("--out", type=str, default="",
                   help="Optional path to save embedding as .npy")
    return p.parse_args()


# ─── device ──────────────────────────────────────────────────────────

def pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ─── load model from checkpoint ──────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> Tuple[ShapeEncoder, Dict[str, Any]]:
    """
    Load checkpoint and reconstruct ShapeEncoder from saved args.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    saved_args = ckpt.get("args", {})

    # reconstruct model from saved hyperparams
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

    print(f"[model] loaded from {ckpt_path}")
    print(f"[model] in_dim={in_dim}  embed_dim={saved_args.get('embed_dim', 256)}")
    if "best_epoch" in ckpt:
        print(f"[model] best_epoch={ckpt['best_epoch']}  best_score={ckpt['best_score']:.4f}")

    return model, ckpt


# ─── read input JSON ─────────────────────────────────────────────────

def read_graph_json(path: str) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Read a graph JSON with 'nodes' and 'edges'.
    Returns raw nodes (N,3) and edge list.
    """
    obj = json.loads(Path(path).read_text())
    nodes = np.asarray(obj["nodes"], dtype=np.float32)
    edges = [[int(i), int(j)] for i, j in obj["edges"]]
    return nodes, edges


# ─── build PyG Data ──────────────────────────────────────────────────

def _compute_curvature(nodes: np.ndarray, adj: Dict[int, List[int]]) -> np.ndarray:
    """Menger curvature at each node (same logic as graph_utils.py)."""
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


def _compute_degree(edges_bidir: List[List[int]], n: int) -> np.ndarray:
    """Node degree from bidirectional edge list."""
    deg = np.zeros(n, dtype=np.float32)
    for u, _ in edges_bidir:
        deg[u] += 1
    return deg


def _make_undirected(edges: List[List[int]]) -> List[List[int]]:
    """Ensure every (u,v) has a matching (v,u). Deduplicate."""
    seen = set()
    out = []
    for e in edges:
        u, v = e[0], e[1]
        if (u, v) not in seen:
            seen.add((u, v))
            seen.add((v, u))
            out.append([u, v])
            out.append([v, u])
    return out


def graph_to_pyg(nodes: np.ndarray, edges: List[List[int]]) -> Data:
    """
    Convert raw nodes + edges into a PyG Data object matching
    the training format: x=[curvature, degree], pos=centred coords,
    bidirectional edge_index.
    """
    n = len(nodes)

    # centre nodes
    nodes = nodes - nodes.mean(axis=0)

    # make edges bidirectional
    edges_bidir = _make_undirected(edges)

    # build adjacency for curvature
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in edges_bidir:
        adj[u].append(v)
    # deduplicate adjacency
    for k in adj:
        adj[k] = list(set(adj[k]))

    # node features: [curvature, degree]
    curvature = _compute_curvature(nodes, adj)
    degree = _compute_degree(edges_bidir, n)

    x = torch.tensor(np.column_stack([curvature, degree]), dtype=torch.float)
    pos = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges_bidir, dtype=torch.long).t().contiguous()

    return Data(x=x, pos=pos, edge_index=edge_index)


# ─── inference ───────────────────────────────────────────────────────

@torch.no_grad()
def embed_one(model: ShapeEncoder, data: Data, device: torch.device) -> np.ndarray:
    """Run a single graph through the encoder, return (embed_dim,) numpy array."""
    data = data.to(device)

    # model expects data.batch for pooling
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    z = model(data)  # (1, embed_dim)
    return z[0].cpu().numpy()


# ─── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[cfg] device={device}")

    model, _ = load_model(args.ckpt, device)

    nodes, edges = read_graph_json(args.input)
    print(f"[input] {args.input}: {len(nodes)} nodes, {len(edges)} edges")

    data = graph_to_pyg(nodes, edges)
    print(f"[pyg] x={data.x.shape}  edge_index={data.edge_index.shape}")

    emb = embed_one(model, data, device)
    print(f"[embed] shape={emb.shape}  norm={np.linalg.norm(emb):.4f}")
    print(f"[embed] first 10: {emb[:10]}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), emb)
        print(f"[io] saved -> {out_path}")


if __name__ == "__main__":
    main()
