"""
graph_utils.py
──────────────
Utilities for building the shape-graph dictionaries and
computing per-node / per-edge geometric features.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any


# ──────────────────────────────────────────────
# Feature computation
# ──────────────────────────────────────────────

def _compute_curvature(nodes: np.ndarray,
                       edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Discrete curvature at every node.

    For interior nodes on a sequential chain the curvature is
    approximated by the Menger curvature:  κ = 2·sin(α) / |c-a|
    where α is the angle ∠(a, b, c).
    Nodes that are not interior chain nodes (e.g. junction hubs,
    endpoints) receive curvature = 0.
    """
    n = len(nodes)
    curvature = np.zeros(n, dtype=np.float64)

    # build adjacency list
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    for idx in range(n):
        neighbours = adj[idx]
        if len(neighbours) != 2:
            continue  # endpoint or junction hub → 0
        a, c = nodes[neighbours[0]], nodes[neighbours[1]]
        b = nodes[idx]
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-12 or norm_bc < 1e-12:
            continue
        cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        ac = np.linalg.norm(a - c)
        if ac < 1e-12:
            continue
        curvature[idx] = 2.0 * np.sin(angle) / ac

    return curvature


def _compute_edge_angles(nodes: np.ndarray,
                         edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    For every consecutive pair of edges that share a node, compute the
    angle (in degrees) between them.  Returns an array aligned to nodes
    (angle at node i = angle between the two edges meeting there, or 0
    if the node is an endpoint / hub).
    """
    n = len(nodes)
    angles = np.zeros(n, dtype=np.float64)

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    for idx in range(n):
        neighbours = adj[idx]
        if len(neighbours) != 2:
            continue
        a, c = nodes[neighbours[0]], nodes[neighbours[1]]
        b = nodes[idx]
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-12 or norm_bc < 1e-12:
            continue
        cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        angles[idx] = np.degrees(np.arccos(cos_angle))

    return angles


# ──────────────────────────────────────────────
# Shape-dict builder
# ──────────────────────────────────────────────

def build_shape_dict(category: str,
                     nodes: np.ndarray,
                     edges: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Wrap raw geometry into the canonical dataset record.
    """
    curvature = _compute_curvature(nodes, edges)
    segment_angle = _compute_edge_angles(nodes, edges)

    return {
        "category": category,
        "nodes": nodes,                       # (N, 3) float64
        "edges": edges,                       # list[(int, int)]
        "features": {
            "curvature": curvature,           # (N,)
            "segment_angle": segment_angle,   # (N,)  degrees
        },
    }
