#!/usr/bin/env python3
"""
FastAPI server for centerline segmentation + embedding inference.

Pipeline:
    1. Upload graph (nodes + edges)
    2. Split into segments (junction / straight / arc)
    3. Downsample each segment to ~16 nodes (uniform arc-length)
    4. (Optional) Embed each segment with trained ShapeEncoder
    5. Return typed segments with angle, curvature, radius, embeddings

Start:
    uvicorn app:app --host 0.0.0.0 --port 8000
    # or
    python app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from centerline_segmenter import (
    split_centerline_graph,
    downsample_segment_for_model,
    estimate_arc_angle,
    SegmentParams,
)

# ── optional: ML embedding (gracefully skip if torch not installed) ──
_MODEL = None
_DEVICE = None

def _load_ml_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return

    ckpt_path = os.environ.get("MODEL_CHECKPOINT", "weights/encoder.pt")
    if not Path(ckpt_path).exists():
        print(f"[model] checkpoint not found at {ckpt_path}, embedding disabled")
        return

    try:
        from inference import load_model, pick_device
        _DEVICE = pick_device(os.environ.get("DEVICE", "auto"))
        _MODEL, _ = load_model(ckpt_path, _DEVICE)
        print(f"[model] loaded from {ckpt_path} on {_DEVICE}")
    except ImportError:
        print("[model] torch/torch_geometric not installed, embedding disabled")
    except Exception as e:
        print(f"[model] failed to load: {e}")


def _embed_segment(ds_pts: np.ndarray, ds_edges: List[List[int]]) -> Optional[List[float]]:
    """Embed a downsampled segment. Returns None if model not available."""
    if _MODEL is None or _DEVICE is None:
        return None
    if len(ds_pts) < 3:
        return None

    try:
        from inference import graph_to_pyg, embed_one
        data = graph_to_pyg(ds_pts.astype(np.float32), ds_edges)
        emb = embed_one(_MODEL, data, _DEVICE)
        return emb.tolist()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Centerline Segmentation API",
    description="Detect junctions, straights, arcs in 3D centerline graphs. Optionally embed with trained GATv2 model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _load_ml_model()


# ══════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════

class GraphInput(BaseModel):
    nodes: List[List[float]] = Field(..., description="Nx3 coordinates")
    edges: List[List[int]] = Field(..., description="Edge pairs [i, j]")
    target_step: float = Field(1.0, description="Resampling step for curvature")
    downsample_nodes: int = Field(16, description="Target nodes per segment")
    embed: bool = Field(False, description="Run ML embedding on each segment")


class SegmentOut(BaseModel):
    segment_id: int
    type: str
    node_count: int
    length: float
    mean_curvature: float
    max_curvature: float
    arc_angle_deg: float
    radius_est: float
    downsampled_nodes: List[List[float]]
    downsampled_edges: List[List[int]]
    embedding: Optional[List[float]] = None


class SegmentationResponse(BaseModel):
    segments: List[SegmentOut]
    summary: dict


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "Centerline Segmentation API",
        "version": "1.0.0",
        "model_loaded": _MODEL is not None,
        "endpoints": {
            "POST /segment": "Segment a graph -> junctions/straights/arcs",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/segment", response_model=SegmentationResponse)
def segment_graph(data: GraphInput):
    """
    Full pipeline: split -> classify -> downsample -> (optional) embed.

    Returns list of segments with type, angle, curvature, downsampled geometry.
    """
    # Validate
    nodes = np.asarray(data.nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise HTTPException(400, "nodes must be Nx3")
    if len(data.edges) == 0:
        raise HTTPException(400, "edges cannot be empty")

    edges = [(int(e[0]), int(e[1])) for e in data.edges]

    # 1. Segment
    params = SegmentParams(target_step=data.target_step)
    try:
        segments = split_centerline_graph(nodes, edges, params)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    # 2. Downsample + (optional) embed
    output: List[SegmentOut] = []
    for seg in segments:
        seg_pts = nodes[seg.node_ids_original]
        ds_pts, ds_edges = downsample_segment_for_model(
            seg_pts, target_nodes=data.downsample_nodes
        )

        # Recompute arc angle on downsampled for consistency
        ds_angle = estimate_arc_angle(ds_pts) if seg.type == "arc" else 0.0

        embedding = None
        if data.embed:
            embedding = _embed_segment(ds_pts, ds_edges)

        output.append(SegmentOut(
            segment_id=seg.segment_id,
            type=seg.type,
            node_count=len(seg.node_ids_original),
            length=round(seg.length, 3),
            mean_curvature=round(seg.mean_curvature, 4),
            max_curvature=round(seg.max_curvature, 4),
            arc_angle_deg=round(seg.arc_angle_deg, 1),
            radius_est=round(seg.radius_est, 1),
            downsampled_nodes=[[round(v, 6) for v in pt] for pt in ds_pts.tolist()],
            downsampled_edges=ds_edges,
            embedding=embedding,
        ))

    # Summary
    type_counts: dict = {}
    for seg in segments:
        type_counts[seg.type] = type_counts.get(seg.type, 0) + 1

    return SegmentationResponse(
        segments=output,
        summary={
            "total_segments": len(segments),
            "counts_by_type": type_counts,
            "input_nodes": len(nodes),
            "input_edges": len(edges),
            "model_available": _MODEL is not None,
        },
    )


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
