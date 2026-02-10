"""
services/segmentation.py
────────────────────────
Segmentation service: split → downsample → embed → store.

Orchestrates the full pipeline and persists results per session.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from centerline_segmenter import (
    split_centerline_graph,
    downsample_segment_for_model,
    estimate_arc_angle,
    SegmentParams,
    SegmentRecord,
    _polyline_arc_lengths,
)


def segment_graph(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    target_step: float = 1.0,
) -> List[SegmentRecord]:
    """Run segmentation pipeline."""
    params = SegmentParams(target_step=target_step)
    return split_centerline_graph(nodes, edges, params)


def downsample_segments(
    nodes: np.ndarray,
    segments: List[SegmentRecord],
    target_nodes: int = 16,
) -> Dict[int, Dict[str, Any]]:
    """
    Downsample each segment to uniform arc-length spacing.

    Returns {segment_id: {"nodes": ndarray, "edges": list, "segment": SegmentRecord}}
    """
    result = {}
    for seg in segments:
        seg_pts = nodes[seg.node_ids_original]
        ds_pts, ds_edges = downsample_segment_for_model(seg_pts, target_nodes=target_nodes)
        result[seg.segment_id] = {
            "nodes": ds_pts,
            "edges": ds_edges,
            "segment": seg,
        }
    return result


def embed_segments(
    ds_data: Dict[int, Dict[str, Any]],
    model=None,
    device=None,
) -> Dict[int, np.ndarray]:
    """
    Embed each downsampled segment with the ML model.
    Returns {segment_id: embedding_vector}.
    Gracefully returns empty if model is None.
    """
    if model is None or device is None:
        return {}

    try:
        from inference import graph_to_pyg, embed_one
    except ImportError:
        return {}

    embeddings = {}
    for seg_id, data in ds_data.items():
        pts = data["nodes"]
        edges = data["edges"]
        if len(pts) < 3:
            continue
        try:
            pyg_data = graph_to_pyg(pts.astype(np.float32), edges)
            emb = embed_one(model, pyg_data, device)
            embeddings[seg_id] = emb
        except Exception:
            continue

    return embeddings


def compute_face_ids_for_segments(
    segments: List[SegmentRecord],
    session_dir: Path,
) -> Dict[int, List[int]]:
    """
    Compute which surface mesh faces belong to each segment.

    Uses the cl_to_mesh_map (centerline node → mesh vertex indices)
    stored during ingest. For each segment, collects the mesh vertices
    associated with its centerline nodes, then finds all faces that
    have at least one vertex in that set.

    Returns:
        {segment_id: [face_idx, face_idx, ...]}
    """
    mesh_path = session_dir / "mesh.json"
    if not mesh_path.exists():
        return {}

    mesh_data = json.loads(mesh_path.read_text())
    faces = mesh_data.get("faces", [])
    cl_to_mesh = mesh_data.get("cl_to_mesh_map", {})

    if not faces or not cl_to_mesh:
        return {}

    # Build: for each mesh vertex, which faces contain it
    vertex_to_faces: Dict[int, List[int]] = {}
    for face_idx, face in enumerate(faces):
        for v in face:
            vertex_to_faces.setdefault(int(v), []).append(face_idx)

    result: Dict[int, List[int]] = {}

    for seg in segments:
        face_set: set = set()
        for cl_node_id in seg.node_ids_original:
            # cl_to_mesh_map keys are strings (from JSON serialization)
            mesh_verts = cl_to_mesh.get(str(cl_node_id), [])
            if isinstance(mesh_verts, int):
                mesh_verts = [mesh_verts]
            for mv in mesh_verts:
                for fi in vertex_to_faces.get(int(mv), []):
                    face_set.add(fi)

        result[seg.segment_id] = sorted(face_set)

    return result


def build_segment_output(
    seg: SegmentRecord,
    ds_nodes: np.ndarray,
    ds_edges: List[List[int]],
    embedding: Optional[np.ndarray] = None,
    face_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serialisable dict for one segment."""
    out = {
        "segment_id": seg.segment_id,
        "type": seg.type,
        "node_count": len(seg.node_ids_original),
        "original_node_ids": seg.node_ids_original,
        "length": round(seg.length, 3),
        "mean_curvature": round(seg.mean_curvature, 4),
        "max_curvature": round(seg.max_curvature, 4),
        "arc_angle_deg": round(seg.arc_angle_deg, 1),
        "radius_est": round(seg.radius_est, 1),
        "corner_angle_deg": round(seg.corner_angle_deg, 1),
        "downsampled_nodes": [[round(v, 6) for v in pt] for pt in ds_nodes.tolist()],
        "downsampled_edges": ds_edges,
        "face_ids": face_ids if face_ids is not None else [],
    }
    if embedding is not None:
        out["embedding"] = embedding.tolist()
    return out


def run_full_pipeline(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    session_dir: Path,
    target_step: float = 1.0,
    target_nodes: int = 16,
    model=None,
    device=None,
    on_progress=None,
) -> Dict[str, Any]:
    """
    Full pipeline: segment → downsample → embed → store.

    on_progress: optional callback(step_name, detail_dict) for WS streaming.

    Returns the complete result dict.
    """
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Segment
    if on_progress:
        on_progress("segmenting", {"status": "splitting graph into segments"})
    segments = segment_graph(nodes, edges, target_step=target_step)

    type_counts = {}
    for s in segments:
        type_counts[s.type] = type_counts.get(s.type, 0) + 1
    if on_progress:
        on_progress("segmented", {
            "total_segments": len(segments),
            "counts_by_type": type_counts,
        })

    # 2. Downsample
    if on_progress:
        on_progress("downsampling", {"target_nodes": target_nodes})
    ds_data = downsample_segments(nodes, segments, target_nodes=target_nodes)
    if on_progress:
        on_progress("downsampled", {"segments_processed": len(ds_data)})

    # 3. Embed
    embeddings = {}
    if model is not None:
        if on_progress:
            on_progress("embedding", {"status": "running ML model"})
        embeddings = embed_segments(ds_data, model=model, device=device)
        if on_progress:
            on_progress("embedded", {"segments_embedded": len(embeddings)})

    # 4. Compute face_ids (map segments to surface mesh triangles)
    if on_progress:
        on_progress("mapping_faces", {"status": "mapping segments to surface mesh faces"})
    seg_face_ids = compute_face_ids_for_segments(segments, session_dir)
    if on_progress:
        on_progress("faces_mapped", {
            "segments_with_faces": sum(1 for v in seg_face_ids.values() if v),
        })

    # 5. Build output
    segment_outputs = []
    for seg in segments:
        ds = ds_data[seg.segment_id]
        emb = embeddings.get(seg.segment_id)
        fids = seg_face_ids.get(seg.segment_id, [])
        out = build_segment_output(seg, ds["nodes"], ds["edges"], embedding=emb, face_ids=fids)
        segment_outputs.append(out)

    result = {
        "segments": segment_outputs,
        "summary": {
            "total_segments": len(segments),
            "counts_by_type": type_counts,
            "input_nodes": len(nodes),
            "input_edges": len(edges),
            "model_used": model is not None,
        },
    }

    # 6. Store
    # JSON for API responses
    result_path = session_dir / "segments.json"
    with open(result_path, "w") as f:
        json.dump(result, f)

    # Pickle for fast reload (includes numpy arrays)
    pickle_path = session_dir / "segments.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "segments": segments,
            "ds_data": ds_data,
            "embeddings": embeddings,
            "nodes": nodes,
            "edges": edges,
        }, f)

    # NPY for embeddings
    if embeddings:
        emb_array = np.array([embeddings[s.segment_id]
                              for s in segments if s.segment_id in embeddings])
        np.save(str(session_dir / "embeddings.npy"), emb_array)

    if on_progress:
        on_progress("stored", {
            "json_path": str(result_path),
            "pickle_path": str(pickle_path),
        })

    return result


def load_session_results(session_dir: Path) -> Optional[Dict[str, Any]]:
    """Load previously computed results from session dir."""
    json_path = session_dir / "segments.json"
    if json_path.exists():
        return json.loads(json_path.read_text())

    pkl_path = session_dir / "segments.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    return None
