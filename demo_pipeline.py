#!/usr/bin/env python3
"""
demo_pipeline.py
────────────────
End-to-end demo with a HIGH-RESOLUTION composite graph:
  1. Generate a large detailed centerline (500+ nodes, fine spacing)
  2. Visualise full graph (before splitting)
  3. Split into segments
  4. Visualise segments colour-coded
  5. Downsample each segment to model resolution (~16 nodes)
  6. Visualise downsampled segments (overlay on original)
  7. Embed each downsampled segment with the trained model
  8. Show embedding similarities

Usage
─────
    python demo_pipeline.py
    python demo_pipeline.py --ckpt processed/encoder.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from centerline_segmenter import (
    split_centerline_graph,
    adaptive_downsample,
    downsample_segment_for_model,
    SegmentParams,
    SegmentRecord,
    _polyline_arc_lengths,
    estimate_arc_angle,
)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo: high-res graph -> split -> downsample -> embed")
    p.add_argument("--ckpt", type=str, default="processed/encoder.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=str, default="demo_output")
    p.add_argument("--target-nodes", type=int, default=16,
                   help="Target nodes per segment after downsampling")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# GENERATE HIGH-RESOLUTION COMPOSITE GRAPH
# ═══════════════════════════════════════════════════════════════════════

def generate_highres_composite(seed: int = 42,
                               step: float = 0.15
                               ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Build a realistic pipe-like centerline with FINE spacing (~0.15 units).

    Layout:
        straight(15) -> arc_90(R=8) -> straight(10) -> T-junction -> {
            branch A: arc_60(R=6) -> straight(8)
            branch B: straight(12) -> arc_120(R=5) -> straight(6)
        }

    With step=0.15 this gives ~500+ nodes (vs ~95 at step=1.0).
    """
    all_nodes: List[np.ndarray] = []
    all_edges: List[Tuple[int, int]] = []
    offset = 0

    def _add_polyline(pts: np.ndarray) -> int:
        nonlocal offset
        start = offset
        for i in range(len(pts)):
            all_nodes.append(pts[i])
        for i in range(len(pts) - 1):
            all_edges.append((start + i, start + i + 1))
        offset += len(pts)
        return start

    def _connect(a: int, b: int):
        all_edges.append((a, b))

    def _line(p0, direction, length):
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        n = max(2, int(round(length / step)) + 1)
        return np.array([p0 + d * t for t in np.linspace(0, length, n)])

    def _arc_xy(center, radius, angle_start, angle_end, z=0.0):
        arc_len = abs(radius * (angle_end - angle_start))
        n = max(3, int(round(arc_len / step)) + 1)
        angles = np.linspace(angle_start, angle_end, n)
        pts = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(n, z),
        ])
        return pts

    def _arc_xz(center, radius, angle_start, angle_end, y=0.0):
        arc_len = abs(radius * (angle_end - angle_start))
        n = max(3, int(round(arc_len / step)) + 1)
        angles = np.linspace(angle_start, angle_end, n)
        pts = np.column_stack([
            center[0] + radius * np.cos(angles),
            np.full(n, y),
            center[2] + radius * np.sin(angles),
        ])
        return pts

    # ── seg 1: straight along +X, 15 units ──
    s1 = _line([0, 0, 0], [1, 0, 0], 15.0)
    _add_polyline(s1)
    end1 = offset - 1

    # ── seg 2: 90° arc in XY plane, R=8, center at (15, 8, 0) ──
    R2 = 8.0
    arc2 = _arc_xy(center=[15, R2, 0], radius=R2,
                   angle_start=-np.pi / 2, angle_end=0)
    start2 = _add_polyline(arc2[1:])
    _connect(end1, start2)
    end2 = offset - 1

    # ── seg 3: straight along +Y, 10 units ──
    p2 = all_nodes[end2]
    s3 = _line(p2, [0, 1, 0], 10.0)
    start3 = _add_polyline(s3[1:])
    _connect(end2, start3)
    end3 = offset - 1

    # ── T-junction at end3 ──
    junc = end3
    p_j = all_nodes[junc]

    # ── branch A: arc_60 curving into +Z (R=6), then straight 8 ──
    R_a = 6.0
    arc_a = _arc_xz(center=[p_j[0] + R_a, p_j[1], p_j[2]], radius=R_a,
                    angle_start=np.pi, angle_end=np.pi + np.pi / 3,
                    y=p_j[1])
    start_a = _add_polyline(arc_a[1:])
    _connect(junc, start_a)
    end_a = offset - 1

    # straight extension from arc_a
    p_a = all_nodes[end_a]
    dir_a = all_nodes[end_a] - all_nodes[end_a - 1]
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-12)
    ext_a = _line(p_a, dir_a, 8.0)
    start_ext_a = _add_polyline(ext_a[1:])
    _connect(end_a, start_ext_a)

    # ── branch B: straight 12 along -X, then arc_120 (R=5), then straight 6 ──
    s_b1 = _line(p_j, [-1, 0, 0], 12.0)
    start_b1 = _add_polyline(s_b1[1:])
    _connect(junc, start_b1)
    end_b1 = offset - 1

    # arc_120 curving down in XZ plane
    p_b1 = all_nodes[end_b1]
    R_b = 5.0
    arc_b = _arc_xz(center=[p_b1[0], p_b1[1], p_b1[2] - R_b], radius=R_b,
                    angle_start=np.pi / 2, angle_end=np.pi / 2 - 2 * np.pi / 3,
                    y=p_b1[1])
    start_b2 = _add_polyline(arc_b[1:])
    _connect(end_b1, start_b2)
    end_b2 = offset - 1

    # final straight
    p_b2 = all_nodes[end_b2]
    dir_b = all_nodes[end_b2] - all_nodes[end_b2 - 1]
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-12)
    ext_b = _line(p_b2, dir_b, 6.0)
    start_ext_b = _add_polyline(ext_b[1:])
    _connect(end_b2, start_ext_b)

    nodes = np.array(all_nodes, dtype=np.float64)
    return nodes, all_edges


# ═══════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════

_COLOURS = {"junction": "#d62728", "straight": "#1f77b4", "arc": "#ff7f0e"}


def _draw_edges(ax, nodes, edges, color="#888888", lw=0.8, zorder=1):
    for u, v in edges:
        ax.plot([nodes[u, 0], nodes[v, 0]],
                [nodes[u, 1], nodes[v, 1]],
                [nodes[u, 2], nodes[v, 2]],
                color=color, linewidth=lw, zorder=zorder)


def plot_full_graph(nodes, edges, title, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_edges(ax, nodes, edges, color="#888888", lw=0.6)
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], color="black", s=3, zorder=5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[vis] saved {save_path}")
    plt.close(fig)


def plot_segments(nodes, edges, segments, title, save_path=None):
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    _draw_edges(ax, nodes, edges, color="#eeeeee", lw=0.3)

    for seg in segments:
        c = _COLOURS.get(seg.type, "#333")
        seg_set = set(seg.node_ids_original)
        for u, v in edges:
            if u in seg_set and v in seg_set:
                ax.plot([nodes[u, 0], nodes[v, 0]],
                        [nodes[u, 1], nodes[v, 1]],
                        [nodes[u, 2], nodes[v, 2]],
                        color=c, linewidth=2, zorder=3)
        seg_n = nodes[seg.node_ids_original]
        ax.scatter(seg_n[:, 0], seg_n[:, 1], seg_n[:, 2], color=c, s=5, zorder=5)
        mid = seg_n[len(seg_n) // 2]
        lbl = seg.type
        if seg.type == "arc":
            lbl = f"arc {seg.arc_angle_deg:.0f}°"
        ax.text(mid[0], mid[1], mid[2], f"  {lbl}", fontsize=7, color=c, fontweight="bold")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, linewidth=3, label=t) for t, c in _COLOURS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[vis] saved {save_path}")
    plt.close(fig)


def plot_downsample_comparison(nodes, edges, segments, ds_data, title, save_path=None):
    """
    Side-by-side: left = original segments, right = downsampled.
    """
    fig = plt.figure(figsize=(20, 8))

    # left: original
    ax1 = fig.add_subplot(121, projection="3d")
    _draw_edges(ax1, nodes, edges, color="#eeeeee", lw=0.2)
    for seg in segments:
        c = _COLOURS.get(seg.type, "#333")
        seg_n = nodes[seg.node_ids_original]
        ax1.scatter(seg_n[:, 0], seg_n[:, 1], seg_n[:, 2], color=c, s=4, zorder=5)
        for i in range(len(seg_n) - 1):
            ax1.plot([seg_n[i, 0], seg_n[i + 1, 0]],
                     [seg_n[i, 1], seg_n[i + 1, 1]],
                     [seg_n[i, 2], seg_n[i + 1, 2]],
                     color=c, linewidth=1.5, zorder=3)
        mid = seg_n[len(seg_n) // 2]
        lbl = f"{seg.type}" if seg.type != "arc" else f"arc {seg.arc_angle_deg:.0f}°"
        ax1.text(mid[0], mid[1], mid[2], f"  {lbl} ({len(seg_n)}pts)",
                 fontsize=6, color=c)
    ax1.set_title("Original resolution", fontsize=11)

    # right: downsampled
    ax2 = fig.add_subplot(122, projection="3d")
    _draw_edges(ax2, nodes, edges, color="#f0f0f0", lw=0.15)
    for seg_id, (ds_pts, ds_edges, seg) in ds_data.items():
        c = _COLOURS.get(seg.type, "#333")
        ax2.scatter(ds_pts[:, 0], ds_pts[:, 1], ds_pts[:, 2],
                    color=c, s=30, zorder=5, edgecolors="black", linewidths=0.3)
        for i, j in ds_edges:
            ax2.plot([ds_pts[i, 0], ds_pts[j, 0]],
                     [ds_pts[i, 1], ds_pts[j, 1]],
                     [ds_pts[i, 2], ds_pts[j, 2]],
                     color=c, linewidth=2.5, zorder=3)
        mid = ds_pts[len(ds_pts) // 2]
        lbl = f"{seg.type}" if seg.type != "arc" else f"arc {seg.arc_angle_deg:.0f}°"
        ax2.text(mid[0], mid[1], mid[2], f"  {lbl} ({len(ds_pts)}pts)",
                 fontsize=6, color=c)
    ax2.set_title("Downsampled for model (~16 nodes)", fontsize=11)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[vis] saved {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════

def embed_segments(ds_data, ckpt_path, device_str):
    try:
        import torch
        from infer import load_model, graph_to_pyg, embed_one, pick_device
    except ImportError:
        print("[warn] torch/infer not available, skipping embedding")
        return {}

    device = pick_device(device_str)
    try:
        model, _ = load_model(ckpt_path, device)
    except FileNotFoundError:
        print(f"[warn] checkpoint {ckpt_path} not found, skipping embedding")
        return {}

    embeddings = {}
    for seg_id, (ds_pts, ds_edges, seg) in ds_data.items():
        if len(ds_pts) < 3:
            continue
        # infer.py's graph_to_pyg handles centering + curvature + degree
        data = graph_to_pyg(ds_pts.astype(np.float32), ds_edges)
        emb = embed_one(model, data, device)
        embeddings[seg_id] = emb
    return embeddings


def print_similarity(segments, ds_data, embeddings):
    ids = [s.segment_id for s in segments if s.segment_id in embeddings]
    if len(ids) < 2:
        print("[info] not enough embeddings for similarity")
        return

    labels = []
    for s in segments:
        if s.segment_id in embeddings:
            lbl = s.type
            if s.type == "arc":
                lbl = f"arc_{s.arc_angle_deg:.0f}"
            n_orig = len(s.node_ids_original)
            n_ds = len(ds_data[s.segment_id][0]) if s.segment_id in ds_data else 0
            labels.append(f"[{s.segment_id}]{lbl}({n_orig}->{n_ds})")

    embs = np.array([embeddings[i] for i in ids])
    sim = embs @ embs.T

    print("\n── embedding cosine similarity ──")
    header = "                    " + "  ".join(f"{l:>20s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>20s}"
        for j in range(len(labels)):
            row += f"  {sim[i, j]:>20.3f}"
        print(row)
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. generate high-res graph ──
    print("=" * 65)
    print("STEP 1: Generate HIGH-RESOLUTION composite graph")
    print("=" * 65)
    nodes, edges = generate_highres_composite(seed=args.seed, step=0.15)
    print(f"  {len(nodes)} nodes, {len(edges)} edges  (step=0.15)")

    graph_path = out / "composite_graph.json"
    with open(graph_path, "w") as f:
        json.dump({"nodes": nodes.tolist(),
                    "edges": [[int(u), int(v)] for u, v in edges]}, f)
    print(f"  saved {graph_path}")

    # ── 2. visualise full graph ──
    print(f"\n{'=' * 65}")
    print("STEP 2: Visualise full graph (before splitting)")
    print("=" * 65)
    plot_full_graph(nodes, edges,
                    title=f"High-res composite ({len(nodes)} nodes, step=0.15)",
                    save_path=str(out / "01_full_graph.png"))

    # ── 3. split ──
    print(f"\n{'=' * 65}")
    print("STEP 3: Split into segments")
    print("=" * 65)
    params = SegmentParams(target_step=0.15)
    segments = split_centerline_graph(nodes, edges, params)

    # ── 4. visualise segments ──
    print(f"\n{'=' * 65}")
    print("STEP 4: Visualise segments (colour-coded)")
    print("=" * 65)
    plot_segments(nodes, edges, segments,
                  title=f"Segmented: {len(segments)} segments",
                  save_path=str(out / "02_segments.png"))

    for seg in segments:
        extra = ""
        if seg.type == "arc":
            extra = f"  angle={seg.arc_angle_deg:.1f}° R={seg.radius_est:.1f}"
        print(f"  [{seg.segment_id:>2d}] {seg.type:>8s}  "
              f"nodes={len(seg.node_ids_original):>4d}  "
              f"len={seg.length:>7.1f}{extra}")

    # ── 5. downsample each segment ──
    print(f"\n{'=' * 65}")
    print(f"STEP 5: Downsample to ~{args.target_nodes} nodes per segment")
    print("=" * 65)
    ds_data: Dict[int, Tuple] = {}
    for seg in segments:
        seg_pts = nodes[seg.node_ids_original]
        ds_pts, ds_edges = downsample_segment_for_model(
            seg_pts, target_nodes=args.target_nodes
        )
        ds_data[seg.segment_id] = (ds_pts, ds_edges, seg)

        # recompute arc angle on downsampled points
        ds_angle = estimate_arc_angle(ds_pts) if seg.type == "arc" else 0.0

        print(f"  [{seg.segment_id:>2d}] {seg.type:>8s}  "
              f"{len(seg.node_ids_original):>4d} -> {len(ds_pts):>3d} nodes"
              f"  (curvature preserved: "
              f"orig_angle={seg.arc_angle_deg:.1f}° ds_angle={ds_angle:.1f}°)"
              if seg.type == "arc" else
              f"  [{seg.segment_id:>2d}] {seg.type:>8s}  "
              f"{len(seg.node_ids_original):>4d} -> {len(ds_pts):>3d} nodes")

    # ── 6. visualise downsampled vs original ──
    print(f"\n{'=' * 65}")
    print("STEP 6: Visualise original vs downsampled (side by side)")
    print("=" * 65)
    plot_downsample_comparison(
        nodes, edges, segments, ds_data,
        title="Original resolution vs Downsampled for model",
        save_path=str(out / "03_downsample_comparison.png"),
    )

    # ── 7. embed ──
    print(f"\n{'=' * 65}")
    print("STEP 7: Embed each downsampled segment")
    print("=" * 65)
    embeddings = embed_segments(ds_data, ckpt_path=args.ckpt, device_str=args.device)

    if embeddings:
        for seg in segments:
            if seg.segment_id in embeddings:
                emb = embeddings[seg.segment_id]
                lbl = seg.type
                if seg.type == "arc":
                    lbl = f"arc_{seg.arc_angle_deg:.0f}°"
                n_ds = len(ds_data[seg.segment_id][0])
                print(f"  [{seg.segment_id:>2d}] {lbl:>12s} ({n_ds}pts)  "
                      f"emb[:5]={emb[:5].round(3)}")

        print_similarity(segments, ds_data, embeddings)
    else:
        print("  (skipped — no checkpoint or torch unavailable)")

    print(f"\nDone. Outputs in {out}/")
    print(f"  01_full_graph.png            - raw high-res graph")
    print(f"  02_segments.png              - after splitting")
    print(f"  03_downsample_comparison.png - original vs downsampled")


if __name__ == "__main__":
    main()
