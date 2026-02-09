#!/usr/bin/env python3
"""
split_centerline.py
───────────────────
CLI tool: split a 3D centerline graph into junction / straight / arc segments.

Usage
─────
    python split_centerline.py --in graph.json --out segments.json
    python split_centerline.py --in graph.json --out segments.json --target-step 0.5
    python split_centerline.py --in graph.json --out segments.json --k-low 0.01 --k-high 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from centerline_segmenter import (
    load_graph,
    split_centerline_graph,
    segments_to_json,
    SegmentParams,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split centerline graph into segments")
    p.add_argument("--in", dest="input", type=str, required=True,
                   help="Input JSON graph (nodes + edges)")
    p.add_argument("--out", type=str, default="segments.json",
                   help="Output segments JSON")
    p.add_argument("--target-step", type=float, default=1.0,
                   help="Resample spacing (default: 1.0)")
    p.add_argument("--k-low", type=float, default=-1.0,
                   help="Straight curvature threshold (< 0 = auto)")
    p.add_argument("--k-high", type=float, default=-1.0,
                   help="Arc curvature threshold (< 0 = auto)")
    p.add_argument("--min-run", type=int, default=6,
                   help="Minimum points per segment")
    p.add_argument("--cooldown", type=int, default=4,
                   help="Cooldown points before switching from arc to straight")
    p.add_argument("--snap-window", type=int, default=3,
                   help="Snap boundary to kappa minimum within +/- this")
    p.add_argument("--smooth", type=int, default=3,
                   help="Curvature smoothing window")
    p.add_argument("--junction-hops", type=int, default=2,
                   help="BFS hops around junction nodes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # load
    nodes, edges = load_graph(args.input)
    print(f"[io] loaded {args.input}: {len(nodes)} nodes, {len(edges)} edges")

    # params
    params = SegmentParams(
        target_step=args.target_step,
        k_low=args.k_low,
        k_high=args.k_high,
        min_run_points=args.min_run,
        cooldown_points=args.cooldown,
        snap_window=args.snap_window,
        smooth_window=args.smooth,
        junction_k_hops=args.junction_hops,
    )

    # split
    segments = split_centerline_graph(nodes, edges, params)

    # save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "input": args.input,
        "params": {
            "target_step": params.target_step,
            "k_low": params.k_low,
            "k_high": params.k_high,
            "min_run_points": params.min_run_points,
            "cooldown_points": params.cooldown_points,
            "snap_window": params.snap_window,
            "smooth_window": params.smooth_window,
            "junction_k_hops": params.junction_k_hops,
        },
        "num_segments": len(segments),
        "segments": segments_to_json(segments),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[io] saved {len(segments)} segments -> {out_path}")

    # summary
    print("\n── segment summary ──")
    for seg in segments:
        extra = ""
        if seg.type == "arc":
            extra = f"  angle={seg.arc_angle_deg:.1f}deg  R={seg.radius_est:.2f}"
        print(f"  [{seg.segment_id:>3d}] {seg.type:>8s}  "
              f"nodes={len(seg.node_ids_original):>4d}  "
              f"len={seg.length:>7.2f}  "
              f"kappa=[{seg.mean_curvature:.4f}, {seg.max_curvature:.4f}]"
              f"{extra}")


if __name__ == "__main__":
    main()
