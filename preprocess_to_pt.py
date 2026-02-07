#!/usr/bin/env python3
"""
preprocess_to_pt.py
───────────────────
Convert JSONL splits (from prepare_dataset.py) into PyG .pt files.

Reads:   dataset/train/*.jsonl, dataset/val/*.jsonl, dataset/test/*.jsonl
         dataset/meta.json
Writes:  processed/train.pt
         processed/val.pt
         processed/test.pt
         processed/meta.json   (copy + augmented with feature info)

Each .pt contains a list[Data] where every Data has:
    x             (N, 3)   node features: [curvature, bend_rad, degree]
    pos           (N, 3)   centred 3D coordinates
    edge_index    (2, 2E)  undirected COO
    motif_type_id (int)    0=arc, 1=corner, 2=junction_T, 3=junction_Y, 4=straight
    arc_angle_deg (int)    10..170 for arcs, -1 otherwise

Usage
─────
    python preprocess_to_pt.py                        # defaults
    python preprocess_to_pt.py --jsonl-dir dataset    # custom JSONL root
    python preprocess_to_pt.py --outdir processed     # custom output
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch_geometric.data import Data


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL splits -> PyG .pt files")
    p.add_argument("--jsonl-dir", type=str, default="dataset",
                   help="Root of JSONL splits (contains train/ val/ test/ meta.json)")
    p.add_argument("--outdir", type=str, default="processed",
                   help="Output directory for .pt and meta.json")
    return p.parse_args()


# ─── load JSONL split ────────────────────────────────────────────────

def load_jsonl_split(split_dir: Path) -> List[Dict[str, Any]]:
    """Read all .jsonl files in *split_dir*, return flat list of records."""
    records: List[Dict[str, Any]] = []
    for path in sorted(split_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ─── convert one record -> Data ──────────────────────────────────────

def record_to_data(rec: Dict[str, Any],
                   motif_type_to_id: Dict[str, int]) -> Data:
    """
    Convert one canonical JSONL record into a PyG Data object.

    Node features x = [curvature, bend_rad, degree]
      - curvature:  from features.curvature
      - bend_rad:   radians(180 - segment_angle_deg)  = radians(bend_deg)
      - degree:     from features.degree
    """
    nodes = np.asarray(rec["nodes"], dtype=np.float32)       # (N, 3)
    n = len(nodes)

    # pos (already centred by prepare_dataset.py)
    pos = torch.tensor(nodes, dtype=torch.float)

    # edge_index (already bidirectional from prepare_dataset.py)
    edges = rec["edges"]  # list of [u, v]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, 2E)

    # node features
    curvature = np.asarray(rec["features"]["curvature"], dtype=np.float32)
    bend_deg = np.asarray(rec["features"]["bend_deg"], dtype=np.float32)
    bend_rad = np.deg2rad(bend_deg).astype(np.float32)
    degree = np.asarray(rec["features"]["degree"], dtype=np.float32)

    x = torch.tensor(
        np.column_stack([curvature, bend_rad, degree]),
        dtype=torch.float,
    )  # (N, 3)

    # labels
    motif_type_id = motif_type_to_id[rec["motif_type"]]
    arc_angle_deg = rec["arc_angle_deg"]

    return Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        motif_type_id=motif_type_id,
        arc_angle_deg=arc_angle_deg,
    )


# ─── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    jsonl_root = Path(args.jsonl_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # load meta
    meta_path = jsonl_root / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    motif_type_to_id: Dict[str, int] = meta["motif_type_to_id"]
    print(f"[meta] motif_type_to_id = {motif_type_to_id}")

    # process each split
    for split in ("train", "val", "test"):
        split_dir = jsonl_root / split
        if not split_dir.exists():
            print(f"[warn] {split_dir} not found, skipping")
            continue

        t0 = time.time()
        records = load_jsonl_split(split_dir)
        print(f"[{split}] loaded {len(records)} records in {time.time()-t0:.1f}s")

        t0 = time.time()
        data_list = [record_to_data(r, motif_type_to_id) for r in records]
        print(f"[{split}] converted {len(data_list)} Data objects in {time.time()-t0:.1f}s")

        pt_path = out / f"{split}.pt"
        torch.save(data_list, pt_path)
        size_mb = pt_path.stat().st_size / (1024 * 1024)
        print(f"[{split}] saved {pt_path}  ({size_mb:.1f} MB)")

    # copy + augment meta
    meta["node_features_pt"] = ["curvature", "bend_rad", "degree"]
    meta["node_feature_dim_pt"] = 3
    out_meta = out / "meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[io] saved {out_meta}")

    # sample check
    d = data_list[0]
    print(f"\n── sample (first from {split}) ──")
    print(f"  x            : {d.x.shape}")
    print(f"  pos          : {d.pos.shape}")
    print(f"  edge_index   : {d.edge_index.shape}")
    print(f"  motif_type_id: {d.motif_type_id}")
    print(f"  arc_angle_deg: {d.arc_angle_deg}")
    print()


if __name__ == "__main__":
    main()
