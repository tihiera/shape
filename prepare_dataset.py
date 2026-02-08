#!/usr/bin/env python3
"""
prepare_dataset.py
──────────────────
Convert the monolithic dataset.json (or .pkl) into training-ready JSONL
splits with a canonical schema.

Input:   dataset_output/dataset.json   (or .pkl)
Output:  dataset/train/<category>.jsonl
         dataset/val/<category>.jsonl
         dataset/test/<category>.jsonl
         dataset/meta.json

Each JSONL line is one self-contained graph record.

Usage
─────
    python prepare_dataset.py
    python prepare_dataset.py --input dataset_output/dataset.pkl
    python prepare_dataset.py --split 90/5/5
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple


# ─── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw JSON → JSONL training splits")
    p.add_argument("--input", type=str, default="dataset_output/dataset.json",
                   help="Path to dataset.json or dataset.pkl")
    p.add_argument("--outdir", type=str, default="dataset",
                   help="Output root directory")
    p.add_argument("--split", type=str, default="80/10/10",
                   help="Train/val/test split ratios (e.g. 80/10/10)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible splits")
    return p.parse_args()


# ─── load raw data ───────────────────────────────────────────────────

def load_raw(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    t0 = time.time()
    print(f"[load] reading {p} ...")
    if p.suffix == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
    elif p.suffix == ".json":
        with open(p, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unknown extension: {p.suffix}")
    print(f"[load] {len(data)} samples in {time.time() - t0:.1f}s")
    return data


# ─── parse category string ──────────────────────────────────────────

_ARC_RE = re.compile(r"^arc_(\d+)$")

def _parse_category(cat: str) -> Tuple[str, int]:
    """Return (motif_type, arc_angle_deg).
    """
    m = _ARC_RE.match(cat)
    if m:
        return "arc", int(m.group(1))
    if cat == "straight":
        return "straight", -1
    if cat == "corner":
        return "corner", -1
    if cat in ("junction_T", "junction_Y"):
        return "junction", -1
    return cat, -1


# ─── canonicalize one sample ────────────────────────────────────────

def _make_undirected(edges: List) -> List[List[int]]:
    """Ensure every (u,v) has a matching (v,u). Deduplicate."""
    seen = set()
    out = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if (u, v) not in seen:
            seen.add((u, v))
            seen.add((v, u))
            out.append([u, v])
            out.append([v, u])
    return out


def _compute_degree(edges_bidir: List[List[int]], n: int) -> List[int]:
    """Degree per node from bidirectional edge list."""
    deg = [0] * n
    for u, _ in edges_bidir:
        deg[u] += 1
    return deg


def canonicalize(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Transform one raw sample into the canonical JSONL schema.
    Returns None if validation fails.
    """
    category = sample["category"]
    motif_type, arc_angle_deg = _parse_category(category)

    # rewrite junction_T / junction_Y -> junction
    if category in ("junction_T", "junction_Y"):
        category = "junction"

    # --- nodes: convert & center ---
    nodes = np.asarray(sample["nodes"], dtype=np.float64)
    n = len(nodes)
    if n < 2:
        return None

    # center (remove translation)
    nodes = nodes - nodes.mean(axis=0)

    # --- features ---
    curvature = np.asarray(sample["features"]["curvature"], dtype=np.float64)
    segment_angle = np.asarray(sample["features"]["segment_angle"], dtype=np.float64)

    if len(curvature) != n or len(segment_angle) != n:
        return None

    # bend_deg = 180 - segment_angle  (0 = straight, high = sharp turn)
    bend_deg = 180.0 - segment_angle

    # enforce endpoint convention
    curvature[0] = curvature[-1] = 0.0
    bend_deg[0] = bend_deg[-1] = 0.0
    segment_angle[0] = segment_angle[-1] = 0.0

    # --- check finite ---
    if not (np.all(np.isfinite(nodes)) and
            np.all(np.isfinite(curvature)) and
            np.all(np.isfinite(segment_angle))):
        return None

    # --- edges: make undirected ---
    raw_edges = sample["edges"]
    edges_bidir = _make_undirected(raw_edges)

    # validate edge indices
    for u, v in edges_bidir:
        if u < 0 or u >= n or v < 0 or v >= n:
            return None

    if len(edges_bidir) < 4:  # at least 2 original edges → 4 bidirectional
        return None

    # --- degree ---
    degree = _compute_degree(edges_bidir, n)

    # --- round floats for compact JSONL ---
    def _r(arr, decimals=6):
        return np.round(arr, decimals).tolist()

    return {
        "category": category,
        "motif_type": motif_type,
        "arc_angle_deg": arc_angle_deg,
        "num_nodes": n,
        "num_edges": len(edges_bidir),
        "nodes": [_r(row, 4) for row in nodes],
        "edges": edges_bidir,
        "features": {
            "curvature": _r(curvature),
            "segment_angle_deg": _r(segment_angle),
            "bend_deg": _r(bend_deg),
            "degree": degree,
        },
    }


# ─── stratified split ───────────────────────────────────────────────

def stratified_split(
    samples_by_cat: Dict[str, List[Dict]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Split each category independently into train/val/test."""
    rng = np.random.default_rng(seed)
    train, val, test = {}, {}, {}

    for cat in sorted(samples_by_cat):
        items = samples_by_cat[cat]
        indices = np.arange(len(items))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train[cat] = [items[i] for i in indices[:n_train]]
        val[cat] = [items[i] for i in indices[n_train:n_train + n_val]]
        test[cat] = [items[i] for i in indices[n_train + n_val:]]

    return train, val, test


# ─── write JSONL ─────────────────────────────────────────────────────

def write_jsonl(records: List[Dict], path: Path) -> int:
    """Write records as JSONL (one JSON object per line). Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(records)


# ─── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # parse split ratios
    parts = [float(x) for x in args.split.split("/")]
    total = sum(parts)
    ratios = (parts[0] / total, parts[1] / total, parts[2] / total)
    print(f"[cfg] split = train {ratios[0]:.0%} / val {ratios[1]:.0%} / test {ratios[2]:.0%}")

    # load
    raw = load_raw(args.input)

    # canonicalize
    print(f"[prep] canonicalizing {len(raw)} samples ...")
    t0 = time.time()
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    n_ok, n_drop = 0, 0

    for sample in raw:
        record = canonicalize(sample)
        if record is None:
            n_drop += 1
            continue
        by_cat[record["category"]].append(record)
        n_ok += 1

    elapsed = time.time() - t0
    print(f"[prep] kept {n_ok}, dropped {n_drop} in {elapsed:.1f}s")

    if n_drop > 0:
        print(f"[warn] {n_drop} samples failed validation and were skipped")

    # hard-fail: no old junction_T / junction_Y labels may survive
    _BANNED = {"junction_T", "junction_Y"}
    for cat, records in by_cat.items():
        assert cat not in _BANNED, f"FATAL: category '{cat}' still exists after merge!"
        for r in records:
            assert r["motif_type"] not in _BANNED, \
                f"FATAL: motif_type '{r['motif_type']}' still exists after merge!"
    print("[check] no junction_T / junction_Y labels remain -- OK")

    # split
    train, val, test = stratified_split(by_cat, ratios, seed=args.seed)

    # write JSONL
    out = Path(args.outdir)
    counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    total_written = 0

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for cat in sorted(split_data):
            records = split_data[cat]
            if not records:
                continue
            path = out / split_name / f"{cat}.jsonl"
            n = write_jsonl(records, path)
            counts[split_name][cat] = n
            total_written += n

    print(f"\n[io] wrote {total_written} records to {out}/")

    # summary table
    all_cats = sorted(by_cat.keys())
    print(f"\n{'category':>14}  {'train':>7}  {'val':>5}  {'test':>5}  {'total':>7}")
    print("-" * 48)
    for cat in all_cats:
        tr = counts["train"].get(cat, 0)
        va = counts["val"].get(cat, 0)
        te = counts["test"].get(cat, 0)
        print(f"{cat:>14}  {tr:>7}  {va:>5}  {te:>5}  {tr+va+te:>7}")
    tr_t = sum(counts["train"].values())
    va_t = sum(counts["val"].values())
    te_t = sum(counts["test"].values())
    print("-" * 48)
    print(f"{'TOTAL':>14}  {tr_t:>7}  {va_t:>5}  {te_t:>5}  {tr_t+va_t+te_t:>7}")

    # meta.json
    motif_types = sorted({r["motif_type"] for cat_list in by_cat.values() for r in cat_list})
    motif_type_to_id = {m: i for i, m in enumerate(motif_types)}
    cat_to_id = {c: i for i, c in enumerate(all_cats)}
    arc_angles = sorted([
        int(c.split("_")[1]) for c in all_cats if c.startswith("arc_")
    ])

    meta = {
        "categories": all_cats,
        "cat_to_id": cat_to_id,
        "motif_types": motif_types,
        "motif_type_to_id": motif_type_to_id,
        "arc_angles": arc_angles,
        "node_feature_names": ["curvature", "segment_angle_deg", "bend_deg", "degree"],
        "node_feature_dim": 4,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "seed": args.seed,
        "total_samples": n_ok,
        "counts_per_split": dict(counts),
    }
    meta_path = out / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[io] saved {meta_path}")

    # sample inspection
    first_cat = all_cats[0]
    s = train[first_cat][0]
    print(f"\n── sample record ({first_cat}, train) ──")
    print(f"  motif_type    : {s['motif_type']}")
    print(f"  arc_angle_deg : {s['arc_angle_deg']}")
    print(f"  num_nodes     : {s['num_nodes']}")
    print(f"  num_edges     : {s['num_edges']}  (bidirectional)")
    print(f"  nodes[0]      : {s['nodes'][0]}")
    print(f"  edges[0:4]    : {s['edges'][:4]}")
    print(f"  features keys : {list(s['features'].keys())}")
    print()


if __name__ == "__main__":
    main()
