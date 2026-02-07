#!/usr/bin/env python3
"""
generate_geometry_dataset.py
────────────────────────────
Main entry-point.  Builds the synthetic graph-based geometry dataset
and visualises samples per category.

Usage
─────
    python generate_geometry_dataset.py
"""

from __future__ import annotations

import json
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from shape_generators import (
    generate_straight,
    generate_arc,
    generate_corner,
    generate_junction_T,
    generate_junction_Y,
)
from graph_utils import build_shape_dict
from visualize import plot_samples, plot_gallery, plot_category_sheets

# ─── configuration ────────────────────────────────────────────────────
SEED = 42
OUTPUT_DIR = Path("dataset_output")
SAVE_FORMAT = "json"

# ─── counts ───────────────────────────────────────────────────────────
N_STRAIGHT  = 10000
N_CORNER    = 10000
N_JUNCT_T   = 10000
N_JUNCT_Y   = 10000
N_PER_ARC   = 10000
ARC_ANGLES  = list(range(10, 180, 10))


# ─── generation helpers ───────────────────────────────────────────────

def _make_samples(category: str, n: int, gen_fn, rng, **kw) -> List[Dict[str, Any]]:
    """Call *gen_fn* n times, wrap each result in a shape dict."""
    samples = []
    for _ in range(n):
        nodes, edges = gen_fn(rng=rng, **kw)
        samples.append(build_shape_dict(category, nodes, edges))
    return samples


def build_dataset(seed: int = SEED) -> List[Dict[str, Any]]:
    """Assemble the full dataset list."""
    rng = np.random.default_rng(seed)
    dataset: List[Dict[str, Any]] = []

    t0 = time.time()
    print(f"[gen] seed={seed}")

    # 1. Straight
    print(f"  → straight  ×{N_STRAIGHT}")
    dataset += _make_samples("straight", N_STRAIGHT, generate_straight, rng)

    # 2. Arcs (one sub-class per angle)
    for angle in ARC_ANGLES:
        cat = f"arc_{angle}"
        print(f"  → {cat}  ×{N_PER_ARC}")
        dataset += _make_samples(cat, N_PER_ARC, generate_arc, rng,
                                 angle_deg=float(angle))

    # 3. Corner
    print(f"  → corner  ×{N_CORNER}")
    dataset += _make_samples("corner", N_CORNER, generate_corner, rng)

    # 4. Junction T
    print(f"  → junction_T  ×{N_JUNCT_T}")
    dataset += _make_samples("junction_T", N_JUNCT_T, generate_junction_T, rng)

    # 5. Junction Y
    print(f"  → junction_Y  ×{N_JUNCT_Y}")
    dataset += _make_samples("junction_Y", N_JUNCT_Y, generate_junction_Y, rng)

    elapsed = time.time() - t0
    print(f"[gen] done — {len(dataset)} samples in {elapsed:.2f}s")
    return dataset


# ─── serialisation ────────────────────────────────────────────────────

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types so json.dumps works."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serialisable(v) for v in obj]
    return obj


def save_dataset(dataset: List[Dict[str, Any]],
                 out_dir: Path = OUTPUT_DIR,
                 fmt: str = SAVE_FORMAT) -> Path:
    """Persist the dataset to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "pickle":
        path = out_dir / "dataset.pkl"
        with open(path, "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif fmt == "json":
        path = out_dir / "dataset.json"
        with open(path, "w") as f:
            json.dump(_to_serialisable(dataset), f, indent=2)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[io]  saved → {path}  ({size_mb:.2f} MB)")
    return path


# ─── first-per-category sampler for visualisation ────────────────────

def first_per_category(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the first sample encountered for every unique category."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for s in dataset:
        cat = s["category"]
        if cat not in seen:
            seen.add(cat)
            out.append(s)
    return out


# ─── summary ──────────────────────────────────────────────────────────

def print_summary(dataset: List[Dict[str, Any]]) -> None:
    """Print a compact per-category count table."""
    from collections import Counter
    counts = Counter(s["category"] for s in dataset)
    print("\n┌─────────────────────────┬────────┐")
    print("│ category                │  count │")
    print("├─────────────────────────┼────────┤")
    for cat in sorted(counts):
        print(f"│ {cat:<23} │ {counts[cat]:>6} │")
    print("├─────────────────────────┼────────┤")
    print(f"│ {'TOTAL':<23} │ {len(dataset):>6} │")
    print("└─────────────────────────┴────────┘\n")


# ─── main ─────────────────────────────────────────────────────────────

def main() -> None:
    dataset = build_dataset()
    print_summary(dataset)

    # save to disk
    save_dataset(dataset)

    # visualise first sample per category
    representatives = first_per_category(dataset)
    plot_samples(representatives,
                 title="Sample per category",
                 save_path=str(OUTPUT_DIR / "overview.png"))

    # gallery: all samples per category (rows = categories, cols = samples)
    plot_gallery(dataset,
                 max_per_cat=10,
                 title="Gallery -- 10 samples per category",
                 save_path=str(OUTPUT_DIR / "gallery.png"))

    # separate PNG per category
    plot_category_sheets(dataset, max_per_cat=10,
                         out_dir=str(OUTPUT_DIR / "categories"))


if __name__ == "__main__":
    main()
