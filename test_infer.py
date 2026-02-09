#!/usr/bin/env python3
"""
test_infer.py
─────────────
Pick 3 samples per category from val set, compute embeddings via infer.py,
and check that same-class embeddings are closer than cross-class ones.

Usage
─────
    python test_infer.py
    python test_infer.py --ckpt processed/encoder.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from infer import load_model, graph_to_pyg, embed_one, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate embeddings on val samples")
    p.add_argument("--ckpt", type=str, default="processed/encoder.pt")
    p.add_argument("--val-dir", type=str, default="dataset/val")
    p.add_argument("--n-per-class", type=int, default=3)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_val_samples(val_dir: str, n_per_class: int) -> Dict[str, List[Dict]]:
    """Load n_per_class samples from each .jsonl file in val_dir."""
    samples: Dict[str, List[Dict]] = {}
    val_path = Path(val_dir)

    for jsonl_file in sorted(val_path.glob("*.jsonl")):
        category = jsonl_file.stem
        records = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    if len(records) >= n_per_class:
                        break
        samples[category] = records

    return samples


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    model, _ = load_model(args.ckpt, device)

    # load samples
    samples = load_val_samples(args.val_dir, args.n_per_class)
    print(f"\n[data] loaded {sum(len(v) for v in samples.values())} samples "
          f"from {len(samples)} categories ({args.n_per_class} each)\n")

    # compute embeddings
    embeddings: Dict[str, List[np.ndarray]] = {}
    for cat, records in sorted(samples.items()):
        embeddings[cat] = []
        for i, rec in enumerate(records):
            nodes = np.asarray(rec["nodes"], dtype=np.float32)
            # use only forward edges (infer.py makes them bidirectional)
            edges = rec["edges"]
            # strip to just nodes+edges (what a real user would provide)
            raw_edges = []
            seen = set()
            for e in edges:
                u, v = int(e[0]), int(e[1])
                if (u, v) not in seen and (v, u) not in seen:
                    raw_edges.append([u, v])
                    seen.add((u, v))

            data = graph_to_pyg(nodes, raw_edges)
            emb = embed_one(model, data, device)
            embeddings[cat].append(emb)

    # ── within-class similarity ──
    print("=" * 60)
    print("WITHIN-CLASS cosine similarity (should be HIGH)")
    print("=" * 60)
    within_sims = {}
    for cat, embs in sorted(embeddings.items()):
        sims = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sims.append(cosine_sim(embs[i], embs[j]))
        avg = float(np.mean(sims)) if sims else 0.0
        within_sims[cat] = avg
        print(f"  {cat:>14s}:  avg_sim = {avg:.4f}  ({len(sims)} pairs)")

    # ── cross-class similarity ──
    print()
    print("=" * 60)
    print("CROSS-CLASS cosine similarity (should be LOW)")
    print("=" * 60)
    cats = sorted(embeddings.keys())
    cross_sims = []
    cross_by_pair: Dict[str, float] = {}
    for ci in range(len(cats)):
        for cj in range(ci + 1, len(cats)):
            cat_a, cat_b = cats[ci], cats[cj]
            sims = []
            for ea in embeddings[cat_a]:
                for eb in embeddings[cat_b]:
                    sims.append(cosine_sim(ea, eb))
            avg = float(np.mean(sims))
            cross_sims.append(avg)
            cross_by_pair[f"{cat_a} vs {cat_b}"] = avg

    # show top-5 most similar cross-class pairs (potential confusion)
    sorted_cross = sorted(cross_by_pair.items(), key=lambda x: -x[1])
    print("\n  Top-5 most similar cross-class pairs:")
    for pair, sim in sorted_cross[:5]:
        print(f"    {pair:>35s}:  {sim:.4f}")

    print(f"\n  Overall cross-class mean: {np.mean(cross_sims):.4f}")

    # ── arc angle ordering check ──
    arc_cats = [c for c in cats if c.startswith("arc_")]
    if len(arc_cats) >= 2:
        print()
        print("=" * 60)
        print("ARC ANGLE ORDERING (nearby angles should have higher similarity)")
        print("=" * 60)

        # sort by angle
        arc_cats_sorted = sorted(arc_cats, key=lambda c: int(c.split("_")[1]))

        # compute mean embedding per arc class
        arc_means = {}
        for cat in arc_cats_sorted:
            arc_means[cat] = np.mean(embeddings[cat], axis=0)

        # pairwise similarity matrix (condensed: consecutive pairs)
        print("\n  Consecutive arc pairs:")
        for i in range(len(arc_cats_sorted) - 1):
            ca, cb = arc_cats_sorted[i], arc_cats_sorted[i + 1]
            sim = cosine_sim(arc_means[ca], arc_means[cb])
            print(f"    {ca:>8s} <-> {cb:<8s}:  {sim:.4f}")

        # far-apart pairs
        print("\n  Far-apart arc pairs:")
        far_pairs = [
            (arc_cats_sorted[0], arc_cats_sorted[-1]),
            (arc_cats_sorted[0], arc_cats_sorted[len(arc_cats_sorted) // 2]),
        ]
        for ca, cb in far_pairs:
            sim = cosine_sim(arc_means[ca], arc_means[cb])
            print(f"    {ca:>8s} <-> {cb:<8s}:  {sim:.4f}")

    # ── summary ──
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_within = float(np.mean(list(within_sims.values())))
    avg_cross = float(np.mean(cross_sims))
    gap = avg_within - avg_cross
    print(f"  avg within-class sim : {avg_within:.4f}")
    print(f"  avg cross-class sim  : {avg_cross:.4f}")
    print(f"  gap (within - cross) : {gap:.4f}  {'GOOD' if gap > 0.1 else 'WEAK' if gap > 0 else 'BAD'}")
    print()


if __name__ == "__main__":
    main()
