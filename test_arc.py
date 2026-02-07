#!/usr/bin/env python3
"""Quick test: verify every arc class gets ~same node count, correct total turn,
and uniform node spacing."""
from shape_generators import generate_arc, generate_straight, generate_corner, generate_junction_T, generate_junction_Y
import numpy as np

rng = np.random.default_rng(42)

print("=" * 75)
print("ARC VERIFICATION")
print(f"{'arc':>8}  {'nodes':>5}  {'arc_len':>8}  {'radius':>8}  {'total_turn':>10}  {'spacing':>18}")
print("-" * 75)

for angle in [10, 30, 60, 90, 120, 150, 170]:
    nodes, edges = generate_arc(angle_deg=angle, rng=rng)

    # edge lengths (distances between consecutive nodes)
    dists = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    arc_len = dists.sum()

    # total angular deviation
    vecs = np.diff(nodes, axis=0)
    total_turn = 0.0
    for i in range(len(vecs) - 1):
        n1, n2 = np.linalg.norm(vecs[i]), np.linalg.norm(vecs[i + 1])
        if n1 > 1e-12 and n2 > 1e-12:
            cos_a = np.clip(np.dot(vecs[i], vecs[i + 1]) / (n1 * n2), -1, 1)
            total_turn += np.degrees(np.arccos(cos_a))

    theta = np.deg2rad(angle)
    radius = arc_len / theta

    spacing_str = f"{dists.min():.3f}–{dists.max():.3f} (σ={dists.std():.4f})"
    print(f"arc_{angle:>3d}  {len(nodes):>5d}  {arc_len:>8.1f}  {radius:>8.1f}  {total_turn:>9.1f}°  {spacing_str}")

print()
print("=" * 75)
print("NODE SPACING CHECK — ALL CATEGORIES")
print(f"{'category':>14}  {'nodes':>5}  {'min_dist':>8}  {'max_dist':>8}  {'mean_dist':>9}  {'std_dist':>8}")
print("-" * 65)

test_shapes = [
    ("straight",    generate_straight(rng=np.random.default_rng(0))),
    ("arc_10",      generate_arc(angle_deg=10,  rng=np.random.default_rng(0))),
    ("arc_90",      generate_arc(angle_deg=90,  rng=np.random.default_rng(0))),
    ("arc_170",     generate_arc(angle_deg=170, rng=np.random.default_rng(0))),
    ("corner",      generate_corner(rng=np.random.default_rng(0))),
    ("junction_T",  generate_junction_T(rng=np.random.default_rng(0))),
    ("junction_Y",  generate_junction_Y(rng=np.random.default_rng(0))),
]

for name, (nodes, edges) in test_shapes:
    # compute distance for every edge
    edge_dists = np.array([np.linalg.norm(nodes[i] - nodes[j]) for i, j in edges])
    print(f"{name:>14}  {len(nodes):>5d}  {edge_dists.min():>8.4f}  {edge_dists.max():>8.4f}  {edge_dists.mean():>9.4f}  {edge_dists.std():>8.4f}")

print()
print("✓ spacing std should be ~0 (uniform) for arcs and straights.")
print("  Junctions may have slight variation due to arm-length randomisation.")
