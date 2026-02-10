#!/usr/bin/env python3
"""
generate_pipe_network.py
────────────────────────
Generate 5 different realistic 3D pipe network meshes (.msh) for testing.

Each pipe is geometrically correct: arcs are tangent-continuous with
adjacent straights, so fluid could actually flow through them.

Pipe types:
  1. simple_bend     — Single 90° bend (L-shape)
  2. s_curve         — S-shaped double bend (up then down)
  3. u_bend          — 180° U-turn
  4. complex_network — Multi-bend network with 6 arcs
  5. t_junction      — T-junction with branching
"""

import json
import math
import sys
import numpy as np
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
# CENTERLINE PRIMITIVES (tangent-continuous)
# ══════════════════════════════════════════════════════════════════════

def make_straight(start, direction, length, n_pts=20):
    """Straight segment from start along direction.
    The last point is snapped to start + length*direction exactly
    to avoid any floating-point accumulation."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    s = np.asarray(start, dtype=np.float64)
    t = np.linspace(0, length, n_pts)
    pts = s + np.outer(t, d)
    # Snap last point to exact analytical end
    pts[-1] = s + length * d
    return pts


def make_arc_tangent(start, tangent, toward, angle_deg, radius, n_pts=40):
    """
    Create a tangent-continuous arc that curves TOWARD a given direction.

    Geometry:
        - The pipe arrives at `start` going in direction `tangent`
        - It needs to curve toward the `toward` direction
        - The arc center is placed so the pipe bends smoothly inward
          (like your whiteboard "correct" drawing — tight inner curve)

    The center of the arc is at:  start + radius * toward_perp
    where toward_perp is the component of `toward` perpendicular to `tangent`,
    normalized. This ensures the arc curves toward the desired side.

    Parameters:
        start:      starting point (3D)
        tangent:    incoming flow direction (unit vector)
        toward:     direction to curve toward (e.g. [0,1,0] = curve toward +Y)
                    Only the component perpendicular to tangent is used.
        angle_deg:  sweep angle in degrees
        radius:     bend radius
        n_pts:      number of points
    """
    t = np.asarray(tangent, dtype=np.float64)
    t = t / np.linalg.norm(t)
    tw = np.asarray(toward, dtype=np.float64)
    s = np.asarray(start, dtype=np.float64)

    # Get the component of `toward` perpendicular to tangent
    # This is the direction from start toward the arc center
    toward_perp = tw - np.dot(tw, t) * t
    tp_norm = np.linalg.norm(toward_perp)
    if tp_norm < 1e-12:
        # toward is parallel to tangent — pick arbitrary perpendicular
        if abs(t[0]) < 0.9:
            toward_perp = np.cross(t, [1, 0, 0])
        else:
            toward_perp = np.cross(t, [0, 1, 0])
        toward_perp /= np.linalg.norm(toward_perp)
    else:
        toward_perp /= tp_norm

    # Center is on the `toward` side: pipe curves toward the center
    center = s + radius * toward_perp

    # Rotation axis: ensures the pipe starts moving in the tangent direction
    # at angle=0. Derived from: d/da(center + R(a)*r_vec)|_{a=0} ∝ tangent
    ax = np.cross(t, toward_perp)
    ax_norm = np.linalg.norm(ax)
    if ax_norm < 1e-12:
        ax = np.array([0, 0, 1.0])
    else:
        ax /= ax_norm

    # Radius vector from center to start
    r_vec = s - center  # = -radius * toward_perp

    # Rotate around the axis
    angles = np.linspace(0, math.radians(angle_deg), n_pts)
    pts = []
    for a in angles:
        cos_a, sin_a = math.cos(a), math.sin(a)
        rot = (r_vec * cos_a
               + np.cross(ax, r_vec) * sin_a
               + ax * np.dot(ax, r_vec) * (1 - cos_a))
        pts.append(center + rot)

    # Snap the last point to exact analytical position to avoid floating-point drift
    # (especially important for 180° where sin(π) ≈ 1.2e-16 instead of 0)
    a_final = math.radians(angle_deg)
    cos_f, sin_f = math.cos(a_final), math.sin(a_final)
    exact_end = center + (r_vec * cos_f
                          + np.cross(ax, r_vec) * sin_f
                          + ax * np.dot(ax, r_vec) * (1 - cos_f))
    # For common angles, snap sin/cos to exact values
    if abs(angle_deg - 180.0) < 1e-6:
        exact_end = center - r_vec  # cos(180)=-1, sin(180)=0
    elif abs(angle_deg - 90.0) < 1e-6:
        exact_end = center + np.cross(ax, r_vec)  # cos(90)=0, sin(90)=1
    elif abs(angle_deg - 270.0) < 1e-6:
        exact_end = center - np.cross(ax, r_vec)  # cos(270)=0, sin(270)=-1
    pts[-1] = exact_end

    return np.array(pts), ax


def end_tangent_after_arc(tangent, turn_axis, angle_deg):
    """Compute the tangent direction at the end of an arc.
    Snaps common angles (90, 180, 270) to exact trig values to avoid
    floating-point drift that accumulates in subsequent straight segments.
    """
    t = np.asarray(tangent, dtype=np.float64)
    t = t / np.linalg.norm(t)
    ax = np.asarray(turn_axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)

    # Snap trig values for common angles
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    if abs(angle_deg - 90.0) < 1e-6:
        cos_a, sin_a = 0.0, 1.0
    elif abs(angle_deg - 180.0) < 1e-6:
        cos_a, sin_a = -1.0, 0.0
    elif abs(angle_deg - 270.0) < 1e-6:
        cos_a, sin_a = 0.0, -1.0
    elif abs(angle_deg - 360.0) < 1e-6:
        cos_a, sin_a = 1.0, 0.0

    rot_t = (t * cos_a
             + np.cross(ax, t) * sin_a
             + ax * np.dot(ax, t) * (1 - cos_a))
    n = np.linalg.norm(rot_t)
    return rot_t / n if n > 1e-12 else t


# ══════════════════════════════════════════════════════════════════════
# ROTATION-MINIMIZING FRAME (RMF)
# ══════════════════════════════════════════════════════════════════════

def _safe_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _make_perpendicular(tangent):
    t = _safe_normalize(tangent)
    if abs(t[0]) < 0.9:
        perp = np.cross(t, [1, 0, 0])
    else:
        perp = np.cross(t, [0, 1, 0])
    return _safe_normalize(perp)


def compute_rmf(points):
    n = len(points)
    if n < 2:
        return (np.tile([1, 0, 0], (n, 1)).astype(float),
                np.tile([0, 1, 0], (n, 1)).astype(float),
                np.tile([0, 0, 1], (n, 1)).astype(float))

    T = np.zeros((n, 3))
    N = np.zeros((n, 3))
    B = np.zeros((n, 3))

    for i in range(n):
        if i == 0:
            T[i] = points[min(1, n - 1)] - points[0]
        elif i == n - 1:
            T[i] = points[-1] - points[max(0, n - 2)]
        else:
            T[i] = points[i + 1] - points[i - 1]
        T[i] = _safe_normalize(T[i])

    N[0] = _make_perpendicular(T[0])
    B[0] = _safe_normalize(np.cross(T[0], N[0]))

    for i in range(n - 1):
        v1 = points[i + 1] - points[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-20:
            N[i + 1], B[i + 1] = N[i].copy(), B[i].copy()
            continue
        rL = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = np.dot(v2, v2)
        if c2 < 1e-20:
            N[i + 1] = rL
        else:
            N[i + 1] = rL - (2.0 / c2) * np.dot(v2, rL) * v2
        nn = np.linalg.norm(N[i + 1])
        if nn < 1e-12:
            N[i + 1] = N[i].copy()
        else:
            N[i + 1] /= nn
        B[i + 1] = np.cross(T[i + 1], N[i + 1])
        bn = np.linalg.norm(B[i + 1])
        if bn < 1e-12:
            B[i + 1] = B[i].copy()
        else:
            B[i + 1] /= bn

    for i in range(n):
        if np.any(np.isnan(N[i])) or np.linalg.norm(N[i]) < 1e-12:
            N[i] = _make_perpendicular(T[i])
        if np.any(np.isnan(B[i])) or np.linalg.norm(B[i]) < 1e-12:
            B[i] = _safe_normalize(np.cross(T[i], N[i]))

    return T, N, B


# ══════════════════════════════════════════════════════════════════════
# TANGENT-CONTINUOUS PIPE BUILDER
# ══════════════════════════════════════════════════════════════════════

class PipeBuilder:
    """
    Build centerlines with guaranteed tangent continuity.
    Each arc starts tangent to the previous segment's end direction.
    """

    def __init__(self, start=None):
        self.points = []
        self.segments = []
        self._tangent = np.array([1, 0, 0], dtype=np.float64)
        if start is not None:
            self.points.append(np.asarray(start, dtype=np.float64))

    def _append_pts(self, pts, seg_type, angle=0.0):
        """Append points to the centerline. Does NOT update self._tangent —
        callers (straight / bend) set the tangent explicitly to avoid
        floating-point drift from numerical differentiation."""
        start_idx = len(self.points)
        arr = np.asarray(pts, dtype=np.float64)
        # Skip first point if it overlaps with current end
        if self.points and np.linalg.norm(arr[0] - self.points[-1]) < 1e-6:
            arr = arr[1:]
        for p in arr:
            self.points.append(p.copy())
        end_idx = len(self.points) - 1
        if end_idx > start_idx:
            self.segments.append({"type": seg_type, "start_idx": start_idx,
                                  "end_idx": end_idx, "angle_deg": angle})

    @property
    def end(self):
        return np.array(self.points[-1]) if self.points else np.zeros(3)

    @property
    def tangent(self):
        return self._tangent.copy()

    def straight(self, length, n=None):
        """Add a straight segment continuing in the current tangent direction.
        The tangent is preserved exactly (not recomputed from points)."""
        n = n or max(10, int(length * 2))
        # Normalize tangent to ensure exact unit vector
        direction = self._tangent / np.linalg.norm(self._tangent)
        pts = make_straight(self.end, direction, length, n)
        self._append_pts(pts, "straight")
        # Tangent stays exactly the same — straight doesn't change direction
        # (no numerical recomputation that could introduce drift)
        self._tangent = direction.copy()
        return self

    def bend(self, toward, angle_deg, radius=5.0, n=None):
        """
        Add a tangent-continuous arc that curves TOWARD a given direction.

        toward:    direction to curve toward (e.g. [0,1,0] = curve toward +Y,
                   [0,0,1] = curve upward, [-1,0,0] = curve toward -X)
                   Only the perpendicular component to the current tangent matters.
        angle_deg: sweep angle in degrees (90 = quarter turn, 180 = U-bend)
        """
        n = n or max(20, int(abs(angle_deg) * 0.5))
        # Use the exact current tangent for the arc computation
        incoming_tangent = self._tangent / np.linalg.norm(self._tangent)
        pts, actual_axis = make_arc_tangent(self.end, incoming_tangent, toward, angle_deg, radius, n)
        self._append_pts(pts, "arc", abs(angle_deg))
        # Set tangent analytically — never from numerical point differences
        self._tangent = end_tangent_after_arc(incoming_tangent, actual_axis, angle_deg)
        return self

    def build(self):
        pts = np.array(self.points)
        edges = [(i, i + 1) for i in range(len(pts) - 1)]
        return pts, edges, self.segments


# ══════════════════════════════════════════════════════════════════════
# 5 PIPE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

def build_simple_bend():
    """
    Pipe 1: Simple 90° L-bend.
    Goes along +X, curves toward +Y (left turn when viewed from above).
    """
    b = PipeBuilder(start=[0, 0, 0])
    b.straight(25)
    b.bend(toward=[0, 1, 0], angle_deg=90, radius=5)
    b.straight(25)
    return b.build()


def build_s_curve():
    """
    Pipe 2: S-curve.
    Goes along +X, curves up to +Z, short vertical run,
    curves back to +X at higher elevation.
    Both bends are 90° with the same radius for symmetry.
    """
    R = 5.0
    b = PipeBuilder(start=[0, 0, 0])
    b.straight(20)
    # Bend 1: curve upward — center goes to +Z side, pipe arcs from +X to +Z
    b.bend(toward=[0, 0, 1], angle_deg=90, radius=R)
    b.straight(15)
    # Bend 2: pipe is going +Z, curve it back toward horizontal
    # "toward" = the direction we want the center to be = where the pipe curves toward
    # Going +Z, to end up going +X, the center must be in the +X direction
    # The pipe will arc from +Z around to +X
    b.bend(toward=[1, 0, 0], angle_deg=90, radius=R)
    b.straight(20)
    return b.build()


def build_u_bend():
    """
    Pipe 3: 180° U-turn.
    Goes along +X, makes a 180° U-bend, comes back along -X.
    The U-bend curves toward +Y (semicircle in the XY plane).

    Layout:
        inlet:  y=0, going +X
        arc:    semicircle in XY plane, center at (25, 8, 0)
        outlet: y=16, going -X  (offset = 2*radius = 16)
    """
    b = PipeBuilder(start=[0, 0, 0])
    b.straight(25)
    b.bend(toward=[0, 1, 0], angle_deg=180, radius=8)
    b.straight(25)
    return b.build()


def build_complex_network():
    """
    Pipe 4: Multi-bend network — 6 arcs of various angles.

    Spatial layout (no collisions):
      Start at origin, go +X for a long run.
      Arc1: 90° turn toward +Y (horizontal turn left).
      Long straight along +Y.
      Arc2: 90° curve upward (+Z).
      Straight going up.
      Arc3: 45° gentle curve toward -X.
      Straight in the new diagonal direction.
      Arc4: 90° curve toward -Y.
      Straight along -Y (going back, but offset in Z so no collision).
      Arc5: 180° U-bend toward -X (reverses to go +Y).
      Straight along +Y.
      Arc6: 90° curve downward (-Z).
      Straight going down, ending well away from start.
    """
    R = 6.0
    b = PipeBuilder(start=[0, 0, 0])

    # Long inlet along +X
    b.straight(30)

    # Arc1: 90° horizontal left turn → now going +Y
    b.bend(toward=[0, 1, 0], angle_deg=90, radius=R)
    b.straight(30)

    # Arc2: 90° curve upward → now going +Z
    b.bend(toward=[0, 0, 1], angle_deg=90, radius=R)
    b.straight(30)

    # Arc3: 45° gentle curve toward +X → deflects diagonally
    b.bend(toward=[1, 0, 0], angle_deg=45, radius=R * 1.5)
    b.straight(30)

    # Arc4: 90° curve toward -Y → turns the pipe toward -Y
    b.bend(toward=[0, -1, 0], angle_deg=90, radius=R)
    b.straight(35)

    # Arc5: 180° U-bend toward -Z → reverses direction (offset in Z)
    b.bend(toward=[0, 0, -1], angle_deg=180, radius=R)
    b.straight(30)

    # Arc6: 90° curve toward +X → pipe exits toward +X
    b.bend(toward=[1, 0, 0], angle_deg=90, radius=R)
    b.straight(25)

    return b.build()


def build_t_junction():
    """
    Pipe 5: T-junction.
    Main pipe goes along +X with a 90° bend, then continues.
    A branch splits off upward (+Z) at the junction point.
    The branch curves smoothly upward — no pipe intersection.
    """
    R = 5.0

    # === Main pipe ===
    b = PipeBuilder(start=[0, 0, 0])
    b.straight(35)

    # Mark junction BEFORE the bend (at the straight section)
    junction_idx = len(b.points) - 1
    junction_pt = b.end.copy()
    junction_tangent = b.tangent.copy()

    # Main pipe continues straight (no bend at junction — clean T)
    b.straight(35)

    # === Branch: from junction, curve upward ===
    branch_start = len(b.points)
    b.points.append(junction_pt.copy())

    # The branch curves toward +Z (upward) from the junction
    branch_pts, branch_axis = make_arc_tangent(
        junction_pt, junction_tangent,
        toward=[0, 0, 1],  # curve upward
        angle_deg=90, radius=R, n_pts=40
    )
    b._append_pts(branch_pts, "arc", 90)
    b._tangent = end_tangent_after_arc(junction_tangent, branch_axis, 90)
    b.straight(20)

    # Build edges
    pts = np.array(b.points)
    edges = []
    for i in range(len(pts) - 1):
        if i == branch_start - 1:
            continue
        edges.append((i, i + 1))
    edges.append((junction_idx, branch_start))

    return pts, edges, b.segments


# ══════════════════════════════════════════════════════════════════════
# MESH GENERATION WITH RMF
# ══════════════════════════════════════════════════════════════════════

def generate_pipe_mesh(cl_points, cl_edges, pipe_radius=1.5, n_circ=24):
    n_cl = len(cl_points)
    adj = {}
    for u, v in cl_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    degree = {}
    for u, v in cl_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1

    visited = set()
    chains = []
    starts = [n for n in range(n_cl) if degree.get(n, 0) == 1]
    if not starts:
        starts = [0]

    def trace_chain(start):
        chain = [start]
        visited.add(start)
        current = start
        while True:
            neighbors = [n for n in adj.get(current, []) if n not in visited]
            if not neighbors:
                break
            nxt = neighbors[0]
            chain.append(nxt)
            visited.add(nxt)
            if degree.get(nxt, 0) > 2:
                break
            current = nxt
        return chain

    for s in starts:
        if s in visited:
            continue
        chain = trace_chain(s)
        chains.append(chain)
        last = chain[-1]
        if degree.get(last, 0) > 2:
            for nb in adj.get(last, []):
                if nb not in visited:
                    sub = trace_chain(nb)
                    chains.append([last] + sub)

    for i in range(n_cl):
        if i not in visited:
            chain = trace_chain(i)
            if len(chain) > 1:
                chains.append(chain)

    all_vertices = []
    all_faces = []
    cl_to_mesh = {i: [] for i in range(n_cl)}

    for chain in chains:
        if len(chain) < 2:
            continue
        chain_pts = cl_points[chain]
        T, N, B = compute_rmf(chain_pts)

        ring_starts = []
        for ci, cl_idx in enumerate(chain):
            pt = chain_pts[ci]
            n_vec, b_vec = N[ci], B[ci]
            rs = len(all_vertices)
            ring_starts.append(rs)
            ring = []
            for j in range(n_circ):
                angle = 2.0 * math.pi * j / n_circ
                offset = pipe_radius * (math.cos(angle) * n_vec + math.sin(angle) * b_vec)
                all_vertices.append(pt + offset)
                ring.append(rs + j)
            cl_to_mesh[cl_idx].extend(ring)

        for ci in range(len(chain) - 1):
            r0 = ring_starts[ci]
            r1 = ring_starts[ci + 1]
            for j in range(n_circ):
                jn = (j + 1) % n_circ
                all_faces.append([r0 + j, r1 + j, r1 + jn])
                all_faces.append([r0 + j, r1 + jn, r0 + jn])

    vertices = np.array(all_vertices, dtype=np.float64)
    faces = np.array(all_faces, dtype=np.int64)

    bad = ~np.isfinite(vertices)
    if bad.any():
        print(f"  WARNING: {bad.any(axis=1).sum()} NaN/Inf vertices replaced with 0")
        vertices[bad] = 0.0

    return vertices, faces, cl_to_mesh


# ══════════════════════════════════════════════════════════════════════
# MSH WRITER
# ══════════════════════════════════════════════════════════════════════

def write_msh(filepath, vertices, faces):
    nv, nf = len(vertices), len(faces)
    with open(filepath, 'w') as f:
        f.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
        f.write("$Entities\n0 0 1 0\n")
        f.write(f"1 {vertices[:,0].min():.6f} {vertices[:,1].min():.6f} {vertices[:,2].min():.6f} "
                f"{vertices[:,0].max():.6f} {vertices[:,1].max():.6f} {vertices[:,2].max():.6f} 0 0\n")
        f.write("$EndEntities\n$Nodes\n")
        f.write(f"1 {nv} 1 {nv}\n2 1 0 {nv}\n")
        for i in range(nv):
            f.write(f"{i+1}\n")
        for i in range(nv):
            f.write(f"{vertices[i,0]:.8f} {vertices[i,1]:.8f} {vertices[i,2]:.8f}\n")
        f.write("$EndNodes\n$Elements\n")
        f.write(f"1 {nf} 1 {nf}\n2 1 2 {nf}\n")
        for i in range(nf):
            f.write(f"{i+1} {faces[i,0]+1} {faces[i,1]+1} {faces[i,2]+1}\n")
        f.write("$EndElements\n")
    print(f"  Wrote {filepath}: {nv} verts, {nf} tris")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

PIPE_CONFIGS = {
    "simple_bend":     {"builder": build_simple_bend,     "radius": 1.5, "desc": "Simple 90° L-bend"},
    "s_curve":         {"builder": build_s_curve,          "radius": 1.5, "desc": "S-shaped double bend"},
    "u_bend":          {"builder": build_u_bend,           "radius": 2.0, "desc": "180° U-turn"},
    "complex_network": {"builder": build_complex_network,  "radius": 1.5, "desc": "Multi-bend network (6 arcs)"},
    "t_junction":      {"builder": build_t_junction,       "radius": 1.5, "desc": "T-junction with branching"},
}


def generate_one(name, config, out_dir):
    print(f"\n{'='*60}")
    print(f"Pipe: {name} — {config['desc']}")
    print(f"{'='*60}")
    cl_pts, cl_edges, seg_info = config["builder"]()
    print(f"  Centerline: {len(cl_pts)} pts, {len(cl_edges)} edges, {len(seg_info)} segments")
    verts, faces, cl_map = generate_pipe_mesh(cl_pts, cl_edges, pipe_radius=config["radius"], n_circ=24)
    print(f"  Mesh: {len(verts)} verts, {len(faces)} tris")
    write_msh(str(out_dir / f"{name}.msh"), verts, faces)
    with open(out_dir / f"{name}_centerline.json", 'w') as f:
        json.dump({"nodes": cl_pts.tolist(),
                    "edges": [[int(u), int(v)] for u, v in cl_edges],
                    "segments": seg_info}, f)
    return cl_pts, cl_edges, verts, faces


def main():
    out_dir = Path(__file__).parent
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name and name in PIPE_CONFIGS:
        generate_one(name, PIPE_CONFIGS[name], out_dir)
    else:
        for n, cfg in PIPE_CONFIGS.items():
            generate_one(n, cfg, out_dir)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
