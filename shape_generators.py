"""
shape_generators.py
───────────────────
Pure-geometry factories for every shape motif.
Each public function returns (nodes, edges) where
  nodes : np.ndarray of shape (N, 3)
  edges : list[tuple[int, int]]
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def _sequential_edges(n: int) -> List[Tuple[int, int]]:
    """Chain 0-1-2-…-(n-1)."""
    return [(i, i + 1) for i in range(n - 1)]


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3x3 rotation (QR decomposition trick)."""
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))       # ensure det +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _apply_random_rigid(nodes: np.ndarray,
                        rng: np.random.Generator,
                        translate: bool = True) -> np.ndarray:
    """Apply a random SO(3) rotation + optional translation."""
    R = _random_rotation_matrix(rng)
    nodes = nodes @ R.T
    if translate:
        t = rng.uniform(-5, 5, size=3)
        nodes += t
    return nodes


# ──────────────────────────────────────────────
# 1. STRAIGHT
# ──────────────────────────────────────────────

def generate_straight(length: float | None = None,
                      rng: np.random.Generator | None = None
                      ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Straight segment of *length* units with ~1-unit node spacing.
    If *length* is None, sample uniformly from [5, 20].
    """
    rng = rng or np.random.default_rng()
    if length is None:
        length = rng.uniform(5.0, 20.0)

    n_pts = max(int(round(length)) + 1, 2)
    t = np.linspace(0, length, n_pts)
    nodes = np.column_stack([t, np.zeros(n_pts), np.zeros(n_pts)])
    nodes = _apply_random_rigid(nodes, rng)
    return nodes, _sequential_edges(n_pts)


# ──────────────────────────────────────────────
# 2. ARC
# ──────────────────────────────────────────────

def generate_arc(angle_deg: float = 90.0,
                 arc_length: float | None = None,
                 rng: np.random.Generator | None = None
                 ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Circular arc whose **total angular deviation** equals *angle_deg*.

    The radius is derived:  R = arc_length / theta.

    Small angles → large radius → gentle curve (almost straight).
    Large angles → small radius → tight curve.
    """
    rng = rng or np.random.default_rng()
    if arc_length is None:
        arc_length = rng.uniform(10.0, 20.0)

    theta = np.deg2rad(angle_deg)
    radius = arc_length / theta            # R derived from desired angle

    n_pts = max(int(round(arc_length)) + 1, 2)   # ~1-unit node spacing
    angles = np.linspace(0, theta, n_pts)

    # optionally flip direction (clockwise / counter-clockwise)
    if rng.random() < 0.5:
        angles = -angles

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(n_pts)
    nodes = np.column_stack([x, y, z])

    # center at arc midpoint for nicer transforms
    nodes -= nodes[len(nodes) // 2]
    nodes = _apply_random_rigid(nodes, rng)
    return nodes, _sequential_edges(n_pts)


# ──────────────────────────────────────────────
# 3. CORNER (sharp 90° turn by default)
# ──────────────────────────────────────────────

def generate_corner(angle_deg: float = 90.0,
                    arm_length: float | None = None,
                    rng: np.random.Generator | None = None
                    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Two straight arms meeting at a sharp angle.
    """
    rng = rng or np.random.default_rng()
    if arm_length is None:
        arm_length = rng.uniform(5.0, 15.0)

    n_arm = max(int(round(arm_length)) + 1, 2)
    t1 = np.linspace(0, arm_length, n_arm)

    # first arm along +X
    arm1 = np.column_stack([t1, np.zeros(n_arm), np.zeros(n_arm)])

    # second arm starts at end of arm1, heading at *angle_deg*
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)
    t2 = np.linspace(0, arm_length, n_arm)[1:]          # skip shared vertex
    arm2_x = arm_length + t2 * dx
    arm2_y = t2 * dy
    arm2 = np.column_stack([arm2_x, arm2_y, np.zeros(len(t2))])

    nodes = np.vstack([arm1, arm2])
    nodes = _apply_random_rigid(nodes, rng)
    return nodes, _sequential_edges(len(nodes))


# ──────────────────────────────────────────────
# 4. JUNCTION – T
# ──────────────────────────────────────────────

def generate_junction_T(arm_length: float | None = None,
                        rng: np.random.Generator | None = None
                        ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    T-junction: one straight through-pipe along X
    plus a perpendicular branch at the midpoint going along +Y.
    The junction node is shared by all three arms.
    """
    rng = rng or np.random.default_rng()
    if arm_length is None:
        arm_length = rng.uniform(5.0, 12.0)

    n_arm = max(int(round(arm_length)) + 1, 2)
    half = arm_length / 2.0

    # main spine along X  (from -half to +half)
    t_main = np.linspace(-half, half, 2 * n_arm - 1)
    spine = np.column_stack([t_main,
                             np.zeros(len(t_main)),
                             np.zeros(len(t_main))])

    # branch along +Y starting at midpoint (index n_arm-1)
    junction_idx = n_arm - 1
    t_branch = np.linspace(0, arm_length, n_arm)[1:]   # skip shared node
    branch = np.column_stack([np.zeros(len(t_branch)),
                              t_branch,
                              np.zeros(len(t_branch))])

    nodes = np.vstack([spine, branch])

    # edges: sequential spine + branch connects to junction node
    edges = _sequential_edges(len(spine))
    branch_start = len(spine)
    # junction → first branch node
    edges.append((junction_idx, branch_start))
    for i in range(branch_start, branch_start + len(branch) - 1):
        edges.append((i, i + 1))

    nodes = _apply_random_rigid(nodes, rng)
    return nodes, edges


# ──────────────────────────────────────────────
# 5. JUNCTION – Y
# ──────────────────────────────────────────────

def generate_junction_Y(splay_deg: float | None = None,
                        arm_length: float | None = None,
                        rng: np.random.Generator | None = None
                        ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Y-junction: a single trunk along −X leading to a shared node at the
    origin, then two arms splaying symmetrically at ±splay_deg/2.
    """
    rng = rng or np.random.default_rng()
    if arm_length is None:
        arm_length = rng.uniform(5.0, 12.0)
    if splay_deg is None:
        splay_deg = rng.uniform(30.0, 90.0)

    n_arm = max(int(round(arm_length)) + 1, 2)

    # trunk along −X into origin
    t_trunk = np.linspace(-arm_length, 0, n_arm)
    trunk = np.column_stack([t_trunk,
                             np.zeros(n_arm),
                             np.zeros(n_arm)])

    junction_idx = n_arm - 1          # origin node

    half_splay = np.deg2rad(splay_deg / 2.0)

    def _make_arm(signed_angle: float) -> np.ndarray:
        t = np.linspace(0, arm_length, n_arm)[1:]
        x = t * np.cos(signed_angle)
        y = t * np.sin(signed_angle)
        return np.column_stack([x, y, np.zeros(len(t))])

    arm_a = _make_arm(half_splay)
    arm_b = _make_arm(-half_splay)

    nodes = np.vstack([trunk, arm_a, arm_b])

    # edges ---
    edges = _sequential_edges(n_arm)                    # trunk chain

    arm_a_start = n_arm
    edges.append((junction_idx, arm_a_start))
    for i in range(arm_a_start, arm_a_start + len(arm_a) - 1):
        edges.append((i, i + 1))

    arm_b_start = arm_a_start + len(arm_a)
    edges.append((junction_idx, arm_b_start))
    for i in range(arm_b_start, arm_b_start + len(arm_b) - 1):
        edges.append((i, i + 1))

    nodes = _apply_random_rigid(nodes, rng)
    return nodes, edges
