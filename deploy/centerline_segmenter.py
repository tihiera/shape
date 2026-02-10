#!/usr/bin/env python3
"""
centerline_segmenter.py
───────────────────────
Split a 3D centerline graph (nodes + edges) into meaningful subgraphs:
junction, straight, arc.

Public API
──────────
    load_graph(path)                -> (nodes, edges)
    split_centerline_graph(...)     -> list[SegmentRecord]

All helpers are importable for testing / custom pipelines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Branch:
    """Ordered polyline path between stop-nodes (junctions / endpoints)."""
    node_ids: List[int]          # indices into the original node array
    start_type: str              # "junction" | "endpoint"
    end_type: str                # "junction" | "endpoint"


@dataclass
class SegmentRecord:
    segment_id: int
    type: str                    # "junction" | "straight" | "arc" | "corner"
    node_ids_original: List[int]
    length: float
    mean_curvature: float
    max_curvature: float
    arc_angle_deg: float = 0.0
    radius_est: float = 0.0
    corner_angle_deg: float = 0.0


@dataclass
class SegmentParams:
    """Tunable parameters for curvature-based segmentation."""
    target_step: float = 1.0     # resample spacing
    k_low: float = -1.0          # straight threshold (< 0 = auto)
    k_high: float = -1.0         # arc threshold (< 0 = auto)
    min_run_points: int = 6      # minimum segment length in points
    cooldown_points: int = 4     # kappa must stay below k_low for this many points
    snap_window: int = 3         # snap boundary to nearest kappa minimum within ± this
    smooth_window: int = 3       # curvature smoothing kernel size
    junction_k_hops: int = 2     # hops around junction node to include
    corner_max_length: float = 3.0   # max arc-length for a corner (short sharp turn)
    corner_min_angle: float = 20.0   # min turning angle (deg) to qualify as corner
    corner_sharpness: float = 5.0    # min ratio of max_curvature / mean_curvature for corner


# ═══════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════

def load_graph(json_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Load nodes (N,3) and edges from a JSON file."""
    obj = json.loads(Path(json_path).read_text())
    nodes = np.asarray(obj["nodes"], dtype=np.float64)
    edges = [(int(e[0]), int(e[1])) for e in obj["edges"]]
    return nodes, edges


# ═══════════════════════════════════════════════════════════════════════
# GRAPH TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════

def build_adjacency(edges: List[Tuple[int, int]],
                    n_nodes: int) -> Dict[int, Set[int]]:
    """Undirected adjacency as {node: set(neighbours)}."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def detect_junction_nodes(adj: Dict[int, Set[int]]) -> Set[int]:
    """Nodes with degree >= 3."""
    return {n for n, nbrs in adj.items() if len(nbrs) >= 3}


def _node_type(node: int, adj: Dict[int, Set[int]],
               junctions: Set[int]) -> str:
    if node in junctions:
        return "junction"
    if len(adj[node]) <= 1:
        return "endpoint"
    return "regular"


# ═══════════════════════════════════════════════════════════════════════
# BRANCH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def extract_branches(adj: Dict[int, Set[int]],
                     junction_nodes: Set[int]) -> List[Branch]:
    """
    Walk degree-2 chains between stop-nodes (junctions / endpoints).
    Each branch is an ordered list of node ids.
    """
    stop_nodes = junction_nodes | {n for n, nbrs in adj.items() if len(nbrs) <= 1}
    visited_edges: Set[Tuple[int, int]] = set()
    branches: List[Branch] = []

    def _walk(start: int, first_step: int) -> List[int]:
        """Walk from start through first_step along degree-2 chain."""
        path = [start, first_step]
        visited_edges.add((start, first_step))
        visited_edges.add((first_step, start))
        cur = first_step
        prev = start
        while cur not in stop_nodes:
            nxt = [n for n in adj[cur] if n != prev]
            if len(nxt) != 1:
                break
            prev = cur
            cur = nxt[0]
            visited_edges.add((prev, cur))
            visited_edges.add((cur, prev))
            path.append(cur)
        return path

    # start walks from every stop-node
    for start in sorted(stop_nodes):
        for nbr in sorted(adj[start]):
            edge = (start, nbr)
            if edge in visited_edges:
                continue
            path = _walk(start, nbr)
            branches.append(Branch(
                node_ids=path,
                start_type=_node_type(path[0], adj, junction_nodes),
                end_type=_node_type(path[-1], adj, junction_nodes),
            ))

    return branches


# ═══════════════════════════════════════════════════════════════════════
# RESAMPLING
# ═══════════════════════════════════════════════════════════════════════

def _polyline_arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a polyline (N,3) -> (N,)."""
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def resample_polyline(points: np.ndarray,
                      target_step: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a 3D polyline so consecutive points are ~target_step apart.
    Endpoints are kept fixed.

    Returns
    ───────
    resampled : (M, 3)
    arc_params : (M,)  parameter t in [0, total_length] for each resampled point
    """
    cum = _polyline_arc_lengths(points)
    total = cum[-1]
    if total < 1e-12:
        return points.copy(), cum.copy()

    n_seg = max(1, int(round(total / target_step)))
    t_new = np.linspace(0.0, total, n_seg + 1)

    resampled = np.zeros((len(t_new), 3), dtype=np.float64)
    resampled[0] = points[0]
    resampled[-1] = points[-1]

    j = 0
    for i in range(1, len(t_new) - 1):
        t = t_new[i]
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        if seg_len < 1e-12:
            resampled[i] = points[j]
        else:
            frac = (t - cum[j]) / seg_len
            resampled[i] = points[j] + frac * (points[j + 1] - points[j])

    return resampled, t_new


# ═══════════════════════════════════════════════════════════════════════
# CURVATURE
# ═══════════════════════════════════════════════════════════════════════

def menger_curvature(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Menger curvature = 4*A / (a*b*c) = 1/R.
    A = triangle area from cross product.
    """
    v01 = p1 - p0
    v02 = p2 - p0
    area2 = float(np.linalg.norm(np.cross(v01, v02)))  # = 2*A

    a = float(np.linalg.norm(p1 - p2))
    b = float(np.linalg.norm(p0 - p2))
    c = float(np.linalg.norm(p0 - p1))
    denom = a * b * c

    if denom < 1e-15:
        return 0.0
    return (2.0 * area2) / denom  # = 4A / (abc)


def compute_branch_curvature(points: np.ndarray,
                             smooth_window: int = 3) -> np.ndarray:
    """
    Per-point curvature via Menger formula on (i-1, i, i+1).
    Endpoints get 0. Optional moving-average smoothing.
    """
    n = len(points)
    kappa = np.zeros(n, dtype=np.float64)

    for i in range(1, n - 1):
        kappa[i] = menger_curvature(points[i - 1], points[i], points[i + 1])

    # smooth
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # pad to keep length
        padded = np.pad(kappa, (smooth_window // 2, smooth_window // 2), mode="edge")
        kappa = np.convolve(padded, kernel, mode="valid")[:n]
        kappa[0] = 0.0
        kappa[-1] = 0.0

    return kappa


# ═══════════════════════════════════════════════════════════════════════
# COLLINEARITY TEST (cosine-based, robust to dense graphs)
# ═══════════════════════════════════════════════════════════════════════

def _cosine_deviation(p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray) -> float:
    """
    Compute 1 - |cos(angle between vectors AB and AC)|.
    Returns 0.0 if A, B, C are perfectly collinear, >0 otherwise.
    """
    ab = p_b - p_a
    ac = p_c - p_a
    nab = np.linalg.norm(ab)
    nac = np.linalg.norm(ac)
    if nab < 1e-15 or nac < 1e-15:
        return 0.0
    cos_val = np.dot(ab, ac) / (nab * nac)
    return 1.0 - abs(float(cos_val))


def compute_collinearity_mask(points: np.ndarray,
                              collinear_tol: float = 1e-4,
                              lookback: int = 0) -> np.ndarray:
    """
    For each interior point i, test collinearity using the cosine method.

    Multi-scale approach to handle dense graphs:
      - Test (i-1, i, i+1) for local collinearity
      - Also test (i-K, i, i+K) for K in adaptive hops
        This catches arcs that look locally straight due to fine spacing.

    The lookback parameter is auto-computed if 0: we pick K so that
    the hop distance spans ~1 unit of arc length (enough to see curvature).

    Returns a boolean array (N,) where True = point is on a straight run.
    """
    n = len(points)
    is_straight = np.ones(n, dtype=bool)

    if n < 3:
        return is_straight

    # auto-compute lookback from median edge length
    if lookback <= 0:
        edge_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
        med_step = float(np.median(edge_lens)) if len(edge_lens) > 0 else 1.0
        # we want the test span to cover ~1 unit of arc length
        # for R=8, arc of 1 unit subtends 1/8 rad ≈ 7° → clearly not straight
        lookback = max(1, int(round(1.0 / max(med_step, 1e-6))))
        lookback = min(lookback, n // 4)  # don't exceed quarter of branch

    # test at multiple scales: 1, lookback//2, lookback
    hops = sorted(set([1, max(1, lookback // 2), lookback]))

    for i in range(1, n - 1):
        curved = False
        for k in hops:
            lo = max(0, i - k)
            hi = min(n - 1, i + k)
            if lo == hi or lo == i or hi == i:
                continue
            dev = _cosine_deviation(points[lo], points[i], points[hi])
            if dev > collinear_tol:
                curved = True
                break
        if curved:
            is_straight[i] = False

    return is_straight


def detect_straight_runs(is_straight: np.ndarray,
                         min_run: int = 4) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of True in is_straight that are >= min_run long.
    Returns list of (start, end_exclusive) index pairs.
    """
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(is_straight)
    while i < n:
        if is_straight[i]:
            j = i
            while j < n and is_straight[j]:
                j += 1
            if (j - i) >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# ═══════════════════════════════════════════════════════════════════════
# AUTO THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════

def auto_thresholds(kappa: np.ndarray,
                    is_straight: np.ndarray,
                    k_low_frac: float = 0.25,
                    k_high_frac: float = 0.50,
                    k_low_min: float = 0.005,
                    k_high_min: float = 0.02) -> Tuple[float, float]:
    """
    Compute k_low / k_high from the curvature distribution,
    but ONLY from non-straight points (avoids noise floor bias).

    Uses fractions of the median curved-kappa rather than percentiles,
    so that arcs with constant curvature still get a threshold well
    below their kappa value (allowing the hysteresis to trigger).
    """
    # only consider points that the collinearity test flagged as curved
    curved_kappa = kappa[~is_straight]
    curved_kappa = curved_kappa[curved_kappa > 1e-9]

    if len(curved_kappa) < 3:
        return k_low_min, k_high_min

    med_k = float(np.median(curved_kappa))
    k_low = max(k_low_min, med_k * k_low_frac)
    k_high = max(k_high_min, med_k * k_high_frac)

    if k_high <= k_low:
        k_high = k_low * 2.0

    return k_low, k_high


# ═══════════════════════════════════════════════════════════════════════
# CURVATURE-BASED SEGMENTATION (collinearity + hysteresis + no-arc-split)
# ═══════════════════════════════════════════════════════════════════════

def segment_branch_by_curvature(
    points: np.ndarray,
    kappa: np.ndarray,
    params: SegmentParams,
    k_low: float,
    k_high: float,
    collinear_tol: float = 1e-4,
) -> List[Tuple[str, List[int]]]:
    """
    Segment a single resampled branch into STRAIGHT / ARC runs.

    Two-pass approach:
      Pass 0: Collinearity test — definitively mark truly-straight points
              using cosine(AB, AC). This catches dense straight runs that
              Menger curvature misclassifies due to floating-point noise.
      Pass 1: Hysteresis on remaining points using curvature thresholds.

    Returns list of (label, [point_indices_within_branch]).
    """
    n = len(kappa)
    if n < 3:
        return [("straight", list(range(n)))]

    # ── pass 0: collinearity pre-classification ──
    is_straight = compute_collinearity_mask(points, collinear_tol=collinear_tol)

    # force curvature to 0 for collinear points (removes noise floor)
    kappa_clean = kappa.copy()
    kappa_clean[is_straight] = 0.0

    # ── pass 1: hysteresis labelling (on cleaned curvature) ──
    labels = ["straight"] * n
    state = "straight"
    low_streak = 0

    for i in range(n):
        # collinear points are definitively straight
        if is_straight[i]:
            if state == "arc":
                low_streak += 1
                if low_streak >= params.cooldown_points:
                    state = "straight"
                    low_streak = 0
            # label collinear points as "straight" always (not the arc state)
            labels[i] = "straight"
            continue

        if state == "straight":
            if kappa_clean[i] > k_high:
                state = "arc"
                low_streak = 0
        else:  # arc
            if kappa_clean[i] < k_low:
                low_streak += 1
                if low_streak >= params.cooldown_points:
                    state = "straight"
                    low_streak = 0
            else:
                low_streak = 0
        labels[i] = state

    # ── pass 2: find run boundaries ──
    runs: List[Tuple[str, int, int]] = []  # (label, start, end_exclusive)
    run_start = 0
    for i in range(1, n):
        if labels[i] != labels[run_start]:
            runs.append((labels[run_start], run_start, i))
            run_start = i
    runs.append((labels[run_start], run_start, n))

    # ── pass 3: merge tiny runs into neighbours ──
    merged = True
    while merged:
        merged = False
        new_runs = []
        for label, s, e in runs:
            length = e - s
            if length < params.min_run_points and len(new_runs) > 0:
                # merge into previous run
                prev_label, prev_s, prev_e = new_runs[-1]
                new_runs[-1] = (prev_label, prev_s, e)
                merged = True
            else:
                new_runs.append((label, s, e))
        runs = new_runs

    # ── pass 4: snap boundaries to kappa local minima ──
    snapped: List[Tuple[str, List[int]]] = []
    for label, s, e in runs:
        # snap start
        if s > 0:
            win_lo = max(0, s - params.snap_window)
            win_hi = min(n, s + params.snap_window + 1)
            best = win_lo + int(np.argmin(kappa[win_lo:win_hi]))
            s = best
        # snap end
        if e < n:
            win_lo = max(0, e - params.snap_window)
            win_hi = min(n, e + params.snap_window + 1)
            best = win_lo + int(np.argmin(kappa[win_lo:win_hi]))
            e = best

        indices = list(range(max(0, s), min(n, e)))
        if len(indices) > 0:
            snapped.append((label, indices))

    # deduplicate / fill gaps: ensure every point is covered exactly once
    covered = set()
    final: List[Tuple[str, List[int]]] = []
    for label, indices in snapped:
        clean = [i for i in indices if i not in covered]
        if clean:
            final.append((label, clean))
            covered.update(clean)

    # fill any uncovered points (assign to nearest segment)
    all_covered = set()
    for _, idx in final:
        all_covered.update(idx)
    uncovered = sorted(set(range(n)) - all_covered)
    if uncovered and final:
        # assign each uncovered point to the nearest segment
        for pt in uncovered:
            best_seg = 0
            best_dist = abs(pt - final[0][1][0])
            for si, (_, idx) in enumerate(final):
                d = min(abs(pt - idx[0]), abs(pt - idx[-1]))
                if d < best_dist:
                    best_dist = d
                    best_seg = si
            final[best_seg][1].append(pt)
        # re-sort indices within each segment
        final = [(lbl, sorted(idx)) for lbl, idx in final]

    return final if final else [("straight", list(range(n)))]


# ═══════════════════════════════════════════════════════════════════════
# ARC ANGLE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

def estimate_arc_angle(points: np.ndarray) -> float:
    """
    Sum of per-vertex turning angles (degrees).
    Robust to noise — no circle fitting needed.
    """
    if len(points) < 3:
        return 0.0

    total = 0.0
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        total += math.degrees(math.acos(cos_a))

    return total


# ═══════════════════════════════════════════════════════════════════════
# JUNCTION NEIGHBOURHOOD
# ═══════════════════════════════════════════════════════════════════════

def _junction_neighbourhood(adj: Dict[int, Set[int]],
                            junction_node: int,
                            k_hops: int) -> Set[int]:
    """BFS k-hop neighbourhood around a junction node."""
    visited = {junction_node}
    frontier = {junction_node}
    for _ in range(k_hops):
        new_frontier = set()
        for n in frontier:
            for nbr in adj[n]:
                if nbr not in visited:
                    visited.add(nbr)
                    new_frontier.add(nbr)
        frontier = new_frontier
    return visited


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE DOWNSAMPLING (curvature-preserving)
# ═══════════════════════════════════════════════════════════════════════

def adaptive_downsample(points: np.ndarray,
                        target_nodes: int = 16,
                        min_nodes: int = 5) -> np.ndarray:
    """
    Downsample a polyline to ~target_nodes with UNIFORM arc-length spacing.

    Strategy (dichotomy / bisection-based):
      1. Compute cumulative arc-length along the polyline
      2. Place target_nodes at equally-spaced arc-length positions
      3. For each target position, pick the nearest original point
      4. Endpoints are always kept

    This guarantees perfectly even spacing along the curve, regardless
    of whether the segment is straight or curved. No clustering at
    high-curvature regions, no sparse gaps at the tail.

    Parameters
    ──────────
    points       : (N, 3) polyline
    target_nodes : desired output count (~10-20 for model input)
    min_nodes    : never go below this

    Returns
    ───────
    downsampled : (M, 3) where M = max(min_nodes, min(N, target_nodes))
    """
    n = len(points)
    target = max(min_nodes, min(n, target_nodes))
    if n <= target:
        return points.copy()

    # cumulative arc-length
    cum = _polyline_arc_lengths(points)
    total = cum[-1]

    if total < 1e-12:
        # degenerate: all points at same location
        return points[:target].copy()

    # place target_nodes at equally-spaced arc-length positions
    t_targets = np.linspace(0.0, total, target)

    # interpolate along the polyline at each target arc-length
    result = np.zeros((target, points.shape[1]), dtype=points.dtype)
    result[0] = points[0]
    result[-1] = points[-1]

    j = 0  # current segment index
    for i in range(1, target - 1):
        t = t_targets[i]
        # advance j so that cum[j] <= t < cum[j+1]
        while j < n - 2 and cum[j + 1] < t:
            j += 1
        seg_len = cum[j + 1] - cum[j]
        if seg_len < 1e-12:
            result[i] = points[j]
        else:
            frac = (t - cum[j]) / seg_len
            result[i] = points[j] + frac * (points[j + 1] - points[j])

    return result


def downsample_segment_for_model(segment_points: np.ndarray,
                                  target_nodes: int = 16) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Downsample a segment's points for model inference.
    Returns (downsampled_nodes, sequential_edges).
    """
    ds = adaptive_downsample(segment_points, target_nodes=target_nodes)
    edges = [[i, i + 1] for i in range(len(ds) - 1)]
    return ds, edges


# ═══════════════════════════════════════════════════════════════════════
# SUBGRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_segment_subgraph(
    original_nodes: np.ndarray,
    original_edges: List[Tuple[int, int]],
    node_ids: List[int],
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[int, int]]:
    """
    Extract a subgraph for the given node_ids.
    Returns (sub_nodes, sub_edges, old_to_new_mapping).
    """
    id_set = set(node_ids)
    old_to_new = {old: new for new, old in enumerate(sorted(node_ids))}
    sub_nodes = original_nodes[sorted(node_ids)]
    sub_edges = []
    for u, v in original_edges:
        if u in id_set and v in id_set:
            sub_edges.append((old_to_new[u], old_to_new[v]))
    return sub_nodes, sub_edges, old_to_new


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def split_centerline_graph(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]],
    params: Optional[SegmentParams] = None,
) -> List[SegmentRecord]:
    """
    Full pipeline: split a centerline graph into junction / straight / arc segments.

    Parameters
    ──────────
    nodes  : (N, 3) float
    edges  : list of (i, j)
    params : SegmentParams (uses defaults if None)

    Returns
    ───────
    list[SegmentRecord]
    """
    if params is None:
        params = SegmentParams()

    n_nodes = len(nodes)
    adj = build_adjacency(edges, n_nodes)
    junction_nodes = detect_junction_nodes(adj)

    print(f"[seg] {n_nodes} nodes, {len(edges)} edges, {len(junction_nodes)} junctions")

    # ── 1. junction neighbourhoods ──
    junction_node_ids: Set[int] = set()
    junction_segments: List[SegmentRecord] = []
    seg_id = 0

    for jn in sorted(junction_nodes):
        hood = _junction_neighbourhood(adj, jn, params.junction_k_hops)
        junction_node_ids.update(hood)

        length = 0.0
        sorted_hood = sorted(hood)
        for i in range(len(sorted_hood) - 1):
            a, b = sorted_hood[i], sorted_hood[i + 1]
            if b in adj[a]:
                length += np.linalg.norm(nodes[a] - nodes[b])

        junction_segments.append(SegmentRecord(
            segment_id=seg_id,
            type="junction",
            node_ids_original=sorted(hood),
            length=length,
            mean_curvature=0.0,
            max_curvature=0.0,
        ))
        seg_id += 1

    print(f"[seg] {len(junction_segments)} junction segments "
          f"({len(junction_node_ids)} nodes in junction zones)")

    # ── 2. extract branches ──
    branches = extract_branches(adj, junction_nodes)
    print(f"[seg] {len(branches)} branches extracted")

    # ── 3. process each branch ──
    curve_segments: List[SegmentRecord] = []

    for bi, branch in enumerate(branches):
        branch_pts = nodes[branch.node_ids]
        branch_len = float(_polyline_arc_lengths(branch_pts)[-1])

        # resample
        resampled, _ = resample_polyline(branch_pts, params.target_step)
        n_res = len(resampled)

        # curvature
        kappa = compute_branch_curvature(resampled, smooth_window=params.smooth_window)

        # collinearity pre-pass
        is_straight = compute_collinearity_mask(resampled, collinear_tol=1e-4)
        n_collinear = int(is_straight.sum())

        # auto thresholds if needed (computed from curved points only)
        k_low = params.k_low if params.k_low > 0 else None
        k_high = params.k_high if params.k_high > 0 else None
        if k_low is None or k_high is None:
            auto_lo, auto_hi = auto_thresholds(kappa, is_straight)
            k_low = k_low or auto_lo
            k_high = k_high or auto_hi

        print(f"  branch {bi}: {len(branch.node_ids)} nodes -> {n_res} resampled, "
              f"len={branch_len:.1f}, collinear={n_collinear}/{n_res}, "
              f"kappa=[{kappa.min():.4f}, {kappa.max():.4f}], "
              f"k_low={k_low:.4f}, k_high={k_high:.4f}")

        # segment by curvature (with collinearity pre-filter)
        runs = segment_branch_by_curvature(resampled, kappa, params, k_low, k_high)

        # map resampled indices back to original branch node ids
        # (approximate: map each resampled point to nearest original node)
        orig_pts = branch_pts
        orig_cum = _polyline_arc_lengths(orig_pts)
        res_cum = _polyline_arc_lengths(resampled)

        def _resample_idx_to_original(res_idx: int) -> int:
            """Map a resampled point index to the nearest original node index."""
            if res_idx >= len(res_cum):
                res_idx = len(res_cum) - 1
            t = res_cum[res_idx] if res_idx < len(res_cum) else res_cum[-1]
            # scale t to original arc-length space
            if res_cum[-1] > 1e-12 and orig_cum[-1] > 1e-12:
                t_orig = t * (orig_cum[-1] / res_cum[-1])
            else:
                t_orig = t
            nearest = int(np.argmin(np.abs(orig_cum - t_orig)))
            return nearest

        for label, res_indices in runs:
            # map to original node ids
            orig_branch_indices = sorted(set(
                _resample_idx_to_original(ri) for ri in res_indices
            ))
            orig_node_ids = [branch.node_ids[oi] for oi in orig_branch_indices]

            # skip nodes already claimed by junction zones
            orig_node_ids = [nid for nid in orig_node_ids if nid not in junction_node_ids]
            if len(orig_node_ids) < 2:
                continue

            seg_pts = nodes[orig_node_ids]
            seg_len = float(_polyline_arc_lengths(seg_pts)[-1])
            seg_kappa = np.array([kappa[ri] for ri in res_indices if ri < len(kappa)])
            mean_k = float(seg_kappa.mean()) if len(seg_kappa) > 0 else 0.0
            max_k = float(seg_kappa.max()) if len(seg_kappa) > 0 else 0.0

            rec = SegmentRecord(
                segment_id=seg_id,
                type=label,
                node_ids_original=orig_node_ids,
                length=seg_len,
                mean_curvature=mean_k,
                max_curvature=max_k,
            )

            if label == "arc":
                angle = estimate_arc_angle(seg_pts)
                rec.arc_angle_deg = angle
                if mean_k > 1e-8:
                    rec.radius_est = 1.0 / mean_k

                # ── corner detection: short arc with sharp curvature spike ──
                is_short = seg_len <= params.corner_max_length
                has_angle = angle >= params.corner_min_angle
                # sharpness: how concentrated is the curvature?
                # a corner has a spike (max >> mean), an arc is uniform (max ≈ mean)
                is_sharp = (max_k / max(mean_k, 1e-9)) >= params.corner_sharpness
                # also: if the segment is very short relative to its turning
                # (high angle-per-unit-length), it's a corner
                angle_density = angle / max(seg_len, 1e-9)  # deg per unit length
                is_dense_turn = angle_density > 30.0  # >30 deg/unit = sharp

                if is_short and has_angle and (is_sharp or is_dense_turn):
                    rec.type = "corner"
                    rec.corner_angle_deg = angle

            curve_segments.append(rec)
            seg_id += 1

    all_segments = junction_segments + curve_segments

    # ── summary ──
    type_counts = {}
    for s in all_segments:
        type_counts[s.type] = type_counts.get(s.type, 0) + 1
    print(f"[seg] final: {len(all_segments)} segments — {type_counts}")

    # ── validation ──
    _validate_coverage(nodes, adj, all_segments, junction_node_ids)

    return all_segments


def _validate_coverage(nodes: np.ndarray,
                       adj: Dict[int, Set[int]],
                       segments: List[SegmentRecord],
                       junction_node_ids: Set[int]) -> None:
    """Check that every node is covered by at least one segment."""
    covered = set()
    for seg in segments:
        covered.update(seg.node_ids_original)

    n_total = len(nodes)
    uncovered = set(range(n_total)) - covered
    # isolated nodes (degree 0) are OK to miss
    isolated = {n for n in uncovered if len(adj.get(n, set())) == 0}
    real_uncovered = uncovered - isolated

    if real_uncovered:
        print(f"[warn] {len(real_uncovered)} non-isolated nodes not covered by any segment")
    else:
        print(f"[check] all {n_total} connected nodes covered -- OK")


# ═══════════════════════════════════════════════════════════════════════
# SERIALISATION
# ═══════════════════════════════════════════════════════════════════════

def segments_to_json(segments: List[SegmentRecord]) -> List[Dict[str, Any]]:
    """Convert segment records to JSON-serialisable dicts."""
    out = []
    for s in segments:
        d = asdict(s)
        # round floats
        for k in ("length", "mean_curvature", "max_curvature",
                   "arc_angle_deg", "radius_est", "corner_angle_deg"):
            if k in d:
                d[k] = round(d[k], 4)
        out.append(d)
    return out
