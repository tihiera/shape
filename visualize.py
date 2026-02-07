"""
visualize.py
────────────
3D scatter + edge plotting for shape-graph samples.
  plot_samples  – one representative per category (overview)
  plot_gallery  – rows = categories, cols = multiple samples each (gallery)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers projection)
from typing import Dict, List, Any

# one colour per major category family
_CATEGORY_COLOURS: Dict[str, str] = {
    "straight":   "#1f77b4",
    "arc":        "#ff7f0e",
    "corner":     "#2ca02c",
    "junction_T": "#d62728",
    "junction_Y": "#9467bd",
}


def _colour_for(category: str) -> str:
    """Return the colour mapped to the category family."""
    for prefix, colour in _CATEGORY_COLOURS.items():
        if category.startswith(prefix) or category == prefix:
            return colour
    return "#333333"


# ──────────────────────────────────────────────
# helper: draw one shape into an Axes3D
# ──────────────────────────────────────────────

def _draw_shape(ax, sample: Dict[str, Any], show_labels: bool = True) -> None:
    """Render a single shape-graph dict on the given 3-D axes."""
    nodes = sample["nodes"]
    edges = sample["edges"]
    colour = _colour_for(sample["category"])

    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]],
                [nodes[i, 1], nodes[j, 1]],
                [nodes[i, 2], nodes[j, 2]],
                color=colour, linewidth=1.5)

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
               color="black", s=14, depthshade=True, zorder=5)

    if show_labels:
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)
    _set_equal_aspect_3d(ax, nodes)


# ──────────────────────────────────────────────
# 1. plot_samples  (overview – one per category)
# ──────────────────────────────────────────────

def plot_samples(samples: List[Dict[str, Any]],
                 title: str = "Shape Samples (first per category)",
                 save_path: str | None = None) -> None:
    """
    Plot one sample per category in a grid of 3-D subplots.
    """
    n = len(samples)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle(title, fontsize=14, y=1.02)

    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _draw_shape(ax, sample)
        ax.set_title(sample["category"], fontsize=11)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[vis] figure saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 2. plot_gallery  (many samples per category)
# ──────────────────────────────────────────────

def plot_gallery(dataset: List[Dict[str, Any]],
                 max_per_cat: int = 10,
                 title: str = "Gallery — all samples per category",
                 save_path: str | None = None) -> None:
    """
    Gallery grid:  rows = categories (sorted), cols = individual samples.
    Shows up to *max_per_cat* samples per category.
    """
    # group by category, preserving insertion order
    groups: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for s in dataset:
        cat = s["category"]
        if cat not in groups:
            groups[cat] = []
        if len(groups[cat]) < max_per_cat:
            groups[cat].append(s)

    n_rows = len(groups)
    n_cols = max(len(v) for v in groups.values())

    cell_w, cell_h = 3.4, 3.4
    fig = plt.figure(figsize=(cell_w * n_cols, cell_h * n_rows))
    fig.suptitle(title, fontsize=14, y=1.01)

    for row_idx, (cat, samples) in enumerate(groups.items()):
        for col_idx, sample in enumerate(samples):
            ax_idx = row_idx * n_cols + col_idx + 1
            ax = fig.add_subplot(n_rows, n_cols, ax_idx, projection="3d")
            _draw_shape(ax, sample, show_labels=False)

            # row label on the first column
            if col_idx == 0:
                ax.set_ylabel(cat, fontsize=9, fontweight="bold", labelpad=12)
            # column header on first row
            if row_idx == 0:
                ax.set_title(f"#{col_idx + 1}", fontsize=8, color="gray")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[vis] gallery saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 3. plot_category_sheets  (one PNG per category)
# ──────────────────────────────────────────────

def plot_category_sheets(dataset: List[Dict[str, Any]],
                         max_per_cat: int = 10,
                         out_dir: str = "dataset_output/categories") -> List[str]:
    """
    Save one separate PNG per category.

    Each image shows up to *max_per_cat* samples laid out in a 2-row
    grid (5 columns × 2 rows for 10 samples).

    Returns the list of saved file paths.
    """
    from pathlib import Path

    # group samples by category
    groups: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for s in dataset:
        cat = s["category"]
        if cat not in groups:
            groups[cat] = []
        if len(groups[cat]) < max_per_cat:
            groups[cat].append(s)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for cat, samples in groups.items():
        n = len(samples)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(4.5 * cols, 4.5 * rows))
        fig.suptitle(f"{cat}  ({n} samples)", fontsize=15, fontweight="bold",
                     y=1.02)

        colour = _colour_for(cat)

        for idx, sample in enumerate(samples):
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            _draw_shape(ax, sample, show_labels=True)
            ax.set_title(f"#{idx + 1}", fontsize=9, color="gray")

        plt.tight_layout()
        path = out / f"{cat}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    print(f"[vis] {len(saved)} category sheets saved -> {out}/")
    return saved


# ──────────────────────────────────────────────
# helper
# ──────────────────────────────────────────────

def _set_equal_aspect_3d(ax, pts: np.ndarray) -> None:
    """Force equal axis scaling on a 3-D axes."""
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
