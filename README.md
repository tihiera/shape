---
license: mit
pretty_name: Shape Geometry Dataset
tags: [synthetic, geometry, graph-neural-network, 3d]
task_categories: [graph-ml]
size_categories: [100K<n<1M]
---

# Shape Geometry Dataset

Synthetic graph-based centerline representations of 3D geometric motifs (pipe-like structures).

## JSON Schema

`dataset.json` is an array of shape records. Each record:

```json
{
  "category": "arc_90",
  "nodes": [[x, y, z], ...],
  "edges": [[i, j], ...],
  "features": {
    "curvature": [0.0, 0.1, ...],
    "segment_angle": [0.0, 160.5, ...]
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | Shape class label (e.g. `straight`, `arc_90`, `corner`) |
| `nodes` | float[][] (N×3) | 3D points with ~1-unit spacing between neighbors |
| `edges` | int[][] (E×2) | Index pairs connecting nodes |
| `features.curvature` | float[] (N) | Menger curvature per node (0 at endpoints/hubs) |
| `features.segment_angle` | float[] (N) | Angle in degrees between edges at each node |

## Categories

| Category | What it is | Samples |
|----------|-----------|---------|
| `straight` | Random-length line segment (5–20 units) | ![](dataset_output/categories/straight.png) |
| `arc_10` | 10° gentle arc, R=10 | ![](dataset_output/categories/arc_10.png) |
| `arc_20` | 20° arc | ![](dataset_output/categories/arc_20.png) |
| `arc_30` | 30° arc | ![](dataset_output/categories/arc_30.png) |
| `arc_40` | 40° arc | ![](dataset_output/categories/arc_40.png) |
| `arc_50` | 50° arc | ![](dataset_output/categories/arc_50.png) |
| `arc_60` | 60° arc | ![](dataset_output/categories/arc_60.png) |
| `arc_70` | 70° arc | ![](dataset_output/categories/arc_70.png) |
| `arc_80` | 80° arc | ![](dataset_output/categories/arc_80.png) |
| `arc_90` | 90° quarter-circle bend | ![](dataset_output/categories/arc_90.png) |
| `arc_100` | 100° arc | ![](dataset_output/categories/arc_100.png) |
| `arc_110` | 110° arc | ![](dataset_output/categories/arc_110.png) |
| `arc_120` | 120° wide arc | ![](dataset_output/categories/arc_120.png) |
| `arc_130` | 130° arc | ![](dataset_output/categories/arc_130.png) |
| `arc_140` | 140° arc | ![](dataset_output/categories/arc_140.png) |
| `arc_150` | 150° near-semicircle | ![](dataset_output/categories/arc_150.png) |
| `arc_160` | 160° arc | ![](dataset_output/categories/arc_160.png) |
| `arc_170` | 170° near-full semicircle | ![](dataset_output/categories/arc_170.png) |
| `corner` | Two straight arms at a sharp 90° angle | ![](dataset_output/categories/corner.png) |
| `junction_T` | Through-pipe with perpendicular branch (T-shape) | ![](dataset_output/categories/junction_T.png) |
| `junction_Y` | Trunk splitting into two splayed arms (Y-shape) | ![](dataset_output/categories/junction_Y.png) |

## Why this structure?

- **Nodes + edges = graph** — directly loadable into GNN frameworks (PyTorch Geometric, DGL).
- **Unit spacing** — standardises graph density across shapes; a 15-unit straight and a 90° arc at R=10 both have proportional node counts.
- **Random rigid pose** — every sample gets a random 3D rotation + translation so the model can't memorise orientation.
- **Per-node features** — curvature and segment angle give the network local geometric cues beyond raw xyz, helping distinguish arcs from straights even when node counts overlap.
- **Junctions share a hub node** — T and Y shapes have one node connected to 3 edges, matching real pipe topology and giving the GNN a clear topological signal.

## Regenerate

```bash
python generate_geometry_dataset.py
```

Generates 10000 samples per category (17 arc classes + straight + corner + junction_T + junction_Y = 21 categories, 21k samples total).
