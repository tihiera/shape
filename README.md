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
git fi| `category` | string | Shape class label (e.g. `straight`, `arc_90`, `corner`, `junction`) |
| `nodes` | float[][] (N x 3) | 3D points with ~1-unit spacing between neighbors |
| `edges` | int[][] (E x 2) | Index pairs connecting nodes |
| `features.curvature` | float[] (N) | Menger curvature per node (0 at endpoints/hubs) |
| `features.segment_angle` | float[] (N) | Angle in degrees between edges at each node |

## Categories

| Category | What it is | Samples |
|----------|-----------|---------|
| `straight` | Random-length line segment (5-20 units) | ![](dataset_output/categories/straight.png) |
| `arc_10` | 10 deg gentle arc | ![](dataset_output/categories/arc_10.png) |
| `arc_20` | 20 deg arc | ![](dataset_output/categories/arc_20.png) |
| `arc_30` | 30 deg arc | ![](dataset_output/categories/arc_30.png) |
| `arc_40` | 40 deg arc | ![](dataset_output/categories/arc_40.png) |
| `arc_50` | 50 deg arc | ![](dataset_output/categories/arc_50.png) |
| `arc_60` | 60 deg arc | ![](dataset_output/categories/arc_60.png) |
| `arc_70` | 70 deg arc | ![](dataset_output/categories/arc_70.png) |
| `arc_80` | 80 deg arc | ![](dataset_output/categories/arc_80.png) |
| `arc_90` | 90 deg quarter-circle bend | ![](dataset_output/categories/arc_90.png) |
| `arc_100` | 100 deg arc | ![](dataset_output/categories/arc_100.png) |
| `arc_110` | 110 deg arc | ![](dataset_output/categories/arc_110.png) |
| `arc_120` | 120 deg wide arc | ![](dataset_output/categories/arc_120.png) |
| `arc_130` | 130 deg arc | ![](dataset_output/categories/arc_130.png) |
| `arc_140` | 140 deg arc | ![](dataset_output/categories/arc_140.png) |
| `arc_150` | 150 deg near-semicircle | ![](dataset_output/categories/arc_150.png) |
| `arc_160` | 160 deg arc | ![](dataset_output/categories/arc_160.png) |
| `arc_170` | 170 deg near-full semicircle | ![](dataset_output/categories/arc_170.png) |
| `corner` | Two straight arms at a sharp 90 deg angle | ![](dataset_output/categories/corner.png) |
| `junction` | T or Y junction (3 arms meeting at a hub node) | ![](dataset_output/categories/junction_T.png) |

## Motif types (for training)

After preprocessing, categories are mapped to 4 motif types:

| motif_type | categories |
|------------|-----------|
| `arc` | `arc_10` .. `arc_170` |
| `straight` | `straight` |
| `corner` | `corner` |
| `junction` | `junction` (merged from junction_T + junction_Y) |

## Why this structure?

- **Nodes + edges = graph** -- directly loadable into GNN frameworks (PyTorch Geometric, DGL).
- **Unit spacing** -- standardises graph density across shapes.
- **Random rigid pose** -- every sample gets a random 3D rotation + translation so the model can't memorise orientation.
- **Per-node features** -- curvature and degree give the network local geometric + topological cues.
- **Junctions share a hub node** -- one node connected to 3 edges, matching real pipe topology.

## Regenerate

```bash
python generate_geometry_dataset.py
```

Generates 10000 samples per category (17 arc classes + straight + corner + junction_T + junction_Y = 21 raw categories, merged to 20 for training).
