# Centerline Segmentation API

FastAPI server that detects **junctions**, **straights**, and **arcs** in 3D centerline graphs.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Server runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t centerline-api .
docker run -p 8000:8000 centerline-api
```

## API

### `POST /segment`

```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [[0,0,0], [1,0,0], [2,0,0], [3,1,0], [4,2,0]],
    "edges": [[0,1], [1,2], [2,3], [3,4]],
    "target_step": 1.0,
    "downsample_nodes": 16,
    "embed": false
  }'
```

**Response:**
```json
{
  "segments": [
    {
      "segment_id": 0,
      "type": "straight",
      "node_count": 5,
      "length": 4.83,
      "mean_curvature": 0.0,
      "max_curvature": 0.0,
      "arc_angle_deg": 0.0,
      "radius_est": 0.0,
      "downsampled_nodes": [[0,0,0], ...],
      "downsampled_edges": [[0,1], [1,2], ...],
      "embedding": null
    }
  ],
  "summary": {
    "total_segments": 1,
    "counts_by_type": {"straight": 1},
    "input_nodes": 5,
    "input_edges": 4,
    "model_available": false
  }
}
```

### Parameters

| Field | Default | Description |
|-------|---------|-------------|
| `nodes` | required | Nx3 coordinates `[[x,y,z], ...]` |
| `edges` | required | Edge pairs `[[i,j], ...]` |
| `target_step` | 1.0 | Resampling resolution for curvature |
| `downsample_nodes` | 16 | Target nodes per segment |
| `embed` | false | Run ML embedding (requires torch) |

### Segment Types

| Type | Description | Extra fields |
|------|-------------|-------------|
| `junction` | Node with ≥3 connections | — |
| `straight` | Zero curvature | — |
| `arc` | Constant curvature | `arc_angle_deg`, `radius_est` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `DEVICE` | auto | PyTorch device (cpu/cuda/mps) |
| `MODEL_CHECKPOINT` | weights/encoder.pt | Path to model weights |

## Files

```
deploy/
├── app.py                  # FastAPI server (entry point)
├── centerline_segmenter.py # Segmentation algorithm
├── inference.py            # ML embedding inference
├── model.py                # ShapeEncoder (GATv2)
├── weights/
│   ├── encoder.pt          # Trained model checkpoint
│   └── meta.json           # Model metadata
├── requirements.txt
├── Dockerfile
└── README.md
```
