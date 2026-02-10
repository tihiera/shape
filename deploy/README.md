# Centerline Segmentation API v2

FastAPI server: **STEP/JSON → centerline → segment → query**.
Detects **junctions**, **straights**, **arcs**, and **corners**.

## Quick Start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # optional, for AI queries
python app.py
# -> http://localhost:8000
# -> http://localhost:8000/docs (Swagger UI)
```

## Architecture

```
Frontend (VTK.js + localStorage)
  │
  │  localStorage: { email, uid }
  │  If no email → redirect to login page
  │
  ├── POST /auth/login {email}     → returns {uid, sessions, is_new}
  ├── GET  /auth/me?uid=...        → verify uid, get sessions
  │
  ├── POST /upload                  Upload STEP/JSON file
  ├── POST /segment                 Segment inline graph
  ├── POST /query                   NL -> AI -> result
  ├── POST /dsl                     Direct DSL execution
  │
  └── WS /ws/{uid}/{session_id}     Streaming pipeline + chat
        ↕ streams progress + results in real-time
```

### Pipeline

```
STEP/JSON file
  → 1. Ingest (gmsh/meshio for STEP, direct parse for JSON)
  → 2. Centerline extraction (preserves original node IDs)
  → 3. Segmentation (junction/straight/arc/corner)
  → 4. Downsample (uniform arc-length, ~16 nodes)
  → 5. Embed (optional, GATv2 model)
  → 6. Store (JSON + pickle + npy per session)
  → 7. Query (NL -> DSL -> execute)
```

## Segment Types

| Type | Description | Key Fields |
|------|-------------|------------|
| **junction** | Node with ≥3 connections | — |
| **straight** | Zero curvature (κ≈0) | `length` |
| **arc** | Gradual curve (constant κ) | `arc_angle_deg`, `radius_est` |
| **corner** | Sharp turn (short, high κ spike) | `corner_angle_deg` |

## API Endpoints

### `POST /auth/login`

Login with email (no password):
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "alice@example.com"}'
# Returns: {"uid": "a1b2c3d4...", "is_new": true, "sessions": []}
```

### `GET /auth/me`

Verify uid on page load:
```bash
curl "http://localhost:8000/auth/me?uid=a1b2c3d4..."
# Returns: {"uid": "...", "email": "alice@example.com", "sessions": [...]}
```

### `POST /upload`

Upload a STEP or JSON file:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@pipe.step" \
  -F "uid=a1b2c3d4..."
```

### `POST /segment`

Segment an inline graph:
```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{"nodes": [[0,0,0],[1,0,0]], "edges": [[0,1]], "uid": "a1b2c3d4..."}'
```

### `POST /query`

Natural language query:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show all arcs greater than 30 degrees", "uid": "a1b2c3d4...", "session_id": "abc123"}'
```

### `POST /dsl`

Direct DSL command:
```bash
curl -X POST http://localhost:8000/dsl \
  -H "Content-Type: application/json" \
  -d '{"action": "filter", "params": {"type": "arc", "arc_angle_deg__gt": 30}, "uid": "a1b2c3d4...", "session_id": "abc123"}'
```

### `WS /ws/{uid}/{session_id}`

WebSocket for streaming:
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/a1b2c3d4.../session1");

// Send segmentation request
ws.send(JSON.stringify({
  type: "segment",
  nodes: [[0,0,0], [1,0,0], ...],
  edges: [[0,1], ...],
  target_step: 0.15
}));

// Receive streaming updates
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  // msg.type = "progress" | "result" | "error"
  // msg.step = "segmenting" | "downsampling" | "embedding" | ...
  console.log(msg);
};

// Query
ws.send(JSON.stringify({
  type: "query",
  query: "how many arcs are there?"
}));

// Highlight specific segments (for VTK.js overlay)
ws.send(JSON.stringify({
  type: "highlight",
  segment_ids: [1, 3, 5]
}));
```

## AI-Powered Queries (Gemini)

Queries use **Gemini 2.0 Flash** with function calling. The LLM:
- Understands natural language and maps it to tool calls
- Chains multiple tools for complex queries
- Provides natural language explanations of results
- Falls back to rule-based parsing if Gemini is unavailable

```bash
export GEMINI_API_KEY=your_key_here
```

### Example complex queries the AI handles:

| Query | What happens |
|-------|-------------|
| "Find all arcs > 30° and highlight the longest one" | filter → topk → highlight (3 tool calls) |
| "Compare the total length of straights vs arcs" | sum(straight) + sum(arc) (2 tool calls) |
| "What's the sharpest corner and where is it?" | topk_by(corner_angle) → describe → highlight |
| "Give me an overview of the pipe layout" | group_by + list_segments |

### DSL Tools (called by Gemini)

| Tool | Description |
|------|-------------|
| `list_segments` | Show all segments |
| `filter_segments` | Filter by type, angle, length, curvature |
| `count_segments` | Count segments (optionally filtered) |
| `sum_field` | Sum a numeric field |
| `group_by` | Group by any field |
| `topk_by` | Top-K by any numeric field |
| `describe_segment` | Full detail of one segment |
| `highlight_segments` | Highlight for VTK.js overlay |

## Auth Flow (No Auth Service)

```
Frontend                              Backend
───────                              ───────
1. Check localStorage for {email,uid}
2. If missing → show login page
3. User enters email
4. POST /auth/login {email}  ──────→  Lookup email in data/users.json
                                       ├─ New? Create uid + folder → is_new=true
                                       └─ Exists? Update last_login → is_new=false
5. Store {email, uid} in localStorage
6. All requests use uid (not email)
7. On page reload: GET /auth/me?uid=  → verify uid still valid
```

**`data/users.json`** (the "database"):
```json
{
  "alice@example.com": {
    "uid": "c3da791c78204e44a18c12e3be069540",
    "created_at": "2026-02-09T12:05:44",
    "last_login": "2026-02-09T14:30:00"
  }
}
```

## Session Storage

```
data/
├── users.json                          # Email → UUID registry
└── {uid}/                              # 32-char hex UUID (no dashes)
    └── {session_id}/
        ├── upload/                     # Raw uploaded files
        ├── centerline.json             # Extracted graph
        ├── mesh.json                   # Mesh + cl-to-mesh mapping
        ├── segments.json               # Results (JSON)
        ├── segments.pkl                # Results (pickle, fast)
        ├── embeddings.npy              # ML embeddings
        └── chat.jsonl                  # Chat history
```

## Files

```
deploy/
├── app.py                      # FastAPI entry point (REST + WebSocket)
├── centerline_segmenter.py     # Segmentation algorithm
├── inference.py                # ML embedding (standalone)
├── model.py                    # ShapeEncoder GATv2 (standalone)
├── ai/
│   ├── gemini.py               # Gemini client + function calling
│   └── prompts.py              # System prompts + context builder
├── dsl/
│   └── engine.py               # DSL executor + AI query + rule fallback
├── services/
│   ├── geometry_ingest.py      # STEP/JSON ingest
│   ├── segmentation.py         # Pipeline orchestrator
│   └── session.py              # User/session management
├── weights/
│   ├── encoder.pt              # Trained model
│   └── meta.json               # Model metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | gemini-3-flash-preview| Gemini model name |
| `PORT` | 8000 | Server port |
| `DEVICE` | auto | PyTorch device |
| `MODEL_CHECKPOINT` | weights/encoder.pt | Model path |
