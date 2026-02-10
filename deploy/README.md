# Centerline Segmentation API

FastAPI server: **STEP/JSON → centerline → segment → query**.
Detects **junctions**, **straights**, **arcs**, and **corners**.

> 📐 **Full architecture documentation**: see [`ARCHITECTURE.md`](./ARCHITECTURE.md) for detailed diagrams, data flows, and Gemini 3 integration deep-dive.

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

## How We Use Gemini 3 Flash

> For the full deep-dive with diagrams, see [`ARCHITECTURE.md` → "How We Use Gemini 3 Flash"](./ARCHITECTURE.md#how-we-use-gemini-3-flash)

**Gemini 3 Flash** (`gemini-3-flash-preview` via `google-genai` Python SDK) is the core intelligence layer. It replaces what would traditionally be a brittle rule-based NLP parser with a context-aware AI agent that can reason over geometry data.

### Function Calling (Tool Use)

We define **8 DSL tools** as Gemini function declarations. When a user asks a question in natural language, Gemini autonomously decides which tools to call and in what order:

```
User: "Find all arcs above 60° and highlight the sharpest one"

  Round 1 → Gemini calls: filter_segments(type="arc", arc_angle_deg__gt=60)
  Round 2 → Gemini calls: topk_by(field="arc_angle_deg", k=1)
  Round 3 → Gemini calls: highlight_segments(segment_ids=[9])
  Round 4 → Gemini generates answer:
            "There are 4 arcs above 60°. The sharpest is Segment #9,
             a 180° U-bend. I've highlighted it for you."
```

Each round, Gemini receives the tool result and decides whether to call another tool or produce a final answer. Up to 8 rounds per query.

### Conversation Context & Follow-ups

Chat history (including previous tool calls and results) is injected into every Gemini request, enabling natural follow-ups:

```
User: "how many arcs above 40°?"     → Gemini calls filter_segments → "6 arcs"
User: "and above 80°?"               → Gemini understands context → "3 arcs"
User: "does it contain straights?"   → Gemini switches type → "7 straight sections"
```

### 8 DSL Tools Available to Gemini

| Tool | Purpose | Example Trigger |
|------|---------|----------------|
| `list_segments` | Overview of all segments | "describe this geometry" |
| `filter_segments` | Filter by type/angle/length/curvature | "show arcs above 90°" |
| `count_segments` | Count (optionally filtered) | "how many straights?" |
| `sum_field` | Sum a numeric field | "total length of all arcs" |
| `group_by` | Group by type with stats | "break down by type" |
| `topk_by` | Top-K by any numeric field | "3 sharpest bends" |
| `describe_segment` | Full detail of one segment | "tell me about segment #5" |
| `highlight_segments` | Highlight in 3D viewer | "highlight the U-bends" |

### Configuration

```bash
# .env file in deploy/
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3-flash-preview   # default
```

| Setting | Value |
|---------|-------|
| Model | `gemini-3-flash-preview` |
| SDK | `google-genai` (Python) |
| Temperature | `0.1` (deterministic, factual) |
| Max tool rounds | `8` per query |

### Example Complex Queries

| Query | What Gemini Does |
|-------|-----------------|
| "Find all arcs > 30° and highlight the longest one" | filter → topk → highlight (3 tool calls) |
| "Compare the total length of straights vs arcs" | sum(straight) + sum(arc) (2 tool calls) |
| "What's the sharpest corner and where is it?" | topk_by(corner_angle) → describe → highlight |
| "Give me an overview of the pipe layout" | group_by + list_segments |

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
│   ├── gemini.py               # Gemini 3 client + function calling + multi-turn
│   ├── prompts.py              # System prompts + context builder
│   └── test_gemini.py          # Integration tests
├── dsl/
│   └── engine.py               # DSL executor + Gemini-powered query_smart()
├── services/
│   ├── geometry_ingest.py      # STEP/MSH/JSON ingest + centerline extraction
│   ├── segmentation.py         # Pipeline orchestrator
│   └── session.py              # User/session management
├── weights/
│   ├── encoder.pt              # Trained model
│   └── meta.json               # Model metadata
├── mesh/                       # Test pipe meshes (5 types)
├── requirements.txt
├── Dockerfile
├── ARCHITECTURE.md             # Full architecture docs + Gemini deep-dive
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
