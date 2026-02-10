# Architecture: 3D Pipe Geometry Analysis Backend

## Problem Statement

Industrial pipe networks (oil & gas, HVAC, plumbing) are represented as 3D CAD models (STEP, MSH). Engineers need to **understand**, **query**, and **inspect** these geometries:  "How many bends exceed 90°?", "Where are the U-turns?", "Show me the sharpest elbow." This requires:

1. **Parsing** raw CAD into a meaningful graph representation (centerline extraction)
2. **Segmenting** the centerline into typed components (straight, arc, corner, junction)
3. **Querying** those segments in natural language with real-time feedback
4. **Visualizing** results on the original 3D mesh

Our solution is a **full-stack pipeline** that ingests geometry, segments it, and exposes an **AI-powered natural language interface** built on **Gemini 3 Flash** for querying and exploring the results.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React + VTK.js)                     │
│                                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │  Login    │  │  File Upload │  │  3D Mesh View │  │  Chat Panel  │  │
│  │  (email)  │  │  (.msh/.step)│  │  (VTK.js)    │  │  (NL query)  │  │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘  └──────┬───────┘  │
│       │               │                  │                  │          │
└───────┼───────────────┼──────────────────┼──────────────────┼──────────┘
        │               │                  │                  │
        │  REST         │  REST            │  REST            │  WebSocket
        │  POST         │  POST            │  GET             │  /ws/{uid}/{sid}
        │  /auth/login  │  /upload         │  /mesh/..        │
        ▼               ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI SERVER (app.py)                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        REST Endpoints                           │    │
│  │                                                                 │    │
│  │  POST /auth/login       → Static email auth (no password)      │    │
│  │  GET  /auth/me          → Validate UID, return user info       │    │
│  │  POST /upload           → Ingest .msh/.step/.json file         │    │
│  │  POST /query            → NL query → Gemini AI → answer        │    │
│  │  GET  /mesh/{uid}/{sid} → Surface mesh (vertices + faces)      │    │
│  │  GET  /chat/{uid}/{sid} → Chat history (messages array)        │    │
│  │  GET  /segments/{uid}/{sid} → Segmentation results             │    │
│  │  GET  /sessions/{uid}   → List all user sessions               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    WebSocket /ws/{uid}/{sid}                     │    │
│  │                                                                 │    │
│  │  Client → {"type": "upload_and_segment"}                       │    │
│  │  Server ← {"type": "progress", "step": "segmenting", ...}     │    │
│  │  Server ← {"type": "progress", "step": "downsampling", ...}   │    │
│  │  Server ← {"type": "result", "data": {segments, summary}}     │    │
│  │                                                                 │    │
│  │  Client → {"type": "query", "query": "how many arcs > 90°?"}  │    │
│  │  Server ← {"type": "progress", "step": "tool_call", ...}      │    │
│  │  Server ← {"type": "result", "data": {answer, tool_calls}}    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Telegram     │  │  services/   │  │    dsl/      │  │    ai/    │  │
│  │  Notifier     │  │              │  │              │  │           │  │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
│                           │                 │                 │        │
└───────────────────────────┼─────────────────┼─────────────────┼────────┘
                            │                 │                 │
                            ▼                 ▼                 ▼
```

---

## Module Breakdown

### 1. `services/geometry_ingest.py` — Geometry Ingest & Centerline Extraction

```
.msh / .step file
       │
       ▼
┌─────────────────────────────┐
│  read_msh_file()            │  ← meshio reads Gmsh .msh
│  read_step_to_mesh()        │  ← gmsh + meshio reads STEP
│                             │
│  Returns: vertices, faces,  │
│           edges (surface)   │
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  Centerline Extraction                           │
│                                                  │
│  Single-layer mesh?                              │
│  ├─ YES → extract_centerline_from_single_layer() │
│  │        (BFS ring detection → ring centroids)  │
│  │                                               │
│  └─ NO  → extract_centerline_from_mesh()         │
│           (VMTK: boundary loops → seed points    │
│            → vmtkcenterlines → polyline output)  │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────┐
│  Outputs (per session):     │
│  • centerline.json          │  ← nodes + edges (graph)
│  • mesh.json                │  ← full mesh + cl_to_mesh_map
│  • mesh_surface.json        │  ← vertices + faces (for VTK.js)
└─────────────────────────────┘
```

### 2. `centerline_segmenter.py` — Segmentation Engine

```
centerline graph (nodes, edges)
       │
       ▼
┌──────────────────────────────────────┐
│  1. Junction Detection               │
│     Nodes with degree ≥ 3            │
│     → flood-fill junction zones      │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  2. Branch Extraction                │
│     BFS from junction → junction     │
│     or junction → endpoint           │
│     → ordered polyline paths         │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  3. Per-Branch Curvature Analysis    │
│     • Resample to uniform spacing    │
│     • Menger curvature per point     │
│     • Collinearity test (cosine)     │
│     • Auto-threshold (k_low, k_high) │
│     • Hysteresis segmentation        │
│       ├─ collinear → "straight"      │
│       ├─ κ > k_high → "curved"       │
│       └─ in between → keep state     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  4. Corner Reclassification          │
│     Short arcs with sharp angle      │
│     → reclassified as "corner"       │
└──────────┬───────────────────────────┘
           │
           ▼
  List[SegmentRecord]
  Each has: type, node_ids, length,
  mean/max curvature, arc_angle_deg,
  corner_angle_deg, radius_est
```

**Segment Types:**
| Type | Description | Key Property |
|------|-------------|-------------|
| `junction` | Where 3+ pipes meet (T, Y, cross) | degree ≥ 3 |
| `straight` | Zero-curvature pipe run | κ ≈ 0 |
| `arc` | Constant-curvature bend (elbow) | arc_angle_deg, radius |
| `corner` | Very short, very sharp turn | corner_angle_deg |

### 3. `services/segmentation.py` — Pipeline Orchestrator

```
┌───────────────────────────────────────────────────────┐
│  run_full_pipeline(nodes, edges, session_dir, ...)    │
│                                                       │
│  Step 1: segment_graph()        → List[SegmentRecord] │
│  Step 2: downsample_segments()  → uniform arc-length  │
│  Step 3: embed_segments()       → GATv2 embeddings    │
│  Step 4: compute_face_ids()     → segment → mesh map  │
│  Step 5: build_segment_output() → JSON-serializable   │
│  Step 6: store results          → .json + .pkl + .npy │
│                                                       │
│  on_progress() callback → streams to WebSocket        │
└───────────────────────────────────────────────────────┘
```

### 4. `model.py` + `inference.py` — GATv2 Shape Encoder (Optional)

```
Downsampled segment (16 nodes, edges)
       │
       ▼
┌─────────────────────────────────────┐
│  ShapeEncoder (GATv2)               │
│                                     │
│  Node features: [curvature, degree] │
│  → Node MLP (2 → 128)              │
│  → 4× GATv2Conv (4 heads, residual)│
│  → AttentionalAggregation pool      │
│  → Projection MLP → 256-dim        │
│  → L2 normalize                     │
│                                     │
│  Output: 256-dim embedding vector   │
└─────────────────────────────────────┘
```

### 5. `services/session.py` — User & Session Management

```
data/
├── users.json                    ← email → {uid, created_at, last_login}
└── {uid}/                        ← 32-char hex UUID (no dashes)
    └── {session_id}/             ← 8-char hex
        ├── upload/               ← original uploaded file
        ├── centerline.json       ← extracted centerline graph
        ├── mesh.json             ← full mesh + mapping
        ├── mesh_surface.json     ← vertices + faces for VTK.js
        ├── segments.json         ← segmentation results (JSON)
        ├── segments.pkl          ← segmentation results (pickle)
        ├── embeddings.npy        ← ML embeddings
        ├── chat.jsonl            ← conversation history
        └── meta.json             ← session metadata
```

---

## How We Use Gemini 3 Flash

Gemini 3 Flash (`gemini-3-flash-preview`) is the **core intelligence layer** of the system. It replaces what would traditionally be a brittle rule-based NLP parser with a powerful, context-aware AI agent.

### Architecture: AI Layer (`ai/` folder)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ai/ MODULE                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  gemini.py — Gemini Client Wrapper                          │    │
│  │                                                             │    │
│  │  • Singleton client (google-genai SDK)                      │    │
│  │  • 8 tool/function declarations (DSL primitives)            │    │
│  │  • Multi-turn function calling loop (up to 8 rounds)        │    │
│  │  • Conversation history injection (with tool call context)  │    │
│  │  • Async variant for WebSocket handlers                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  prompts.py — System Prompts & Context Builders             │    │
│  │                                                             │    │
│  │  • System prompt: role, segment types, field definitions,   │    │
│  │    follow-up handling instructions, response style guide    │    │
│  │  • build_segments_context(): rich text summary of segments  │    │
│  │  • format_step_explanation(): human-readable pipeline steps │    │
│  │  • summarize_pipeline_intro(): upload welcome message       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### How Gemini 3 Is Integrated

**1. Function Calling (Tool Use)**

We define **8 DSL tools** as Gemini function declarations. Gemini decides which tools to call based on the user's natural language query:

| Tool | Purpose | Example Trigger |
|------|---------|----------------|
| `list_segments` | List all segments | "describe this geometry" |
| `filter_segments` | Filter by type/angle/length | "show arcs above 90°" |
| `count_segments` | Count (optionally filtered) | "how many straights?" |
| `sum_field` | Sum a numeric field | "total length of all arcs" |
| `group_by` | Group by type with stats | "break down by type" |
| `topk_by` | Top-K by a field | "3 sharpest bends" |
| `describe_segment` | Full detail of one segment | "tell me about segment #5" |
| `highlight_segments` | Highlight in 3D viewer | "highlight the U-bends" |

**2. Multi-Turn Reasoning**

Gemini can chain multiple tool calls in a single query. For example:

```
User: "Find all arcs above 60° and highlight the sharpest one"

  Round 1: Gemini calls filter_segments(type="arc", arc_angle_deg__gt=60)
           → gets 4 matching arcs

  Round 2: Gemini calls topk_by(field="arc_angle_deg", k=1)
           → gets segment #9 (180° U-bend)

  Round 3: Gemini calls highlight_segments(segment_ids=[9])
           → returns highlight data for frontend

  Round 4: Gemini generates natural language answer:
           "There are 4 arcs above 60°. The sharpest is Segment #9,
            a 180° U-bend with radius 7.6. I've highlighted it for you."
```

**3. Conversation Context & Follow-ups**

Each user message is sent to Gemini with the **full conversation history**, including summaries of previous tool calls and their results. This enables follow-up queries:

```
User: "how many arcs above 40°?"
AI:   calls filter_segments(type="arc", arc_angle_deg__gt=40) → 6 arcs
      "There are 6 arcs above 40°..."

User: "and above 80°?"
AI:   understands this is a follow-up, calls filter_segments(type="arc", arc_angle_deg__gt=80) → 3 arcs
      "Narrowing down: 3 arcs exceed 80°..."

User: "does it contain straight sections?"
AI:   calls filter_segments(type="straight") → 7 straights
      "Yes, there are 7 straight sections..."
```

The chat history (stored in `chat.jsonl`) includes tool call metadata so Gemini always knows what was previously computed.

**4. System Prompt Engineering**

The system prompt (`ai/prompts.py`) provides Gemini with:
- **Role definition**: "You are a 3D geometry analysis assistant"
- **Domain knowledge**: Segment types, fields, and their physical meanings
- **Live segment data**: Injected at query time via `build_segments_context()`
- **Follow-up instructions**: Explicit rules for handling contextual queries
- **Response style**: Natural language, bullet points, specific measurements, no raw data dumps

**5. Configuration**

| Setting | Value |
|---------|-------|
| Model | `gemini-3-flash-preview` |
| SDK | `google-genai` (Python) |
| Temperature | `0.1` (deterministic, factual) |
| Max tool rounds | `8` per query |
| Auth | API key via `.env` file |

### Why Gemini 3 Flash?

- **Function calling**: Native support for tool/function declarations — Gemini decides *which* tools to call and *how* to combine them
- **Multi-turn**: Can chain 2-5 tool calls to answer complex queries in a single interaction
- **Context window**: Handles 20+ messages of conversation history with tool call metadata
- **Speed**: Flash variant is fast enough for real-time WebSocket streaming (typically 2-5 seconds per query)
- **Structured output**: Returns clean function call parameters that map directly to our DSL engine

---

## Data Flow: Complete User Journey

```
1. LOGIN
   User enters email → POST /auth/login → uid created/returned → stored in localStorage

2. UPLOAD
   User uploads .msh file → POST /upload
   → geometry_ingest.py: read mesh → extract centerline → store files
   → returns session_id

3. SEGMENT (via WebSocket)
   Frontend connects WS /ws/{uid}/{sid}
   → sends {"type": "upload_and_segment"}
   → Backend streams progress:
      ← {"type": "progress", "step": "segmenting", ...}
      ← {"type": "progress", "step": "downsampling", ...}
      ← {"type": "progress", "step": "mapping_faces", ...}
      ← {"type": "result", "data": {segments: [...], summary: {...}}}
   → Frontend renders 3D mesh + colored segments

4. QUERY (via WebSocket)
   User types: "how many arcs above 90°?"
   → sends {"type": "query", "query": "..."}
   → Backend:
      a. Loads chat history (for follow-up context)
      b. Saves user message to chat.jsonl
      c. Sends to Gemini 3 Flash with segment data + tools
      d. Gemini calls filter_segments(type="arc", arc_angle_deg__gt=90)
      e. Tool result fed back to Gemini
      f. Gemini generates natural language answer
      g. Answer + tool calls saved to chat.jsonl
   ← {"type": "result", "data": {answer: "...", highlight_ids: [...]}}
   → Frontend displays answer + highlights segments on 3D view

5. FOLLOW-UP
   User types: "and above 150°?"
   → Same flow, but Gemini sees previous tool calls in history
   → Correctly interprets as "filter arcs above 150° (updating previous threshold)"

6. SESSION RESTORE
   User returns later → GET /chat/{uid}/{sid} + GET /segments/{uid}/{sid}
   → Chat history + segment results restored in frontend
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI** | Gemini 3 Flash (`google-genai`) | NL understanding, function calling, multi-turn reasoning |
| **API** | FastAPI + Uvicorn | REST + WebSocket server |
| **Geometry** | gmsh, meshio, VTK, scipy | CAD ingest, centerline extraction |
| **Segmentation** | Custom (Menger curvature, hysteresis) | Centerline → typed segments |
| **ML Embedding** | PyTorch + PyG (GATv2) | Optional shape encoding |
| **Storage** | JSON, JSONL, pickle, NPY (local filesystem) | Per-user, per-session data |
| **Notifications** | python-telegram-bot | Real-time alerts on queries |
| **Frontend** | React + VTK.js | 3D visualization + chat |

---

## File Structure

```
deploy/
├── app.py                      # FastAPI server (routes, WebSocket, auth)
├── centerline_segmenter.py     # Segmentation algorithm (curvature, hysteresis)
├── model.py                    # ShapeEncoder (GATv2) model definition
├── inference.py                # ML inference utilities
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
│
├── ai/                         # AI layer (Gemini integration)
│   ├── gemini.py               # Gemini client, function calling, multi-turn
│   ├── prompts.py              # System prompts, context builders
│   └── test_gemini.py          # Integration tests
│
├── dsl/                        # Domain-Specific Language engine
│   └── engine.py               # SegmentQueryEngine + Gemini-powered query_smart()
│
├── services/                   # Backend services
│   ├── geometry_ingest.py      # File parsing, centerline extraction
│   ├── segmentation.py         # Pipeline orchestrator (segment → downsample → embed → store)
│   └── session.py              # User registry + session management
│
├── weights/                    # Pre-trained model weights
│   ├── encoder.pt
│   └── meta.json
│
├── mesh/                       # Test pipe meshes (5 types)
│   ├── simple_bend.msh
│   ├── s_curve.msh
│   ├── u_bend.msh
│   ├── complex_network.msh
│   └── t_junction.msh
│
└── data/                       # Runtime data (per-user, per-session)
    ├── users.json
    └── {uid}/{session_id}/...
```
