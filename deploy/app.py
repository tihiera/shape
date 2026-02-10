#!/usr/bin/env python3
"""
app.py — FastAPI server for centerline segmentation + query.

Auth:
    POST /auth/login              -> Email login (creates user if new)
    GET  /auth/me?uid=...         -> Get user info + session history

Pipeline:
    POST /upload                  -> Ingest STEP/JSON file
    POST /segment                 -> Run segmentation on uploaded graph
    POST /query                   -> NL -> AI -> execute -> result
    POST /dsl                     -> Execute DSL command directly

Sessions:
    GET  /sessions/{uid}          -> List sessions
    GET  /session/{uid}/{sid}     -> Get full session (results + chat)
    GET  /chat/{uid}/{sid}        -> Get chat messages only
    GET  /segments/{uid}/{sid}    -> Get segmentation results only

Streaming:
    WS   /ws/{uid}/{session_id}   -> Streaming pipeline + chat

Start:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env file (must be before any code that reads env vars)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"[app] 📄 Loaded .env from {_env_file}")
    else:
        load_dotenv()
except ImportError:
    pass

import numpy as np
from pydantic import BaseModel, Field

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from centerline_segmenter import (
    split_centerline_graph,
    downsample_segment_for_model,
    estimate_arc_angle,
    SegmentParams,
)
from services.session import (
    login_or_create_user, get_user_by_uid, validate_uid,
    create_session, get_session_dir, list_sessions,
    append_chat, load_chat,
)
from services.geometry_ingest import ingest_file, read_centerline_json
from services.segmentation import run_full_pipeline, load_session_results
from dsl.engine import SegmentQueryEngine, query_smart
from ai.prompts import format_step_explanation, summarize_pipeline_intro
from telegram import Bot
from telegram.constants import ParseMode


# ══════════════════════════════════════════════════════════════════════
# ML MODEL (optional)
# ══════════════════════════════════════════════════════════════════════

_MODEL = None
_DEVICE = None

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")

bot = Bot(token=TOKEN)
async def send_telegram_message(message: str):
    try:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
        print("[telegram] ✅ Message sent")
    except Exception as e:
        print(f"[telegram] ❌ Error sending message: {e}")
        return False
    return True


def _tg_notify(message: str):
    """Fire-and-forget telegram notification (safe from sync or async context)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(send_telegram_message(message))
        else:
            loop.run_until_complete(send_telegram_message(message))
    except Exception as e:
        print(f"[telegram] ❌ Could not schedule message: {e}")


def _load_ml_model():
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return

    ckpt = os.environ.get("MODEL_CHECKPOINT", "weights/encoder.pt")
    if not Path(ckpt).exists():
        print(f"[model] checkpoint not found at {ckpt}, embedding disabled")
        return
    try:
        from inference import load_model, pick_device
        _DEVICE = pick_device(os.environ.get("DEVICE", "auto"))
        _MODEL, _ = load_model(ckpt, _DEVICE)
        print(f"[model] loaded on {_DEVICE}")
    except ImportError:
        print("[model] torch not installed, embedding disabled")
    except Exception as e:
        print(f"[model] load failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _require_uid(uid: str):
    """Validate that a uid exists. Raises 401 if not."""
    if not uid or not validate_uid(uid):
        raise HTTPException(401, "Invalid or unknown uid. Call POST /auth/login first.")


# ══════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Centerline Segmentation API",
    description=(
        "Ingest STEP/JSON geometry → extract centerline → "
        "segment (junction/straight/arc/corner) → downsample → "
        "embed → query via NL/DSL. WebSocket streaming for live updates."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _load_ml_model()


# ══════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")


class SegmentRequest(BaseModel):
    nodes: List[List[float]] = Field(..., description="Nx3 coordinates")
    edges: List[List[int]] = Field(..., description="Edge pairs [i, j]")
    target_step: float = Field(1.0)
    downsample_nodes: int = Field(16)
    embed: bool = Field(False)
    uid: str = Field(...)
    session_id: Optional[str] = Field(None)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query or DSL command")
    uid: str = Field(...)
    session_id: str = Field(...)


class DSLRequest(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    uid: str = Field(...)
    session_id: str = Field(...)


# ══════════════════════════════════════════════════════════════════════
# ROUTES: HEALTH
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "Centerline Segmentation API",
        "version": "2.0.0",
        "model_loaded": _MODEL is not None,
        "segment_types": ["junction", "straight", "arc", "corner"],
        "endpoints": {
            "POST /auth/login": "Login with email (creates user if new)",
            "GET /auth/me?uid=...": "Get user info + sessions",
            "POST /upload": "Upload MSH/STEP/JSON file",
            "POST /segment": "Segment a graph (inline JSON)",
            "POST /query": "NL query on segments",
            "POST /dsl": "Execute DSL command directly",
            "GET /mesh/{uid}/{session_id}": "Get surface mesh (vertices + faces)",
            "GET /chat/{uid}/{session_id}": "Get saved chat messages",
            "GET /segments/{uid}/{session_id}": "Get segmentation results",
            "WS /ws/{uid}/{session_id}": "WebSocket streaming pipeline",
            "GET /sessions/{uid}": "List user sessions",
            "GET /session/{uid}/{session_id}": "Get full session (results + chat)",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _MODEL is not None}


# ══════════════════════════════════════════════════════════════════════
# ROUTES: AUTH (static email-based, no password)
# ══════════════════════════════════════════════════════════════════════

@app.post("/auth/login")
def auth_login(req: LoginRequest):
    """
    Static login with email. No password, no auth service.

    Frontend flow:
      1. Check localStorage for stored email + uid
      2. If missing → show login page → user enters email
      3. POST /auth/login {"email": "alice@example.com"}
      4. Backend:
         - If email exists → update last_login, return uid + sessions
         - If email is new → create uid + folder, return uid + empty sessions
      5. Frontend stores {email, uid} in localStorage
      6. All subsequent requests use uid

    Returns:
        {
            "uid": "a1b2c3d4e5f6...",
            "email": "alice@example.com",
            "is_new": true/false,
            "created_at": "...",
            "last_login": "...",
            "sessions": [...]
        }
    """
    try:
        user = login_or_create_user(req.email)
    except ValueError as e:
        raise HTTPException(400, str(e))

    sessions = list_sessions(user["uid"])

    return {
        **user,
        "sessions": sessions,
    }


@app.get("/auth/me")
def auth_me(uid: str):
    """
    Get user info by uid. Used by frontend on page load to verify
    the stored uid is still valid.

    Returns user info + session list, or 404 if uid unknown.
    """
    user = get_user_by_uid(uid)
    if user is None:
        raise HTTPException(404, "User not found. Please login again.")

    sessions = list_sessions(uid)

    return {
        **user,
        "sessions": sessions,
    }


# ══════════════════════════════════════════════════════════════════════
# ROUTES: UPLOAD
# ══════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    uid: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """Upload a STEP or JSON file. Creates a session and ingests the geometry."""
    _require_uid(uid)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".step", ".stp", ".json", ".msh"):
        raise HTTPException(400, f"Unsupported file type: {suffix}. Use .msh, .step, .stp, or .json")

    meta = create_session(uid, session_id)
    session_dir = Path(meta["path"])

    upload_path = session_dir / "upload" / file.filename
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        ingest_result = ingest_file(str(upload_path), session_dir)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")

    return {"session": meta, "ingest": ingest_result}


# ══════════════════════════════════════════════════════════════════════
# ROUTES: SEGMENT (inline JSON)
# ══════════════════════════════════════════════════════════════════════

@app.post("/segment")
def segment_inline(data: SegmentRequest):
    """Segment a graph provided inline as JSON."""
    _require_uid(data.uid)

    nodes = np.asarray(data.nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise HTTPException(400, "nodes must be Nx3")
    edges = [(int(e[0]), int(e[1])) for e in data.edges]

    meta = create_session(data.uid, data.session_id)
    session_dir = Path(meta["path"])

    result = run_full_pipeline(
        nodes, edges, session_dir,
        target_step=data.target_step,
        target_nodes=data.downsample_nodes,
        model=_MODEL if data.embed else None,
        device=_DEVICE,
    )

    return {"session": meta, **result}


# ══════════════════════════════════════════════════════════════════════
# ROUTES: QUERY (NL -> AI -> execute) 
# ══════════════════════════════════════════════════════════════════════

@app.post("/query")
def query_segments(req: QueryRequest, background_tasks: BackgroundTasks):
    """NL query on session segments. Uses Gemini AI with function calling."""
    _require_uid(req.uid)

    session_dir = get_session_dir(req.uid, req.session_id)
    if not session_dir.exists():
        raise HTTPException(404, f"Session not found: {req.session_id}")

    results = load_session_results(session_dir)
    if results is None:
        raise HTTPException(400, "No segmentation results. Run /segment first.")

    # Load chat BEFORE appending user message — so AI sees conversation context
    chat_history = load_chat(session_dir)

    # Append user message to persistent log
    append_chat(session_dir, "user", req.query)

    # Notify: user asked a question
    user = get_user_by_uid(req.uid)
    email = user.get("email", req.uid) if user else req.uid
    _tg_notify(
        f"❓ *Question* from `{email}`\n"
        f"• Session: `{req.session_id}`\n"
        f"• Query: _{req.query}_"
    )

    ai_result = query_smart(
        user_message=req.query,
        segments=results["segments"],
        chat_history=chat_history,
    )

    # Save assistant response with tool_calls for follow-up context
    append_chat(session_dir, "assistant", ai_result.get("answer", ""),
                extra={
                    "tool_calls": ai_result.get("tool_calls", []),
                    "mode": ai_result.get("mode", "unknown"),
                })

    # Notify: answer generated
    answer_preview = ai_result.get("answer", "")[:200]
    _tg_notify(
        f"✅ *Answer* for `{email}`\n"
        f"• Mode: `{ai_result.get('mode', 'unknown')}`\n"
        f"• Answer: _{answer_preview}_"
    )

    return {
        "query": req.query,
        "answer": ai_result.get("answer", ""),
        "tool_calls": ai_result.get("tool_calls", []),
        "highlight_ids": ai_result.get("highlight_ids", []),
        "mode": ai_result.get("mode", "unknown"),
        "fallback_reason": ai_result.get("fallback_reason"),
    }


@app.post("/dsl")
def execute_dsl(req: DSLRequest):
    """Execute a DSL command directly (no NL parsing)."""
    _require_uid(req.uid)

    session_dir = get_session_dir(req.uid, req.session_id)
    if not session_dir.exists():
        raise HTTPException(404, f"Session not found: {req.session_id}")

    results = load_session_results(session_dir)
    if results is None:
        raise HTTPException(400, "No segmentation results.")

    engine = SegmentQueryEngine(results["segments"])
    return engine.execute({"action": req.action, "params": req.params})


# ══════════════════════════════════════════════════════════════════════
# ROUTES: SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

@app.get("/sessions/{uid}")
def get_sessions(uid: str):
    _require_uid(uid)
    return {"sessions": list_sessions(uid)}


@app.get("/session/{uid}/{session_id}")
def get_session(uid: str, session_id: str):
    _require_uid(uid)

    session_dir = get_session_dir(uid, session_id)
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    results = load_session_results(session_dir)
    chat = load_chat(session_dir)

    return {
        "session_id": session_id,
        "uid": uid,
        "results": results,
        "chat_history": chat,
    }


@app.get("/chat/{uid}/{session_id}")
def get_chat_history(uid: str, session_id: str):
    """
    Return the saved chat messages for a session.

    Returns:
        [
            {"role": "user", "content": "...", "timestamp": "..."},
            {"role": "assistant", "content": "...", "timestamp": "...", "tool_calls": [...], "mode": "ai"},
            ...
        ]

    Returns 404 if session or chat doesn't exist yet.
    """
    _require_uid(uid)

    session_dir = get_session_dir(uid, session_id)
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    chat_path = session_dir / "chat.jsonl"
    if not chat_path.exists():
        raise HTTPException(404, "No chat history for this session")

    chat = load_chat(session_dir)
    return chat


@app.get("/segments/{uid}/{session_id}")
def get_segments(uid: str, session_id: str):
    """
    Return pre-computed segmentation results for a session.

    Returns:
        {
            "segments": [ { "segment_id": 0, "type": "straight", ... }, ... ],
            "summary": { "total_segments": 13, "counts_by_type": {...}, ... }
        }

    Returns 404 if session doesn't exist or hasn't been segmented yet.
    """
    _require_uid(uid)

    session_dir = get_session_dir(uid, session_id)
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    results = load_session_results(session_dir)
    if results is None:
        raise HTTPException(404, "No segmentation results for this session")

    return results


# ══════════════════════════════════════════════════════════════════════
# ROUTES: MESH (surface mesh for frontend 3D rendering)
# ══════════════════════════════════════════════════════════════════════

@app.get("/mesh/{uid}/{session_id}")
def get_mesh(uid: str, session_id: str):
    """
    Return the triangulated surface mesh as JSON for VTK.js rendering.

    Returns:
        {
            "vertices": [[x, y, z], ...],   # every point in the surface mesh
            "faces": [[i, j, k], ...],       # triangle indices (0-based)
        }
    """
    _require_uid(uid)

    session_dir = get_session_dir(uid, session_id)
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")

    surface_path = session_dir / "mesh_surface.json"
    if not surface_path.exists():
        raise HTTPException(404, "No surface mesh available. Upload a .msh or .step file first.")

    data = json.loads(surface_path.read_text())

    # Sanitize: replace NaN/Inf with 0 (safety net)
    verts = data.get("vertices", [])
    clean_verts = []
    for v in verts:
        clean_verts.append([
            0.0 if (x != x or abs(x) == float('inf')) else x
            for x in v
        ])
    data["vertices"] = clean_verts

    return data


# ══════════════════════════════════════════════════════════════════════
# WEBSOCKET: STREAMING PIPELINE + CHAT
# ══════════════════════════════════════════════════════════════════════

@app.websocket("/ws/{uid}/{session_id}")
async def websocket_pipeline(ws: WebSocket, uid: str, session_id: str):
    """
    WebSocket for streaming pipeline updates + interactive chat.

    Client sends:
        {"type": "segment", "nodes": [...], "edges": [...], ...}
        {"type": "upload_and_segment", "target_step": 1.0}
        {"type": "query", "query": "show all arcs > 30 degrees"}
        {"type": "dsl", "action": "filter", "params": {...}}
        {"type": "highlight", "segment_ids": [1, 3]}

    Server streams back:
        {"type": "progress", "step": "...", "detail": {...}}
        {"type": "result", "data": {...}}
        {"type": "error", "message": "..."}
    """
    await ws.accept()

    # Validate uid
    if not validate_uid(uid):
        await ws.send_json({"type": "error", "message": "Invalid uid. Login first."})
        await ws.close(code=4001)
        return

    meta = create_session(uid, session_id)
    session_dir = Path(meta["path"])

    async def send(msg_type: str, **kwargs):
        await ws.send_json({"type": msg_type, **kwargs})

    def sync_progress(step: str, detail: dict):
        _progress_queue.append({"type": "progress", "step": step, "detail": detail})

    _progress_queue: list = []

    try:
        await send("connected", session=meta)

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send("error", message="Invalid JSON")
                continue

            msg_type = msg.get("type", "")

            # ── SEGMENT (inline) ──
            if msg_type == "segment":
                try:
                    nodes = np.asarray(msg["nodes"], dtype=np.float64)
                    edges = [(int(e[0]), int(e[1])) for e in msg["edges"]]
                    target_step = msg.get("target_step", 1.0)
                    target_nodes = msg.get("downsample_nodes", 16)
                    do_embed = msg.get("embed", False)

                    _progress_queue.clear()
                    result = await asyncio.to_thread(
                        run_full_pipeline,
                        nodes, edges, session_dir,
                        target_step=target_step,
                        target_nodes=target_nodes,
                        model=_MODEL if do_embed else None,
                        device=_DEVICE,
                        on_progress=sync_progress,
                    )

                    for p in _progress_queue:
                        await ws.send_json(p)
                    _progress_queue.clear()

                    await send("result", data=result)

                except Exception as e:
                    await send("error", message=str(e), traceback=traceback.format_exc())

            # ── SEGMENT from uploaded file ──
            elif msg_type == "upload_and_segment":
                try:
                    cl_path = session_dir / "centerline.json"
                    if not cl_path.exists():
                        await send("error", message="No uploaded file. Use /upload first.")
                        continue

                    nodes, edges = read_centerline_json(str(cl_path))
                    target_step = msg.get("target_step", 1.0)
                    target_nodes = msg.get("downsample_nodes", 16)
                    do_embed = msg.get("embed", False)

                    _progress_queue.clear()
                    result = await asyncio.to_thread(
                        run_full_pipeline,
                        nodes, edges, session_dir,
                        target_step=target_step,
                        target_nodes=target_nodes,
                        model=_MODEL if do_embed else None,
                        device=_DEVICE,
                        on_progress=sync_progress,
                    )

                    for p in _progress_queue:
                        await ws.send_json(p)
                    _progress_queue.clear()

                    await send("result", data=result)

                except Exception as e:
                    await send("error", message=str(e))

            # ── QUERY (NL -> AI with function calling) ──
            elif msg_type == "query":
                try:
                    query_text = msg.get("query", "")

                    # 1. Load chat history BEFORE appending the new user message
                    #    so the AI sees the full conversation context for follow-ups
                    chat_history = load_chat(session_dir)

                    # 2. Now append the user message to the persistent chat log
                    append_chat(session_dir, "user", query_text)

                    # Notify: user asked a question
                    ws_user = get_user_by_uid(uid)
                    ws_email = ws_user.get("email", uid) if ws_user else uid
                    await send_telegram_message(
                        f"❓ *Question* from `{ws_email}`\n"
                        f"• Session: `{session_id}`\n"
                        f"• Query: _{query_text}_"
                    )

                    await send("progress", step="parsing_query",
                               detail={"query": query_text},
                               explanation=format_step_explanation("parsing_query", {"query": query_text}))

                    results = load_session_results(session_dir)
                    if results is None:
                        await send("error", message="No segmentation results. Segment first.")
                        continue

                    _ai_progress: list = []

                    def ai_on_step(step: str, detail: dict):
                        _ai_progress.append({
                            "type": "progress",
                            "step": step,
                            "detail": detail,
                            "explanation": format_step_explanation(step, detail),
                        })

                    ai_result = await asyncio.to_thread(
                        query_smart,
                        query_text,
                        results["segments"],
                        chat_history,
                        ai_on_step,
                    )

                    for p in _ai_progress:
                        await ws.send_json(p)
                    _ai_progress.clear()

                    # 3. Save assistant response with tool_calls so future
                    #    follow-ups can see what was previously computed
                    append_chat(session_dir, "assistant", ai_result.get("answer", ""),
                                extra={
                                    "tool_calls": ai_result.get("tool_calls", []),
                                    "mode": ai_result.get("mode"),
                                })

                    # Notify: answer generated
                    answer_preview = ai_result.get("answer", "")[:200]
                    await send_telegram_message(
                        f"✅ *Answer* for `{ws_email}`\n"
                        f"• Mode: `{ai_result.get('mode', 'unknown')}`\n"
                        f"• Answer: _{answer_preview}_"
                    )

                    await send("result", data={
                        "query": query_text,
                        "answer": ai_result.get("answer", ""),
                        "tool_calls": ai_result.get("tool_calls", []),
                        "highlight_ids": ai_result.get("highlight_ids", []),
                        "mode": ai_result.get("mode", "unknown"),
                        "fallback_reason": ai_result.get("fallback_reason"),
                    })

                except Exception as e:
                    await send("error", message=str(e))

            # ── DSL (direct) ──
            elif msg_type == "dsl":
                try:
                    results = load_session_results(session_dir)
                    if results is None:
                        await send("error", message="No segmentation results.")
                        continue

                    engine = SegmentQueryEngine(results["segments"])
                    dsl_result = engine.execute({
                        "action": msg.get("action", ""),
                        "params": msg.get("params", {}),
                    })
                    await send("result", data=dsl_result)

                except Exception as e:
                    await send("error", message=str(e))

            # ── HIGHLIGHT ──
            elif msg_type == "highlight":
                try:
                    results = load_session_results(session_dir)
                    if results is None:
                        await send("error", message="No segmentation results.")
                        continue

                    engine = SegmentQueryEngine(results["segments"])
                    highlight_data = engine.highlight_segments(msg.get("segment_ids", []))
                    await send("result", data=highlight_data)

                except Exception as e:
                    await send("error", message=str(e))

            else:
                await send("error", message=f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await send("error", message=str(e))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
