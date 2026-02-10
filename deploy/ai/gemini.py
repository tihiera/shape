"""
ai/gemini.py
────────────
Gemini client wrapper using google-genai.
Handles function calling, streaming, and conversation context.

Uses Gemini 2.5 Flash (gemini-3-flash-preview).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Load .env from deploy/ directory
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_file if _env_file.exists() else None)
except ImportError:
    pass

from google import genai
from google.genai import types

# ══════════════════════════════════════════════════════════════════════
# CLIENT SINGLETON
# ══════════════════════════════════════════════════════════════════════

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. "
                "Create a .env file in deploy/ with: GEMINI_API_KEY=your_key\n"
                "Or export it: export GEMINI_API_KEY=your_key"
            )
        _client = genai.Client(api_key=api_key)
    return _client


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")


# ══════════════════════════════════════════════════════════════════════
# TOOL / FUNCTION DEFINITIONS FOR GEMINI
# ══════════════════════════════════════════════════════════════════════

SEGMENT_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="list_segments",
            description="List all segments with their type, length, angle. Use when user wants to see all segments or get an overview.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "fields": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="STRING"),
                        description="Optional list of fields to return. If empty, returns all summary fields.",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="filter_segments",
            description="Filter segments by conditions. Supports type, angle, length, curvature with comparison operators.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "type": types.Schema(type="STRING", description="Segment type: junction, straight, arc, corner"),
                    "arc_angle_deg__gt": types.Schema(type="NUMBER", description="Arc angle greater than (degrees)"),
                    "arc_angle_deg__lt": types.Schema(type="NUMBER", description="Arc angle less than (degrees)"),
                    "arc_angle_deg__gte": types.Schema(type="NUMBER", description="Arc angle >= (degrees)"),
                    "arc_angle_deg__lte": types.Schema(type="NUMBER", description="Arc angle <= (degrees)"),
                    "corner_angle_deg__gt": types.Schema(type="NUMBER", description="Corner angle greater than"),
                    "corner_angle_deg__lt": types.Schema(type="NUMBER", description="Corner angle less than"),
                    "length__gt": types.Schema(type="NUMBER", description="Length greater than"),
                    "length__lt": types.Schema(type="NUMBER", description="Length less than"),
                    "radius_est__gt": types.Schema(type="NUMBER", description="Radius greater than"),
                    "radius_est__lt": types.Schema(type="NUMBER", description="Radius less than"),
                    "mean_curvature__gt": types.Schema(type="NUMBER", description="Mean curvature greater than"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="count_segments",
            description="Count segments, optionally filtered by type or conditions.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "type": types.Schema(type="STRING", description="Filter by type: junction, straight, arc, corner"),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="sum_field",
            description="Sum a numeric field across segments (e.g. total length of all arcs).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "field": types.Schema(type="STRING", description="Field to sum: length, arc_angle_deg, mean_curvature"),
                    "type": types.Schema(type="STRING", description="Optional: filter by segment type first"),
                },
                required=["field"],
            ),
        ),
        types.FunctionDeclaration(
            name="group_by",
            description="Group segments by a field and get counts and total lengths per group.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "field": types.Schema(type="STRING", description="Field to group by: type, arc_angle_deg, etc."),
                },
                required=["field"],
            ),
        ),
        types.FunctionDeclaration(
            name="topk_by",
            description="Get top-K segments sorted by a numeric field (e.g. top 3 arcs by angle).",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "field": types.Schema(type="STRING", description="Field to sort by"),
                    "k": types.Schema(type="INTEGER", description="Number of results (default 3)"),
                    "ascending": types.Schema(type="BOOLEAN", description="Sort ascending (default false = descending)"),
                },
                required=["field"],
            ),
        ),
        types.FunctionDeclaration(
            name="describe_segment",
            description="Get full details of a specific segment by its ID.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "segment_id": types.Schema(type="INTEGER", description="The segment ID to describe"),
                },
                required=["segment_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="highlight_segments",
            description="Highlight specific segments in the 3D viewer. Returns geometry data for VTK.js overlay.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "segment_ids": types.Schema(
                        type="ARRAY",
                        items=types.Schema(type="INTEGER"),
                        description="List of segment IDs to highlight",
                    ),
                },
                required=["segment_ids"],
            ),
        ),
    ]),
]


# ══════════════════════════════════════════════════════════════════════
# CALL GEMINI WITH FUNCTION CALLING (MULTI-TURN)
# ══════════════════════════════════════════════════════════════════════

def call_gemini(
    user_message: str,
    segments_context: str,
    chat_history: List[Dict[str, str]],
    execute_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    on_step: Optional[Callable[[str, Any], None]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a user query to Gemini with function calling.

    Gemini can:
      - Call one or multiple tools in sequence
      - Chain tool results to answer complex queries
      - Provide a natural language summary of results

    Args:
        user_message:     The user's natural language query
        segments_context: JSON summary of available segments (compact)
        chat_history:     Previous chat messages [{role, content}, ...]
        execute_fn:       Callback to execute a tool: (tool_name, params) -> result
        on_step:          Optional callback for streaming progress
        system_prompt:    Override system prompt

    Returns:
        {
            "answer": "natural language response",
            "tool_calls": [{"tool": name, "params": {...}, "result": {...}}, ...],
            "highlight_ids": [...]  # if any highlight was requested
        }
    """
    print(f"[gemini] 🔵 call_gemini() invoked — model={MODEL}")
    print(f"[gemini]   query: {user_message!r}")
    print(f"[gemini]   chat_history: {len(chat_history)} messages")

    client = get_client()
    print(f"[gemini] ✅ client obtained (API key is set)")

    if system_prompt is None:
        from ai.prompts import build_system_prompt
        system_prompt = build_system_prompt(segments_context)

    # Build conversation history — include tool call context for follow-ups
    contents = []
    for msg in chat_history[-20:]:  # last 20 messages for context
        role = "user" if msg.get("role") == "user" else "model"
        text = msg.get("content", "")

        # For assistant messages, also include a summary of tool calls
        # so Gemini knows what was previously computed
        if role == "model":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tool_summary_parts = []
                for tc in tool_calls:
                    tool_name = tc.get("tool", "?")
                    tool_params = tc.get("params", {})
                    tool_result = tc.get("result", {})
                    # Compact summary of what was called and returned
                    tool_summary_parts.append(
                        f"[Called {tool_name}({json.dumps(tool_params)}) → "
                        f"{json.dumps(tool_result, default=str)[:500]}]"
                    )
                tool_context = "\n".join(tool_summary_parts)
                text = f"{text}\n\n---\nPrevious tool calls:\n{tool_context}"

        if text.strip():
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            ))

    # Add current user message
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    ))

    # Call Gemini with tools
    tool_calls_log = []
    highlight_ids = []
    max_rounds = 8  # max tool-call rounds to prevent infinite loops

    for round_idx in range(max_rounds):
        print(f"[gemini] 🔄 Round {round_idx + 1}/{max_rounds} — calling Gemini API...")
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=SEGMENT_TOOLS,
                temperature=0.1,
            ),
        )
        print(f"[gemini] ✅ Got response from Gemini")

        # Check if model wants to call functions
        candidate = response.candidates[0]
        parts = candidate.content.parts

        has_function_call = any(p.function_call for p in parts if p)
        if not has_function_call:
            # Model is done — extract text response
            text_parts = [p.text for p in parts if p and p.text]
            answer = "\n".join(text_parts) if text_parts else "Done."
            print(f"[gemini] 💬 Final answer ({len(answer)} chars): {answer[:200]}...")
            break

        # Process function calls
        function_response_parts = []
        for part in parts:
            if not part or not part.function_call:
                continue

            fc = part.function_call
            tool_name = fc.name
            tool_params = dict(fc.args) if fc.args else {}

            print(f"[gemini] 🔧 Tool call: {tool_name}({json.dumps(tool_params, default=str)})")

            if on_step:
                on_step("tool_call", {"tool": tool_name, "params": tool_params, "round": round_idx})

            # Execute the tool
            try:
                result = execute_fn(tool_name, tool_params)
                print(f"[gemini]   → result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            except Exception as e:
                result = {"error": str(e)}
                print(f"[gemini]   → ❌ tool error: {e}")

            tool_calls_log.append({
                "tool": tool_name,
                "params": tool_params,
                "result": result,
            })

            # Collect highlight IDs
            if tool_name == "highlight_segments":
                ids = tool_params.get("segment_ids", [])
                highlight_ids.extend(ids)

            # Build function response
            function_response_parts.append(types.Part.from_function_response(
                name=tool_name,
                response=result,
            ))

        # Add model's function call to conversation
        contents.append(candidate.content)
        # Add function responses
        contents.append(types.Content(
            role="user",
            parts=function_response_parts,
        ))
    else:
        answer = "Reached maximum tool call rounds."

    return {
        "answer": answer,
        "tool_calls": tool_calls_log,
        "highlight_ids": highlight_ids,
    }


# ══════════════════════════════════════════════════════════════════════
# STREAMING VARIANT
# ══════════════════════════════════════════════════════════════════════

async def call_gemini_streaming(
    user_message: str,
    segments_context: str,
    chat_history: List[Dict[str, str]],
    execute_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    on_step: Optional[Callable[[str, Any], None]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Async wrapper around call_gemini for use in WebSocket handlers.
    Runs the sync Gemini call in a thread.
    """
    import asyncio
    return await asyncio.to_thread(
        call_gemini,
        user_message, segments_context, chat_history,
        execute_fn, on_step, system_prompt,
    )
