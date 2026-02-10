"""
dsl/engine.py
─────────────
DSL engine for querying segments — powered by Gemini AI.

The AI (Gemini with function calling) can:
  - Chain multiple tool calls to answer complex queries
  - Break "find all arcs > 30° and highlight the longest one" into filter + topk + highlight
  - Provide natural language explanations of results
  - Handle follow-up questions using conversation history
  - Handle ambiguous queries intelligently
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv("../.env")

# ══════════════════════════════════════════════════════════════════════
# DSL COMMAND EXECUTOR (tools that Gemini calls via function calling)
# ══════════════════════════════════════════════════════════════════════

class SegmentQueryEngine:
    """Execute DSL commands against a list of segment dicts."""

    def __init__(self, segments: List[Dict[str, Any]]):
        self.segments = segments

    def list_segments(self, fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all segments, optionally projecting to specific fields."""
        if fields is None:
            return [_summary(s) for s in self.segments]
        return [{k: s.get(k) for k in fields} for s in self.segments]

    def filter_segments(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Filter segments by field conditions.

        Supports:
            field=value         -> exact match
            field__gt=value     -> greater than
            field__gte=value    -> greater than or equal
            field__lt=value     -> less than
            field__lte=value    -> less than or equal
            field__ne=value     -> not equal
            field__in=[values]  -> in list
        """
        result = self.segments
        for key, value in kwargs.items():
            if value is None:
                continue
            result = _apply_filter(result, key, value)
        return [_summary(s) for s in result]

    def count_segments(self, **filter_kwargs) -> Dict[str, Any]:
        """Count segments, optionally filtered."""
        clean = {k: v for k, v in filter_kwargs.items() if v is not None}
        filtered = self.filter_segments(**clean) if clean else self.segments
        return {"count": len(filtered)}

    def sum_field(self, field: str, **filter_kwargs) -> Dict[str, Any]:
        """Sum a numeric field across segments."""
        clean = {k: v for k, v in filter_kwargs.items() if v is not None}
        filtered = self.filter_segments(**clean) if clean else self.segments
        values = [s.get(field, 0) for s in filtered if isinstance(s.get(field), (int, float))]
        return {"field": field, "sum": round(sum(values), 4), "count": len(values)}

    def group_by(self, field: str) -> Dict[str, Any]:
        """Group segments by a field, with counts and stats."""
        groups: Dict[Any, List[Dict]] = {}
        for s in self.segments:
            key = s.get(field, "unknown")
            groups.setdefault(key, []).append(s)

        result = {}
        for key, segs in sorted(groups.items(), key=lambda x: str(x[0])):
            lengths = [s.get("length", 0) for s in segs]
            result[str(key)] = {
                "count": len(segs),
                "total_length": round(sum(lengths), 3),
                "segment_ids": [s["segment_id"] for s in segs],
            }
        return result

    def topk_by(self, field: str, k: int = 3, ascending: bool = False) -> List[Dict[str, Any]]:
        """Top-k segments sorted by a numeric field."""
        valid = [s for s in self.segments if isinstance(s.get(field), (int, float))]
        sorted_segs = sorted(valid, key=lambda s: s.get(field, 0), reverse=not ascending)
        return [_summary(s) for s in sorted_segs[:k]]

    def describe_segment(self, segment_id: int) -> Optional[Dict[str, Any]]:
        """Full detail of a single segment."""
        for s in self.segments:
            if s["segment_id"] == segment_id:
                return s
        return None

    def highlight_segments(self, segment_ids: List[int]) -> Dict[str, Any]:
        """
        Return highlight data for the frontend (VTK.js overlay).
        Includes original_node_ids + downsampled geometry.
        """
        highlighted = []
        for s in self.segments:
            if s["segment_id"] in segment_ids:
                highlighted.append({
                    "segment_id": s["segment_id"],
                    "type": s["type"],
                    "original_node_ids": s.get("original_node_ids", []),
                    "downsampled_nodes": s.get("downsampled_nodes", []),
                    "downsampled_edges": s.get("downsampled_edges", []),
                    "arc_angle_deg": s.get("arc_angle_deg", 0),
                    "corner_angle_deg": s.get("corner_angle_deg", 0),
                    "length": s.get("length", 0),
                })
        return {
            "highlighted_segments": highlighted,
            "segment_ids": segment_ids,
        }

    # ── unified tool dispatcher (used by Gemini function calling) ──

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool by name. Used as callback for Gemini function calling."""
        dispatch = {
            "list_segments": lambda p: {"segments": self.list_segments(**p)},
            "filter_segments": lambda p: {"segments": self.filter_segments(**p)},
            "count_segments": lambda p: self.count_segments(**p),
            "sum_field": lambda p: self.sum_field(**p),
            "group_by": lambda p: {"groups": self.group_by(**p)},
            "topk_by": lambda p: {"segments": self.topk_by(**p)},
            "describe_segment": lambda p: {"segment": self.describe_segment(**p)},
            "highlight_segments": lambda p: self.highlight_segments(**p),
        }

        handler = dispatch.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}", "available": list(dispatch.keys())}

        try:
            return handler(params)
        except Exception as e:
            return {"error": str(e)}

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a legacy DSL command dict (backward compat)."""
        action = command.get("action", "")
        params = command.get("params", {})
        return {"action": action, "result": self.execute_tool(action, params)}


# ══════════════════════════════════════════════════════════════════════
# AI-POWERED QUERY (Gemini function calling)
# ══════════════════════════════════════════════════════════════════════

def query_with_ai(
    user_message: str,
    segments: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    on_step: Optional[Callable[[str, Any], None]] = None,
) -> Dict[str, Any]:
    """
    Process a user query using Gemini with function calling.

    Gemini sees the segment data, understands the query, and calls
    tools as needed — potentially chaining multiple calls.

    Returns:
        {
            "answer": "NL response from Gemini",
            "tool_calls": [...],
            "highlight_ids": [...],
            "mode": "ai"
        }
    """
    from ai.gemini import call_gemini
    from ai.prompts import build_segments_context

    engine = SegmentQueryEngine(segments)
    context = build_segments_context(segments)

    result = call_gemini(
        user_message=user_message,
        segments_context=context,
        chat_history=chat_history,
        execute_fn=engine.execute_tool,
        on_step=on_step,
    )

    return {**result, "mode": "ai"}


async def query_with_ai_async(
    user_message: str,
    segments: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    on_step: Optional[Callable[[str, Any], None]] = None,
) -> Dict[str, Any]:
    """Async version for WebSocket handlers."""
    from ai.gemini import call_gemini_streaming
    from ai.prompts import build_segments_context

    engine = SegmentQueryEngine(segments)
    context = build_segments_context(segments)

    result = await call_gemini_streaming(
        user_message=user_message,
        segments_context=context,
        chat_history=chat_history,
        execute_fn=engine.execute_tool,
        on_step=on_step,
    )

    return {**result, "mode": "ai"}


# ══════════════════════════════════════════════════════════════════════
# SMART QUERY DISPATCHER
# ══════════════════════════════════════════════════════════════════════

def query_smart(
    user_message: str,
    segments: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    on_step: Optional[Callable[[str, Any], None]] = None,
) -> Dict[str, Any]:
    """
    Query segments using Gemini AI with function calling.
    No rule-based fallback — always uses the real AI.
    If Gemini fails, the error is surfaced directly.
    """
    chat_history = chat_history or []

    print(f"[query_smart] 🤖 Calling AI for: {user_message!r}")
    result = query_with_ai(user_message, segments, chat_history, on_step)
    print(f"[query_smart] ✅ AI responded — mode={result.get('mode')}")
    return result




# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _summary(seg: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary of a segment (no heavy geometry)."""
    return {
        "segment_id": seg["segment_id"],
        "type": seg["type"],
        "node_count": seg.get("node_count", 0),
        "length": seg.get("length", 0),
        "arc_angle_deg": seg.get("arc_angle_deg", 0),
        "corner_angle_deg": seg.get("corner_angle_deg", 0),
        "radius_est": seg.get("radius_est", 0),
        "mean_curvature": seg.get("mean_curvature", 0),
    }


def _apply_filter(segments: List[Dict], key: str, value: Any) -> List[Dict]:
    """Apply a single filter condition."""
    if "__" in key:
        field, op = key.rsplit("__", 1)
    else:
        field, op = key, "eq"

    ops = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "in": lambda a, b: a in b,
    }

    cmp = ops.get(op)
    if cmp is None:
        return segments

    return [s for s in segments if field in s and cmp(s[field], value)]
