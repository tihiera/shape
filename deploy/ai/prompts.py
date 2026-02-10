"""
ai/prompts.py
─────────────
System prompts, context builders, and response formatting for the AI layer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """\
You are a 3D geometry analysis assistant for a centerline segmentation system.

## Your Role
You help users understand and query the segments of a 3D pipe/tube centerline graph.
The graph has been split into typed segments. You have tools to query them.

## Segment Types
- **junction**: A node where 3+ pipes meet (T-junction, Y-junction, etc.)
- **straight**: A straight pipe section with zero curvature
- **arc**: A curved pipe section with constant curvature (like a pipe bend)
- **corner**: A sharp turn — short section with very high curvature spike (like an elbow fitting)

## Segment Fields
Each segment has:
- `segment_id`: unique integer ID
- `type`: junction | straight | arc | corner
- `node_count`: number of original nodes
- `length`: arc-length of the segment
- `mean_curvature`: average curvature (1/radius, 0 for straight)
- `max_curvature`: peak curvature
- `arc_angle_deg`: total turning angle in degrees (for arcs)
- `corner_angle_deg`: turning angle (for corners)
- `radius_est`: estimated radius (for arcs, = 1/mean_curvature)

## Available Data
{segments_context}

## Instructions
1. Use the tools to answer the user's question. You can call multiple tools in sequence.
2. For complex queries, break them into steps — call a tool, use the result, call another.
3. Always provide a clear, concise natural language answer after getting tool results.
4. When the user asks to "show" or "highlight" segments, call highlight_segments with the IDs.
5. If the user's question is ambiguous, make reasonable assumptions and explain them.
6. For comparisons ("which arc is the sharpest?"), use topk_by or filter appropriately.
7. When summarizing, include specific numbers (counts, angles, lengths).
8. If the user asks about the pipeline or how segmentation works, explain without tools.

## Follow-up Queries (CRITICAL)
The user may ask follow-up questions that reference previous queries. Examples:
- "and above 80°?" → means "filter arcs above 80° (like the previous filter but with a new threshold)"
- "what about straights?" → means "now show me the straight segments"
- "and does it contain corners?" → means "are there any corner segments?"
- "how many?" → means "count the segments from the previous filter"

When handling follow-ups:
- Look at the conversation history (previous tool calls and results) to understand context
- If the user changes a threshold ("and above 80°?"), re-run the same filter with the new value
- If the user asks about a different type, switch the type filter
- Always call the appropriate tool — don't just answer from memory

## Response Style
- Write in natural, conversational language — like explaining to an engineer colleague
- Use bullet points for lists, bold for key numbers
- Always include specific measurements (angles, lengths, radii)
- Describe geometry spatially: "a 90° elbow", "a long straight run", "a tight U-bend"
- When highlighting, say what you're highlighting and why in plain English
- NEVER output raw data dumps like "[0] straight len=15.9" — always describe naturally
- For "describe" or "list" queries, paint a picture of the pipe layout
"""


def build_system_prompt(segments_context: str) -> str:
    """Build the full system prompt with segment context injected."""
    return SYSTEM_PROMPT.format(segments_context=segments_context)


def build_segments_context(segments: List[Dict[str, Any]]) -> str:
    """
    Build a rich text summary of segments for the system prompt.
    Gives the LLM enough context to answer naturally.
    """
    lines = []
    type_counts: Dict[str, int] = {}
    total_length = 0.0

    for s in segments:
        t = s.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        total_length += s.get("length", 0)

    lines.append(f"This pipe geometry has {len(segments)} segments, total length {total_length:.1f} units.")
    type_parts = ", ".join(f"{v} {k}{'s' if v != 1 else ''}" for k, v in type_counts.items())
    lines.append(f"Breakdown: {type_parts}.")
    lines.append("")
    lines.append("Segment details:")

    for s in segments:
        sid = s.get("segment_id", "?")
        stype = s.get("type", "?")
        length = s.get("length", 0)

        if stype == "arc":
            angle = s.get("arc_angle_deg", 0)
            radius = s.get("radius_est", 0)
            if angle >= 150:
                desc = f"U-bend ({angle:.0f}°, R={radius:.1f})"
            elif angle >= 80:
                desc = f"elbow ({angle:.0f}°, R={radius:.1f})"
            elif angle >= 40:
                desc = f"moderate bend ({angle:.0f}°, R={radius:.1f})"
            else:
                desc = f"gentle curve ({angle:.0f}°, R={radius:.1f})"
            lines.append(f"  #{sid}: arc — {desc}, length {length:.1f}")
        elif stype == "corner":
            angle = s.get("corner_angle_deg", 0)
            lines.append(f"  #{sid}: corner — sharp {angle:.0f}° turn, length {length:.1f}")
        elif stype == "straight":
            lines.append(f"  #{sid}: straight — {length:.1f} units long")
        elif stype == "junction":
            lines.append(f"  #{sid}: junction — pipe branching point, {length:.1f} units")
        else:
            lines.append(f"  #{sid}: {stype} — length {length:.1f}")

    return "\n".join(lines)


def format_step_explanation(step_name: str, detail: Dict[str, Any]) -> str:
    """Generate a natural, human-readable explanation for a pipeline step."""
    counts = detail.get("counts_by_type", {})

    if step_name == "segmenting":
        return (
            "🔍 **Analyzing the pipe geometry...**\n"
            "Looking at the graph topology to find junctions, then tracing each branch "
            "and measuring curvature to identify straight runs, curved bends, and sharp corners."
        )
    elif step_name == "segmented":
        total = detail.get("total_segments", 0)
        parts = []
        for t, c in counts.items():
            if t == "straight":
                parts.append(f"**{c}** straight section{'s' if c != 1 else ''}")
            elif t == "arc":
                parts.append(f"**{c}** curved bend{'s' if c != 1 else ''}")
            elif t == "corner":
                parts.append(f"**{c}** sharp corner{'s' if c != 1 else ''}")
            elif t == "junction":
                parts.append(f"**{c}** junction{'s' if c != 1 else ''}")
        summary = ", ".join(parts) if parts else f"{total} segments"
        return f"✅ **Analysis complete!** Found {summary} in this geometry."

    elif step_name == "downsampling":
        return (
            f"📐 **Simplifying each segment** to ~{detail.get('target_nodes', 16)} evenly-spaced points "
            "for efficient processing."
        )
    elif step_name == "downsampled":
        return f"✅ **Simplified {detail.get('segments_processed', '?')} segments** with uniform spacing."

    elif step_name == "embedding":
        return "🧠 **Computing shape signatures** using the neural network encoder..."

    elif step_name == "embedded":
        return f"✅ **Generated embeddings** for {detail.get('segments_embedded', '?')} segments."

    elif step_name == "mapping_faces":
        return "🗺️ **Mapping segments to surface mesh** for 3D visualization..."

    elif step_name == "faces_mapped":
        n = detail.get("segments_with_faces", 0)
        return f"✅ **Mapped {n} segments** to their surface mesh faces."

    elif step_name == "stored":
        return "💾 **Results saved** and ready for querying."

    elif step_name == "tool_call":
        tool = detail.get("tool", "?")
        params = detail.get("params", {})
        # Make tool calls readable
        if tool == "list_segments":
            return "🔧 Listing all segments..."
        elif tool == "filter_segments":
            return f"🔧 Filtering segments by {', '.join(f'{k}={v}' for k, v in params.items())}..."
        elif tool == "count_segments":
            return "🔧 Counting segments..."
        elif tool == "highlight_segments":
            return f"🔧 Highlighting segments {params.get('segment_ids', [])}..."
        elif tool == "describe_segment":
            return f"🔧 Getting details for segment #{params.get('segment_id', '?')}..."
        else:
            return f"🔧 Running {tool}..."

    elif step_name == "parsing_query":
        return f"💭 Understanding your question: *\"{detail.get('query', '?')}\"*"

    else:
        return f"⚙️ {step_name}"


def summarize_pipeline_intro(num_nodes: int, num_edges: int) -> str:
    """Generate an introduction message when a new graph is uploaded."""
    return (
        f"📊 **Graph loaded**: {num_nodes} nodes, {num_edges} edges.\n\n"
        "I'll now:\n"
        "1. **Split** the graph into segments (junction / straight / arc / corner)\n"
        "2. **Downsample** each segment to ~16 nodes with uniform spacing\n"
        "3. **Classify** each segment type and compute angles\n"
        "4. **Store** results for querying\n\n"
        "Starting analysis..."
    )
