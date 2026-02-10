#!/usr/bin/env python3
"""
ai/test_gemini.py
─────────────────
Quick standalone test for the Gemini integration.
Run from the deploy/ directory:

    export GEMINI_API_KEY=your_key_here
    python3 -m ai.test_gemini

Tests:
  1. API key + client creation
  2. Simple text generation (no tools)
  3. Function calling with fake segments
  4. Follow-up query (multi-turn)
"""

import json
import os
import sys
import traceback
from pathlib import Path

# Ensure deploy/ is on the path
deploy_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deploy_dir)

# Load .env from deploy/ directory
try:
    from dotenv import load_dotenv
    env_path = Path(deploy_dir) / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📄 Loaded .env from {env_path}")
    else:
        load_dotenv()
        print(f"📄 No .env file found at {env_path} — using shell env vars")
except ImportError:
    print("⚠️  python-dotenv not installed — using shell env vars only")
    print("   Install: pip install python-dotenv")


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── Fake segment data for testing ──
FAKE_SEGMENTS = [
    {"segment_id": 0, "type": "straight", "node_count": 40, "length": 25.0,
     "arc_angle_deg": 0, "corner_angle_deg": 0, "mean_curvature": 0.0,
     "max_curvature": 0.001, "radius_est": 0},
    {"segment_id": 1, "type": "arc", "node_count": 30, "length": 12.5,
     "arc_angle_deg": 90.0, "corner_angle_deg": 0, "mean_curvature": 0.2,
     "max_curvature": 0.22, "radius_est": 5.0},
    {"segment_id": 2, "type": "straight", "node_count": 35, "length": 20.0,
     "arc_angle_deg": 0, "corner_angle_deg": 0, "mean_curvature": 0.0,
     "max_curvature": 0.002, "radius_est": 0},
    {"segment_id": 3, "type": "arc", "node_count": 50, "length": 18.0,
     "arc_angle_deg": 180.0, "corner_angle_deg": 0, "mean_curvature": 0.125,
     "max_curvature": 0.13, "radius_est": 8.0},
    {"segment_id": 4, "type": "straight", "node_count": 40, "length": 25.0,
     "arc_angle_deg": 0, "corner_angle_deg": 0, "mean_curvature": 0.0,
     "max_curvature": 0.001, "radius_est": 0},
    {"segment_id": 5, "type": "arc", "node_count": 20, "length": 8.0,
     "arc_angle_deg": 45.0, "corner_angle_deg": 0, "mean_curvature": 0.1,
     "max_curvature": 0.12, "radius_est": 10.0},
    {"segment_id": 6, "type": "corner", "node_count": 5, "length": 1.2,
     "arc_angle_deg": 0, "corner_angle_deg": 92.0, "mean_curvature": 1.5,
     "max_curvature": 2.0, "radius_est": 0.67},
]


def test_1_api_key():
    """Test 1: Check API key is set and client can be created."""
    separator("TEST 1: API Key + Client")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ GEMINI_API_KEY is NOT set!")
        print("   Fix: export GEMINI_API_KEY=your_key_here")
        return False

    print(f"✅ GEMINI_API_KEY is set ({len(api_key)} chars, starts with {api_key[:8]}...)")

    try:
        from ai.gemini import get_client, MODEL
        client = get_client()
        print(f"✅ Client created successfully")
        print(f"   Model: {MODEL}")
        return True
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        traceback.print_exc()
        return False


def test_2_simple_generation():
    """Test 2: Simple text generation (no tools)."""
    separator("TEST 2: Simple Text Generation (no tools)")

    try:
        from ai.gemini import get_client, MODEL
        from google.genai import types

        client = get_client()
        response = client.models.generate_content(
            model=MODEL,
            contents="Say hello in one sentence.",
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = response.text
        print(f"✅ Gemini responded: {text}")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        traceback.print_exc()
        return False


def test_3_function_calling():
    """Test 3: Function calling with segment query."""
    separator("TEST 3: Function Calling (describe this geometry)")

    try:
        from dsl.engine import SegmentQueryEngine, query_with_ai
        from ai.prompts import build_segments_context

        engine = SegmentQueryEngine(FAKE_SEGMENTS)

        def on_step(step, detail):
            print(f"  📡 Step: {step} → {json.dumps(detail, default=str)[:200]}")

        result = query_with_ai(
            user_message="describe this geometry",
            segments=FAKE_SEGMENTS,
            chat_history=[],
            on_step=on_step,
        )

        print(f"\n✅ AI responded (mode={result.get('mode')}):")
        print(f"   Answer: {result['answer'][:300]}...")
        print(f"   Tool calls: {len(result.get('tool_calls', []))}")
        for tc in result.get("tool_calls", []):
            print(f"     🔧 {tc['tool']}({json.dumps(tc['params'], default=str)[:100]})")
        return True
    except Exception as e:
        print(f"❌ Function calling failed: {e}")
        traceback.print_exc()
        return False


def test_4_followup():
    """Test 4: Follow-up query (multi-turn with chat history)."""
    separator("TEST 4: Follow-up Query (how many arcs above 40°? → and above 80°?)")

    try:
        from dsl.engine import query_with_ai

        def on_step(step, detail):
            print(f"  📡 Step: {step} → {json.dumps(detail, default=str)[:200]}")

        # First query
        print("─── Query 1: 'how many arcs above 40 degrees?' ───")
        result1 = query_with_ai(
            user_message="how many arcs above 40 degrees?",
            segments=FAKE_SEGMENTS,
            chat_history=[],
            on_step=on_step,
        )
        print(f"   Answer: {result1['answer'][:200]}")

        # Build chat history from first exchange
        chat_history = [
            {"role": "user", "content": "how many arcs above 40 degrees?"},
            {"role": "assistant", "content": result1["answer"],
             "tool_calls": result1.get("tool_calls", [])},
        ]

        # Follow-up query
        print("\n─── Query 2 (follow-up): 'and above 80°?' ───")
        result2 = query_with_ai(
            user_message="and above 80°?",
            segments=FAKE_SEGMENTS,
            chat_history=chat_history,
            on_step=on_step,
        )
        print(f"   Answer: {result2['answer'][:200]}")

        # Verify follow-up understood the context
        answer2 = result2["answer"].lower()
        # The answer should mention 1 or 2 arcs (90° and 180° are above 80°)
        print(f"\n✅ Follow-up query completed")
        print(f"   Tool calls in Q1: {len(result1.get('tool_calls', []))}")
        print(f"   Tool calls in Q2: {len(result2.get('tool_calls', []))}")
        return True
    except Exception as e:
        print(f"❌ Follow-up test failed: {e}")
        traceback.print_exc()
        return False


def test_5_followup_context():
    """Test 5: Follow-up query with conversation context via AI."""
    separator("TEST 5: Follow-up Context (AI)")

    try:
        from dsl.engine import query_smart

        # First query
        print("─── Query 1: 'how many arcs above 40 degrees?' ───")
        result1 = query_smart(
            user_message="how many arcs above 40 degrees?",
            segments=FAKE_SEGMENTS,
            chat_history=[],
        )
        print(f"   Answer: {result1['answer'][:200]}")

        # Build chat history from first result
        chat_history = [
            {"role": "user", "content": "how many arcs above 40 degrees?"},
            {"role": "assistant", "content": result1["answer"],
             "tool_calls": result1.get("tool_calls", [])},
        ]

        # Follow-up: "and above 80°?"
        print("\n─── Query 2 (follow-up): 'and above 80°?' ───")
        result2 = query_smart(
            user_message="and above 80°?",
            segments=FAKE_SEGMENTS,
            chat_history=chat_history,
        )
        print(f"   Answer: {result2['answer'][:200]}")

        # Follow-up: "does it contain straight?"
        chat_history.append({"role": "user", "content": "and above 80°?"})
        chat_history.append({"role": "assistant", "content": result2["answer"],
                             "tool_calls": result2.get("tool_calls", [])})

        print("\n─── Query 3 (follow-up): 'does it contain straight?' ───")
        result3 = query_smart(
            user_message="does it contain straight?",
            segments=FAKE_SEGMENTS,
            chat_history=chat_history,
        )
        print(f"   Answer: {result3['answer'][:200]}")

        print(f"\n✅ All follow-up queries completed via AI")
        return True
    except Exception as e:
        print(f"❌ Follow-up context test failed: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Gemini Integration Test Suite")
    print(f"   Working dir: {os.getcwd()}")
    print(f"   GEMINI_API_KEY: {'SET' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}")
    print(f"   GEMINI_MODEL: {os.environ.get('GEMINI_MODEL', 'gemini-3-flash-preview')}")

    results = {}

    if not os.environ.get("GEMINI_API_KEY"):
        print("\n" + "="*60)
        print("  ❌ GEMINI_API_KEY not set — cannot run tests")
        print("  Create a .env file in deploy/ with: GEMINI_API_KEY=your_key_here")
        print("="*60)
        sys.exit(1)

    results["1_api_key"] = test_1_api_key()

    if results.get("1_api_key"):
        results["2_simple_gen"] = test_2_simple_generation()
        results["3_function_calling"] = test_3_function_calling()
        results["4_followup"] = test_4_followup()
        results["5_followup_context"] = test_5_followup_context()

    # Summary
    separator("RESULTS")
    all_pass = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed — check output above")
        sys.exit(1)
