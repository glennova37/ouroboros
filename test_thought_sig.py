#!/usr/bin/env python3
"""
Diagnostic script: test thoughtSignature handling with Antigravity API.

Run in Colab after the launcher boots (needs antigravity tokens).

Tests:
  1. Simple text-only call (no tools) — should always work
  2. One-shot tool call + roundtrip — must preserve thoughtSignature
  3. Two sequential tool steps — must preserve ALL signatures
  4. Parallel tool call + roundtrip — signature only on first FC
"""
from __future__ import annotations
import json, sys, time, requests, traceback

# ── Auth ──────────────────────────────────────────────────────────────
sys.path.insert(0, "/content/ouroboros_repo")
from ouroboros.antigravity_auth import get_access_token, get_project_id

ENDPOINT = "https://cloudcode-pa.googleapis.com"
MODEL = "gemini-3.1-pro-high"  # same as production

def _headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }

def call_api(contents, tools=None, label=""):
    """Make a raw Antigravity API call and return (status, data_or_error)."""
    import uuid
    inner = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }
    if tools:
        inner["tools"] = tools

    body = {
        "project": get_project_id() or "",
        "model": MODEL,
        "request": {"model": MODEL, **inner},
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"diag-{uuid.uuid4()}",
    }

    resp = requests.post(
        f"{ENDPOINT}/v1internal:generateContent",
        headers=_headers(),
        json=body,
        timeout=60,
    )

    print(f"\n{'='*60}")
    print(f"[{label}] Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"[{label}] ERROR: {resp.text[:500]}")
        return resp.status_code, resp.text

    data = resp.json()
    if "response" in data and "candidates" in data["response"]:
        data = data["response"]

    # Extract parts from response
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    print(f"[{label}] Response parts count: {len(parts)}")
    for i, p in enumerate(parts):
        keys = sorted(p.keys())
        has_ts = "thoughtSignature" in p
        is_thought = p.get("thought", False)
        if "functionCall" in p:
            print(f"  Part {i}: functionCall(name={p['functionCall'].get('name')}) "
                  f"thoughtSignature={'YES' if has_ts else 'NO'} keys={keys}")
        elif "text" in p:
            snippet = p["text"][:80].replace("\n", "\\n")
            print(f"  Part {i}: text({snippet}...) thought={is_thought} "
                  f"thoughtSignature={'YES' if has_ts else 'NO'}")
        else:
            print(f"  Part {i}: keys={keys}")

    return resp.status_code, data


SIMPLE_TOOLS = [{
    "functionDeclarations": [{
        "name": "get_time",
        "description": "Returns the current UTC time",
        "parameters": {"type": "object", "properties": {}},
    }]
}]

TWO_TOOLS = [{
    "functionDeclarations": [
        {
            "name": "get_time",
            "description": "Returns the current UTC time",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_weather",
            "description": "Returns weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]
}]


def test_1_no_tools():
    """Test 1: Simple text, no tools."""
    print("\n" + "="*60)
    print("TEST 1: Simple text, no tools")
    contents = [{"role": "user", "parts": [{"text": "Say hello in one word."}]}]
    status, _ = call_api(contents, label="T1")
    return status == 200


def test_2_single_tool_roundtrip():
    """Test 2: Tool call + roundtrip with thoughtSignature."""
    print("\n" + "="*60)
    print("TEST 2: Single tool call + roundtrip")

    # Step 1: Initial request
    contents = [{"role": "user", "parts": [{"text": "What time is it? Use get_time."}]}]
    status, data = call_api(contents, tools=SIMPLE_TOOLS, label="T2-step1")
    if status != 200:
        return False

    # Extract model response parts (filter thought parts)
    model_parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    # Keep all non-thought parts as-is (including thoughtSignature)
    roundtrip_parts = [p for p in model_parts if not p.get("thought")]

    print(f"\n  Roundtrip parts to send back:")
    for i, p in enumerate(roundtrip_parts):
        has_ts = "thoughtSignature" in p
        if "functionCall" in p:
            print(f"    Part {i}: FC({p['functionCall']['name']}) ts={'YES' if has_ts else 'NO'}")
        elif "text" in p:
            print(f"    Part {i}: text ts={'YES' if has_ts else 'NO'}")

    # Find if there's a functionCall
    fc_parts = [p for p in roundtrip_parts if "functionCall" in p]
    if not fc_parts:
        print("  No functionCall in response — model answered directly")
        return True

    # Step 2: Send back with functionResponse
    contents_step2 = [
        {"role": "user", "parts": [{"text": "What time is it? Use get_time."}]},
        {"role": "model", "parts": roundtrip_parts},
        {"role": "user", "parts": [
            {"functionResponse": {
                "name": "get_time",
                "response": {"time": "2026-02-20T14:00:00Z"},
            }}
        ]},
    ]
    status2, _ = call_api(contents_step2, tools=SIMPLE_TOOLS, label="T2-step2")
    return status2 == 200


def test_3_two_sequential_steps():
    """Test 3: Two sequential tool calls (multi-step)."""
    print("\n" + "="*60)
    print("TEST 3: Two sequential tool steps")

    # Step 1
    contents = [{"role": "user", "parts": [
        {"text": "First get the time, then get weather for Moscow. Do them one at a time."}
    ]}]
    status, data = call_api(contents, tools=TWO_TOOLS, label="T3-step1")
    if status != 200:
        return False

    model_parts_1 = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    roundtrip_1 = [p for p in model_parts_1 if not p.get("thought")]

    fc_1 = [p for p in roundtrip_1 if "functionCall" in p]
    if not fc_1:
        print("  No FC in step 1 — model answered directly")
        return True

    fc_name_1 = fc_1[0]["functionCall"]["name"]
    print(f"  Step 1 FC: {fc_name_1}")

    # Step 2: send FR, get second FC
    contents_step2 = [
        {"role": "user", "parts": [
            {"text": "First get the time, then get weather for Moscow. Do them one at a time."}
        ]},
        {"role": "model", "parts": roundtrip_1},
        {"role": "user", "parts": [
            {"functionResponse": {
                "name": fc_name_1,
                "response": {"result": "2026-02-20T14:00:00Z" if fc_name_1 == "get_time"
                             else "Sunny, 5°C"},
            }}
        ]},
    ]
    status2, data2 = call_api(contents_step2, tools=TWO_TOOLS, label="T3-step2")
    if status2 != 200:
        return False

    model_parts_2 = data2.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    roundtrip_2 = [p for p in model_parts_2 if not p.get("thought")]

    fc_2 = [p for p in roundtrip_2 if "functionCall" in p]
    if not fc_2:
        print("  No FC in step 2 — model answered with text")
        return True

    fc_name_2 = fc_2[0]["functionCall"]["name"]
    print(f"  Step 2 FC: {fc_name_2}")

    # Step 3: final response
    contents_step3 = contents_step2 + [
        {"role": "model", "parts": roundtrip_2},
        {"role": "user", "parts": [
            {"functionResponse": {
                "name": fc_name_2,
                "response": {"result": "Sunny, 5°C" if fc_name_2 == "get_weather"
                             else "2026-02-20T14:00:00Z"},
            }}
        ]},
    ]
    status3, _ = call_api(contents_step3, tools=TWO_TOOLS, label="T3-step3")
    return status3 == 200


def test_4_parallel_tool_call():
    """Test 4: Parallel tool calls."""
    print("\n" + "="*60)
    print("TEST 4: Parallel tool calls")

    contents = [{"role": "user", "parts": [
        {"text": "Get the time AND weather for London simultaneously."}
    ]}]
    status, data = call_api(contents, tools=TWO_TOOLS, label="T4-step1")
    if status != 200:
        return False

    model_parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    roundtrip = [p for p in model_parts if not p.get("thought")]

    fc_parts = [p for p in roundtrip if "functionCall" in p]
    print(f"  Got {len(fc_parts)} function calls")
    for i, p in enumerate(fc_parts):
        has_ts = "thoughtSignature" in p
        print(f"    FC{i}: {p['functionCall']['name']} ts={'YES' if has_ts else 'NO'}")

    if len(fc_parts) < 2:
        print("  Model didn't make parallel calls — test inconclusive")
        return True

    # Build FRs for all FCs
    fr_parts = []
    for p in fc_parts:
        name = p["functionCall"]["name"]
        fr_parts.append({
            "functionResponse": {
                "name": name,
                "response": {"result": "14:00 UTC" if name == "get_time" else "Cloudy, 8°C"},
            }
        })

    contents_step2 = [
        contents[0],
        {"role": "model", "parts": roundtrip},
        {"role": "user", "parts": fr_parts},
    ]
    status2, _ = call_api(contents_step2, tools=TWO_TOOLS, label="T4-step2")
    return status2 == 200


def test_5_claude_fc_in_history():
    """Test 5: Simulates Claude FC without thoughtSignature in history."""
    print("\n" + "="*60)
    print("TEST 5: Simulated Claude FC (no thoughtSignature) in current turn")

    # This simulates what happens when Claude responds with a tool call:
    # the functionCall part has NO thoughtSignature
    contents = [
        {"role": "user", "parts": [{"text": "Get the time."}]},
        # Claude-generated FC — NO thoughtSignature!
        {"role": "model", "parts": [
            {"functionCall": {"name": "get_time", "args": {}}}
        ]},
        {"role": "user", "parts": [
            {"functionResponse": {
                "name": "get_time",
                "response": {"time": "2026-02-20T14:00:00Z"},
            }}
        ]},
    ]
    status, _ = call_api(contents, tools=SIMPLE_TOOLS, label="T5")
    print(f"\n  Expected: 400 (missing thought_signature)")
    print(f"  Got: {status}")
    return status  # Return status code so we can report it


if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("T1: No tools", test_1_no_tools),
        ("T2: Single FC roundtrip", test_2_single_tool_roundtrip),
        ("T3: Sequential multi-step", test_3_two_sequential_steps),
        ("T4: Parallel FCs", test_4_parallel_tool_call),
        ("T5: Unsigned FC (Claude sim)", test_5_claude_fc_in_history),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            traceback.print_exc()
            results[name] = f"ERROR: {e}"

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, result in results.items():
        status = "✅ PASS" if result is True else f"❌ {result}"
        print(f"  {name}: {status}")
