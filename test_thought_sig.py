#!/usr/bin/env python3
"""
Diagnostic: brute-force test every combination of endpoint × model × tools.
Run in Colab: python test_thought_sig.py
"""
from __future__ import annotations
import json, sys, uuid, requests

sys.path.insert(0, "/content/ouroboros_repo")
from ouroboros.antigravity_auth import get_access_token, get_project_id

ENDPOINTS = [
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

MODELS = [
    "gemini-3.1-pro-high",
    "gemini-3.1-pro",
    "gemini-3-pro-high",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

SIMPLE_CONTENTS = [{"role": "user", "parts": [{"text": "Say hi in one word."}]}]

SIMPLE_TOOLS = [{"functionDeclarations": [{
    "name": "get_time",
    "description": "Returns current UTC time",
    "parameters": {"type": "object", "properties": {}},
}]}]


def call(endpoint, model, with_tools=False, with_thinking=False):
    token = get_access_token()
    project = get_project_id() or ""

    inner = {"contents": SIMPLE_CONTENTS, "generationConfig": {
        "maxOutputTokens": 1024, "temperature": 1.0,
    }}

    if with_thinking:
        if "flash" in model or "2.5" in model:
            inner["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}
        else:
            inner["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "low"}

    if with_tools:
        inner["tools"] = SIMPLE_TOOLS

    body = {
        "project": project,
        "model": model,
        "request": {"model": model, **inner},
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"diag-{uuid.uuid4()}",
    }

    try:
        resp = requests.post(
            f"{endpoint}/v1internal:generateContent",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=30,
        )
        status = resp.status_code
        if status == 200:
            data = resp.json()
            if "response" in data and "candidates" in data["response"]:
                data = data["response"]
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            fc_count = sum(1 for p in parts if "functionCall" in p)
            text_len = sum(len(p.get("text", "")) for p in parts)
            ts_count = sum(1 for p in parts if "thoughtSignature" in p)
            return status, f"parts={len(parts)} fc={fc_count} text_chars={text_len} ts={ts_count}"
        else:
            err = resp.text[:120].replace("\n", " ")
            return status, err
    except Exception as e:
        return -1, str(e)[:80]


def main():
    token = get_access_token()
    project = get_project_id()
    print(f"Project: {project}")
    print(f"Token prefix: {token[:20]}...")
    print()

    # Phase 1: Find which model+endpoint combos work at all (no tools, no thinking)
    print("=" * 80)
    print("PHASE 1: Model × Endpoint (no tools, no thinking)")
    print("=" * 80)
    working = []
    for model in MODELS:
        for ep in ENDPOINTS:
            ep_short = "autopush" if "autopush" in ep else "prod"
            status, info = call(ep, model, with_tools=False, with_thinking=False)
            tag = "✅" if status == 200 else "❌"
            print(f"  {tag} {model:30s} @ {ep_short:10s} → {status} | {info}")
            if status == 200:
                working.append((model, ep))

    if not working:
        print("\n⚠️  NO model+endpoint worked at all! Auth or API issue.")
        return

    # Phase 2: Test tools + thinking on working combos
    print()
    print("=" * 80)
    print("PHASE 2: Working combos with tools + thinking")
    print("=" * 80)
    for model, ep in working:
        ep_short = "autopush" if "autopush" in ep else "prod"

        # Tools only (no thinking)
        s1, i1 = call(ep, model, with_tools=True, with_thinking=False)
        tag1 = "✅" if s1 == 200 else "❌"

        # Thinking only (no tools)
        s2, i2 = call(ep, model, with_tools=False, with_thinking=True)
        tag2 = "✅" if s2 == 200 else "❌"

        # Tools + thinking
        s3, i3 = call(ep, model, with_tools=True, with_thinking=True)
        tag3 = "✅" if s3 == 200 else "❌"

        print(f"  {model:30s} @ {ep_short}")
        print(f"    {tag1} tools_only  → {s1} | {i1}")
        print(f"    {tag2} think_only  → {s2} | {i2}")
        print(f"    {tag3} tools+think → {s3} | {i3}")

    # Phase 3: Test tool roundtrip with signature preservation
    print()
    print("=" * 80)
    print("PHASE 3: Tool roundtrip (signature preservation)")
    print("=" * 80)
    for model, ep in working:
        ep_short = "autopush" if "autopush" in ep else "prod"

        # Step 1: get a function call
        inner1 = {
            "contents": [{"role": "user", "parts": [{"text": "What time is it? Use get_time tool."}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 1.0,
                                 "thinkingConfig": {"thinkingLevel": "low"}},
            "tools": SIMPLE_TOOLS,
        }
        body1 = {
            "project": get_project_id() or "", "model": model,
            "request": {"model": model, **inner1},
            "requestType": "agent", "userAgent": "antigravity",
            "requestId": f"diag-{uuid.uuid4()}",
        }
        try:
            r1 = requests.post(f"{ep}/v1internal:generateContent",
                               headers={"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"},
                               json=body1, timeout=30)
            if r1.status_code != 200:
                print(f"  {model:30s} @ {ep_short}: Step1 failed {r1.status_code}")
                continue

            d1 = r1.json()
            if "response" in d1 and "candidates" in d1["response"]:
                d1 = d1["response"]
            parts1 = d1.get("candidates", [{}])[0].get("content", {}).get("parts", [])

            # Filter thought parts, keep everything else
            model_parts = [p for p in parts1 if not p.get("thought")]
            fc_parts = [p for p in model_parts if "functionCall" in p]

            if not fc_parts:
                print(f"  {model:30s} @ {ep_short}: No FC in response (answered directly)")
                continue

            # Show signature info
            for i, p in enumerate(fc_parts):
                has_ts = "thoughtSignature" in p
                print(f"  FC{i}: {p['functionCall']['name']} ts={'YES' if has_ts else 'NO'} "
                      f"keys={sorted(p.keys())}")

            # Step 2: roundtrip
            inner2 = {
                "contents": [
                    {"role": "user", "parts": [{"text": "What time is it? Use get_time tool."}]},
                    {"role": "model", "parts": model_parts},
                    {"role": "user", "parts": [{"functionResponse": {
                        "name": fc_parts[0]["functionCall"]["name"],
                        "response": {"time": "2026-02-20T14:00:00Z"},
                    }}]},
                ],
                "generationConfig": {"maxOutputTokens": 1024, "temperature": 1.0,
                                     "thinkingConfig": {"thinkingLevel": "low"}},
                "tools": SIMPLE_TOOLS,
            }
            body2 = {
                "project": get_project_id() or "", "model": model,
                "request": {"model": model, **inner2},
                "requestType": "agent", "userAgent": "antigravity",
                "requestId": f"diag-{uuid.uuid4()}",
            }
            r2 = requests.post(f"{ep}/v1internal:generateContent",
                               headers={"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"},
                               json=body2, timeout=30)
            tag = "✅" if r2.status_code == 200 else "❌"
            info = ""
            if r2.status_code != 200:
                info = r2.text[:200].replace("\n", " ")
            else:
                d2 = r2.json()
                if "response" in d2 and "candidates" in d2["response"]:
                    d2 = d2["response"]
                parts2 = d2.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                info = f"parts={len(parts2)}"

            print(f"  {tag} {model:30s} @ {ep_short}: Roundtrip → {r2.status_code} | {info}")

        except Exception as e:
            print(f"  ❌ {model:30s} @ {ep_short}: {e}")


if __name__ == "__main__":
    main()
