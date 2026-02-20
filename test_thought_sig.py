#!/usr/bin/env python3
"""
BRUTE-FORCE model name scanner for Cloud Code / Antigravity API.
Tries every combination of model name × suffix × thinkingConfig × tools.
Run: python test_thought_sig.py
"""
from __future__ import annotations
import json, sys, uuid, time, requests
from itertools import product

sys.path.insert(0, "/content/ouroboros_repo")
from ouroboros.antigravity_auth import get_access_token, get_project_id

ENDPOINT = "https://cloudcode-pa.googleapis.com"

# ── All possible model name bases ─────────────────────────────────────
BASES = [
    # Gemini 3.1
    "gemini-3.1-pro",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    # Gemini 3
    "gemini-3-pro",
    "gemini-3-pro-preview",
    # Gemini 3 Flash
    "gemini-3-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash",
    "gemini-3.1-flash-preview",
    # Gemini 2.5
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview",
    # Gemini 2.0
    "gemini-2.0-pro",
    "gemini-2.0-flash",
]

# ── Suffixes to try ───────────────────────────────────────────────────
SUFFIXES = [
    "",           # bare
    "-high",
    "-low",
    "-001",
    "-exp-0206",
    "-latest",
]

# ── ThinkingConfig variants ───────────────────────────────────────────
THINKING_CONFIGS = {
    "none":  None,
    "high":  {"thinkingLevel": "high"},
    "low":   {"thinkingLevel": "low"},
    "off":   {"thinkingLevel": "off"},
    "budget": {"thinkingBudget": 1024},
}

SIMPLE_CONTENTS = [{"role": "user", "parts": [{"text": "Say OK"}]}]

SIMPLE_TOOLS = [{"functionDeclarations": [{
    "name": "get_time",
    "description": "Returns current UTC time",
    "parameters": {"type": "object", "properties": {}},
}]}]


def call(model, thinking_key="none", with_tools=False):
    """Single API call. Returns (status, short_info)."""
    token = get_access_token()
    project = get_project_id() or ""

    gen_config = {"maxOutputTokens": 256, "temperature": 1.0}
    tc = THINKING_CONFIGS[thinking_key]
    if tc is not None:
        gen_config["thinkingConfig"] = tc

    inner = {"contents": SIMPLE_CONTENTS, "generationConfig": gen_config}
    if with_tools:
        inner["tools"] = SIMPLE_TOOLS

    body = {
        "project": project, "model": model,
        "request": {"model": model, **inner},
        "requestType": "agent", "userAgent": "antigravity",
        "requestId": f"scan-{uuid.uuid4()}",
    }

    try:
        r = requests.post(
            f"{ENDPOINT}/v1internal:generateContent",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body, timeout=15,
        )
        if r.status_code == 200:
            d = r.json()
            if "response" in d and "candidates" in d["response"]:
                d = d["response"]
            parts = d.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            fc = sum(1 for p in parts if "functionCall" in p)
            ts = sum(1 for p in parts if "thoughtSignature" in p)
            txt = sum(len(p.get("text", "")) for p in parts)
            return 200, f"p={len(parts)} fc={fc} ts={ts} txt={txt}"
        else:
            # Extract short error
            try:
                msg = r.json().get("error", {}).get("message", "")[:80]
            except Exception:
                msg = r.text[:80]
            return r.status_code, msg
    except Exception as e:
        return -1, str(e)[:60]


def main():
    token = get_access_token()
    project = get_project_id()
    print(f"Project: {project}")
    print(f"Token:   {token[:25]}...")
    print(f"Endpoint: {ENDPOINT}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Quick scan — base+suffix, no thinking, no tools
    # Find which model names exist at all
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("PHASE 1: Model existence scan (no thinking, no tools)")
    print("=" * 80)

    found = []
    total = len(BASES) * len(SUFFIXES)
    i = 0
    for base in BASES:
        for suffix in SUFFIXES:
            model = base + suffix
            i += 1
            status, info = call(model, "none", False)
            tag = "✅" if status == 200 else ("⚠️" if status == 400 else "  ")
            if status not in (403, 404):
                print(f"  {tag} [{i:3d}/{total}] {model:50s} → {status} | {info}")
                found.append(model)
            # Small delay to avoid rate limiting
            time.sleep(0.1)

    if not found:
        print("\n  ❌ NO models responded with anything other than 403/404!")
        print("     Either the API is down or auth is broken.")
        return

    print(f"\n  Found {len(found)} responsive models")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: For responsive models, test thinkingConfig variants
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("PHASE 2: ThinkingConfig variants for responsive models")
    print("=" * 80)

    working = []  # (model, thinking_key) pairs that return 200
    for model in found:
        print(f"\n  {model}:")
        for tk in THINKING_CONFIGS:
            status, info = call(model, tk, False)
            tag = "✅" if status == 200 else "❌"
            print(f"    {tag} think={tk:8s} → {status} | {info}")
            if status == 200:
                working.append((model, tk))
            time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: For working combos, test with tools
    # ═══════════════════════════════════════════════════════════════════
    if working:
        print()
        print("=" * 80)
        print("PHASE 3: Tools for working model+thinking combos")
        print("=" * 80)

        for model, tk in working:
            status, info = call(model, tk, True)
            tag = "✅" if status == 200 else "❌"
            print(f"  {tag} {model:50s} think={tk:8s} tools=yes → {status} | {info}")
            time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("SUMMARY: All working combinations")
    print("=" * 80)
    if working:
        for model, tk in working:
            print(f"  ✅ {model} (thinking={tk})")
    else:
        print("  ❌ No working combinations found!")


if __name__ == "__main__":
    main()
