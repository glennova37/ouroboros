#!/usr/bin/env python3
"""Test Antigravity API with tools to diagnose 404/400."""
import json, sys, os, uuid

# Must have TOKEN_DIR set
os.environ.setdefault("OUROBOROS_TOKEN_DIR", "/tmp/ouroboros_tokens")

from ouroboros.antigravity_auth import get_access_token, get_project_id
from ouroboros.antigravity_client import _get_headers, _resolve_model, ENDPOINTS

token = get_access_token()
project_id = get_project_id()
headers = _get_headers(token)

# Test 1: Simple call without tools (should work)
print("=" * 60)
print("TEST 1: Simple call WITHOUT tools")
body_simple = {
    "project": project_id,
    "model": "gemini-3.1-pro-high",
    "request": {
        "model": "gemini-3.1-pro-high",
        "contents": [{"role": "user", "parts": [{"text": "Say 'hello'"}]}],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    },
    "requestType": "agent",
    "userAgent": "antigravity",
    "requestId": f"test-{uuid.uuid4()}",
}

import requests
url = f"{ENDPOINTS[0]}/v1internal:generateContent"

resp = requests.post(url, headers=headers, json=body_simple, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    if "response" in data:
        data = data["response"]
    candidates = data.get("candidates", [])
    if candidates:
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"Response: {text[:100]}")
    print("✅ TEST 1 PASSED")
else:
    print(f"Response: {resp.text[:300]}")
    print("❌ TEST 1 FAILED")

# Test 2: Call WITH tools (function declarations) - no tool results
print("\n" + "=" * 60)
print("TEST 2: Call WITH tools (functionDeclarations)")
body_tools = {
    "project": project_id,
    "model": "gemini-3.1-pro-high",
    "request": {
        "model": "gemini-3.1-pro-high",
        "contents": [{"role": "user", "parts": [{"text": "What time is it?"}]}],
        "tools": [{"functionDeclarations": [{
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        }]}],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    },
    "requestType": "agent",
    "userAgent": "antigravity",
    "requestId": f"test-{uuid.uuid4()}",
}

resp = requests.post(url, headers=headers, json=body_tools, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(f"Response preview: {resp.text[:200]}")
    print("✅ TEST 2 PASSED")
else:
    print(f"Response: {resp.text[:300]}")
    print("❌ TEST 2 FAILED")

# Test 3: Same but WITHOUT thinkingConfig
print("\n" + "=" * 60)
print("TEST 3: Call WITH tools but WITHOUT thinkingConfig")
body_no_think = {
    "project": project_id,
    "model": "gemini-3.1-pro-high",
    "request": {
        "model": "gemini-3.1-pro-high",
        "contents": [{"role": "user", "parts": [{"text": "What time is it?"}]}],
        "tools": [{"functionDeclarations": [{
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        }]}],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 1.0,
        },
    },
    "requestType": "agent",
    "userAgent": "antigravity",
    "requestId": f"test-{uuid.uuid4()}",
}

resp = requests.post(url, headers=headers, json=body_no_think, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(f"Response preview: {resp.text[:200]}")
    print("✅ TEST 3 PASSED")
else:
    print(f"Response: {resp.text[:300]}")
    print("❌ TEST 3 FAILED")

# Test 4: With tool results (functionCall + functionResponse in history)
print("\n" + "=" * 60)
print("TEST 4: With functionCall + functionResponse in history")
body_fn = {
    "project": project_id,
    "model": "gemini-3.1-pro-high",
    "request": {
        "model": "gemini-3.1-pro-high",
        "contents": [
            {"role": "user", "parts": [{"text": "What time is it?"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "get_time", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_time", "response": {"result": "12:00 PM"}}}]},
        ],
        "tools": [{"functionDeclarations": [{
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        }]}],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 1.0,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    },
    "requestType": "agent",
    "userAgent": "antigravity",
    "requestId": f"test-{uuid.uuid4()}",
}

resp = requests.post(url, headers=headers, json=body_fn, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(f"Response preview: {resp.text[:200]}")
    print("✅ TEST 4 PASSED")
else:
    print(f"Response: {resp.text[:300]}")
    print("❌ TEST 4 FAILED")

# Test 5: claude-sonnet-4-6 with tools
print("\n" + "=" * 60)
print("TEST 5: claude-sonnet-4-6 with tools")
body_claude = {
    "project": project_id,
    "model": "claude-sonnet-4-6",
    "request": {
        "model": "claude-sonnet-4-6",
        "contents": [{"role": "user", "parts": [{"text": "What time is it?"}]}],
        "tools": [{"functionDeclarations": [{
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        }]}],
        "generationConfig": {
            "maxOutputTokens": 16384,
            "temperature": 1.0,
        },
    },
    "requestType": "agent",
    "userAgent": "antigravity",
    "requestId": f"test-{uuid.uuid4()}",
}

resp = requests.post(url, headers=headers, json=body_claude, timeout=30)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    print(f"Response preview: {resp.text[:200]}")
    print("✅ TEST 5 PASSED")
else:
    print(f"Response: {resp.text[:300]}")
    print("❌ TEST 5 FAILED")

print("\n" + "=" * 60)
print("DONE")
