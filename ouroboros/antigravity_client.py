"""Antigravity LLM Client — calls Google Cloud Code API with OAuth tokens.

Translates OpenAI-style messages to Google GenerativeAI format,
calls the Antigravity endpoint, and returns results in OpenAI dict format
for compatibility with the existing Ouroboros LLMClient interface.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.antigravity_auth import (
    get_access_token,
    get_project_id,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints & headers
# ---------------------------------------------------------------------------

ENDPOINTS = [
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
]

def _get_headers(access_token: str) -> Dict[str, str]:
    # IMPORTANT: do NOT include x-goog-user-project, X-Goog-Api-Client,
    # or Client-Metadata — they trigger 403 "Cloud Code Private API"
    # permission checks on the managed project.
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "antigravity/1.18.3 darwin/arm64",
    }

# ---------------------------------------------------------------------------
# Model name mapping (OpenRouter-style → Antigravity API model name)
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    # Exact API model names from opencode-antigravity-auth/model-resolver.ts
    # Gemini 3 Pro: REQUIRES tier suffix (-low or -high)
    # Gemini 3 Flash: bare name (thinkingLevel in config)
    "google/gemini-3-pro-preview": "gemini-3-pro-high",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-high",
    # Claude
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-opus-4.6": "claude-opus-4-6-thinking",
    # Pass-through
    "gemini-3-pro": "gemini-3-pro-high",
    "gemini-3-pro-low": "gemini-3-pro-low",
    "gemini-3-pro-high": "gemini-3-pro-high",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-3.1-pro": "gemini-3.1-pro-high",
    "gemini-3.1-pro-low": "gemini-3.1-pro-low",
    "gemini-3.1-pro-high": "gemini-3.1-pro-high",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6-thinking",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
}

def _resolve_model(model: str) -> str:
    """Map OpenRouter model name to Antigravity model name."""
    return _MODEL_MAP.get(model, model)


# ---------------------------------------------------------------------------
# Message conversion: OpenAI → Google GenerativeAI
# ---------------------------------------------------------------------------

def _resolve_fn_name(messages: List[Dict[str, Any]], tool_call_id: str) -> str:
    """Look up function name from tool_call_id by scanning assistant messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == tool_call_id:
                    return tc.get("function", {}).get("name", tool_call_id)
    return tool_call_id


def _openai_to_google(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convert OpenAI chat messages to Google GenerativeAI request body."""
    system_instruction = None
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            # System messages → systemInstruction
            text = content if isinstance(content, str) else json.dumps(content)
            system_instruction = {"parts": [{"text": text}]}
            continue

        # Map roles
        google_role = "user" if role == "user" else "model"

        # Handle tool calls in assistant messages
        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if content:
                parts.append({"text": content})
            for idx, tc in enumerate(msg["tool_calls"]):
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                fc_part: Dict[str, Any] = {
                    "functionCall": {
                        "name": fn.get("name", ""),
                        "args": args,
                        "id": tc.get("id", f"call_{idx}"),
                    }
                }
                # Restore thoughtSignature if saved during response parsing
                ts = tc.get("_thought_signature")
                if ts:
                    fc_part["thoughtSignature"] = ts
                parts.append(fc_part)
            contents.append({"role": google_role, "parts": parts})
            continue

        # Handle tool results
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            # Resolve function name: prefer explicit name, then look up from
            # the matching tool_call in the preceding assistant message
            name = msg.get("name", "")
            if not name:
                name = _resolve_fn_name(messages, tool_call_id)
            text = content if isinstance(content, str) else json.dumps(content)
            try:
                response_data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": text}
            # google.protobuf.Struct requires top-level dict — wrap lists/scalars
            if not isinstance(response_data, dict):
                response_data = {"result": response_data}
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": name,
                        "id": tool_call_id,
                        "response": response_data,
                    }
                }]
            })
            continue

        # Regular text
        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            # Multimodal content (images etc.)
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item["text"]})
                    elif item.get("type") == "image_url":
                        img_url = item.get("image_url", {}).get("url", "")
                        if img_url.startswith("data:"):
                            # Inline base64
                            mime, b64 = img_url.split(";base64,", 1)
                            mime = mime.replace("data:", "")
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime,
                                    "data": b64,
                                }
                            })
                        else:
                            parts.append({
                                "fileData": {"fileUri": img_url}
                            })
                else:
                    parts.append({"text": str(item)})
        else:
            parts = [{"text": str(content)}]

        contents.append({"role": google_role, "parts": parts})

    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    # Convert OpenAI tools → Google tools
    if tools:
        google_tools = _convert_tools(tools)
        if google_tools:
            body["tools"] = google_tools

    return body


def _convert_tools(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool definitions to Google function declarations."""
    declarations = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        params = fn.get("parameters", {})

        # Clean schema: remove unsupported keys
        cleaned = _clean_schema(params)

        declarations.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": cleaned,
        })

    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


def _clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys not supported by Google GenerativeAI (additionalProperties, etc.)."""
    result = {}
    for k, v in schema.items():
        if k in ("additionalProperties", "default", "$schema"):
            continue
        if isinstance(v, dict):
            result[k] = _clean_schema(v)
        elif isinstance(v, list):
            result[k] = [_clean_schema(item) if isinstance(item, dict) else item for item in v]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Response conversion: Google → OpenAI
# ---------------------------------------------------------------------------

def _google_to_openai_message(response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenerativeAI response to OpenAI message dict."""
    candidates = response.get("candidates", [])
    if not candidates:
        log.warning("Antigravity: no candidates in response: %s", json.dumps(response)[:500])
        return {"role": "assistant", "content": None}

    candidate = candidates[0]
    content_obj = candidate.get("content", {})
    parts = content_obj.get("parts", [])

    text_parts = []
    tool_calls = []
    tc_index = 0

    for part in parts:
        # Skip pure thought/thinking parts (both thought:true and thoughtSignature-only)
        if part.get("thought"):
            continue
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tc_entry: Dict[str, Any] = {
                "id": fc.get("id", f"call_{tc_index}"),
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(fc.get("args", {})),
                },
            }
            # Preserve thoughtSignature for roundtrip (API requires it back)
            if part.get("thoughtSignature"):
                tc_entry["_thought_signature"] = part["thoughtSignature"]
            else:
                log.warning("Antigravity: functionCall '%s' has NO thoughtSignature", fc.get("name"))
            tool_calls.append(tc_entry)
            tc_index += 1

    msg: Dict[str, Any] = {"role": "assistant"}

    content_text = "\n".join(text_parts) if text_parts else None
    msg["content"] = content_text

    if tool_calls:
        msg["tool_calls"] = tool_calls

    # Debug: log when we got parts but no usable content
    if not text_parts and not tool_calls and parts:
        log.warning("Antigravity: response had %d parts but no text/tool_calls. Part keys: %s",
                     len(parts), [list(p.keys()) for p in parts[:3]])

    return msg


def _extract_usage(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract usage from Google response."""
    meta = response.get("usageMetadata", {})
    prompt_tokens = int(meta.get("promptTokenCount", 0))
    completion_tokens = int(meta.get("candidatesTokenCount", 0))
    cached_tokens = int(meta.get("cachedContentTokenCount", 0))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "cost": 0.0,  # Antigravity is free
    }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class AntigravityClient:
    """Google Cloud Code API client using Antigravity OAuth tokens."""

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 16384,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the Antigravity API. Returns (message_dict, usage_dict)."""
        import requests

        api_model = _resolve_model(model)
        access_token = get_access_token()
        project_id = get_project_id()
        headers = _get_headers(access_token)

        # NOTE: do NOT set x-goog-user-project header — causes 403 on managed projects.
        # Project ID goes in the body instead.

        inner_body = _openai_to_google(messages, tools)

        # Generation config
        inner_body["generationConfig"] = {
            "maxOutputTokens": max_tokens,
            "temperature": 1.0,
        }

        # Add thinking config ONLY for thinking-capable models
        is_thinking = ("gemini-3" in api_model) or ("thinking" in api_model)
        if is_thinking:
            if "claude" in api_model:
                # Claude: maxOutputTokens MUST be > thinking_budget
                inner_body["generationConfig"]["thinkingConfig"] = {
                    "include_thoughts": True,
                    "thinking_budget": 32768,
                }
                inner_body["generationConfig"]["maxOutputTokens"] = max(max_tokens, 65536)
            else:
                # Gemini 3 uses thinkingLevel string (not numeric budget)
                # -high/-low are OUR internal suffixes, NOT part of the API model name
                level = "low"  # safe default
                if "-high" in api_model:
                    level = "high"
                inner_body["generationConfig"]["thinkingConfig"] = {
                    "thinkingLevel": level,
                }

        # Strip internal thinking-level suffix from model name before API call.
        # API only knows 'gemini-3.1-pro', not 'gemini-3.1-pro-high'.
        api_model_clean = api_model.replace("-high", "").replace("-low", "")

        # Antigravity wraps the request: {project, model, request, requestType, ...}
        import uuid
        body = {
            "project": project_id or "",
            "model": api_model_clean,
            "request": {
                "model": api_model_clean,
                **inner_body,
            },
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"agent-{uuid.uuid4()}",
        }

        # Try endpoints with fallback
        last_error = None
        for endpoint in ENDPOINTS:
            # URL has NO /models/{model} — model is in the body
            url = f"{endpoint}/v1internal:generateContent"
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=120,
                )

                if resp.status_code == 401:
                    # Token expired — refresh and retry once
                    access_token = get_access_token()
                    headers = _get_headers(access_token)
                    resp = requests.post(url, headers=headers, json=body, timeout=120)

                if resp.status_code == 429:
                    log.warning("Rate limited on %s, trying next endpoint", endpoint)
                    last_error = f"429 from {endpoint}"
                    continue

                if resp.status_code == 403:
                    log.warning("Permission denied on %s: %s", endpoint, resp.text[:200])
                    last_error = f"403 from {endpoint}: {resp.text[:200]}"
                    continue

                if resp.status_code in (400, 404):
                    # Log request details for debugging
                    n_tools = len(body.get("request", {}).get("tools", [{}])[0].get("functionDeclarations", [])) if body.get("request", {}).get("tools") else 0
                    n_contents = len(body.get("request", {}).get("contents", []))
                    has_fn_call = any(
                        any("functionCall" in p for p in c.get("parts", []))
                        for c in body.get("request", {}).get("contents", [])
                    )
                    has_fn_response = any(
                        any("functionResponse" in p for p in c.get("parts", []))
                        for c in body.get("request", {}).get("contents", [])
                    )
                    # Dump which contents have functionCall/Response parts for debugging
                    fc_diag = []
                    for ci, c in enumerate(body.get("request", {}).get("contents", [])):
                        for pi, p in enumerate(c.get("parts", [])):
                            if "functionCall" in p:
                                has_ts = "thoughtSignature" in p
                                fc_name = p["functionCall"].get("name", "?")
                                fc_keys = sorted(p["functionCall"].keys())
                                fc_diag.append(f"C{ci}P{pi}:FC({fc_name},ts={has_ts},keys={fc_keys})")
                            elif "functionResponse" in p:
                                fr_keys = sorted(p["functionResponse"].keys())
                                fc_diag.append(f"C{ci}P{pi}:FR({p['functionResponse'].get('name','?')},keys={fr_keys})")
                    log.warning(
                        "%d on %s: model=%s, tools=%d, contents=%d, FC/FR: [%s], resp=%s",
                        resp.status_code, endpoint, api_model, n_tools, n_contents,
                        ", ".join(fc_diag) if fc_diag else "none",
                        resp.text[:300],
                    )
                    # Also emit to events.jsonl so user can see it
                    try:
                        import pathlib as _pl
                        _drive_logs = _pl.Path(os.environ.get("OUROBOROS_DRIVE_LOGS", "/content/drive/MyDrive/Ouroboros/logs"))
                        append_jsonl(_drive_logs / "events.jsonl", {
                            "ts": utc_now_iso(), "type": "antigravity_api_diag",
                            "status": resp.status_code,
                            "model": api_model,
                            "n_contents": n_contents,
                            "n_tools": n_tools,
                            "fc_fr_diag": fc_diag,
                            "resp_text": resp.text[:500],
                        })
                    except Exception:
                        pass
                    last_error = f"{resp.status_code} from {endpoint}: {resp.text[:200]}"
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Antigravity wraps response: {response: {candidates, ...}, traceId, metadata}
                if "response" in data and "candidates" in data["response"]:
                    data = data["response"]

                msg = _google_to_openai_message(data)
                usage = _extract_usage(data)
                return msg, usage

            except requests.exceptions.Timeout:
                last_error = f"Timeout on {endpoint}"
                log.warning("Timeout on %s", endpoint)
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"Error on {endpoint}: {e}"
                log.warning("Request to %s failed: %s", endpoint, e)
                continue

        raise RuntimeError(f"All Antigravity endpoints failed. Last error: {last_error}")
