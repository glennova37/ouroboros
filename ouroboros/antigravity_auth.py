"""Antigravity OAuth — Google account login for free Gemini/Claude access.

Flow:
  1. Generate OAuth URL → user opens in browser
  2. User logs in → redirect to localhost with ?code=...&state=...
  3. Exchange code for refresh_token + access_token
  4. Store refresh_token; refresh access_token on demand
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from opencode-antigravity-auth)
# ---------------------------------------------------------------------------

CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
CLIENT_SECRET = "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"

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

REDIRECT_URI = "http://localhost:51121/oauth-callback"
CALLBACK_PORT = 51121

TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"

ENDPOINTS_LOAD = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://autopush-cloudcode-pa.sandbox.googleapis.com",
]

DEFAULT_PROJECT_ID = "rising-fact-p41fc"

_TOKEN_DIR = Path(os.environ.get("OUROBOROS_TOKEN_DIR", "~/.ouroboros")).expanduser()
_TOKEN_FILE = _TOKEN_DIR / "antigravity_tokens.json"


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _encode_state(verifier: str, project_id: str = "") -> str:
    """Encode state as base64url JSON (matches opencode-antigravity-auth)."""
    payload = json.dumps({"verifier": verifier, "projectId": project_id})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_state(state: str) -> Tuple[str, str]:
    """Decode state back to (verifier, project_id)."""
    padded = state + "=" * ((4 - len(state) % 4) % 4)
    normalized = padded.replace("-", "+").replace("_", "/")
    data = json.loads(base64.b64decode(normalized).decode("utf-8"))
    return data["verifier"], data.get("projectId", "")


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def _load_stored() -> Dict[str, Any]:
    if _TOKEN_FILE.exists():
        try:
            return json.loads(_TOKEN_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_stored(data: Dict[str, Any]) -> None:
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        _TOKEN_FILE.chmod(0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# OAuth URL generation
# ---------------------------------------------------------------------------

def get_auth_url() -> Tuple[str, str]:
    """Build OAuth URL. Returns (url, verifier)."""
    verifier, challenge = _generate_pkce()
    state = _encode_state(verifier)
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"{AUTH_ENDPOINT}?{urlencode(params)}"
    return url, verifier


# ---------------------------------------------------------------------------
# Token exchange & refresh
# ---------------------------------------------------------------------------

def _http_post_form(url: str, data: Dict[str, str], timeout: int = 15) -> Dict[str, Any]:
    """POST form-urlencoded, return JSON."""
    import requests
    resp = requests.post(
        url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "User-Agent": "google-api-nodejs-client/9.15.1",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def exchange_code(code: str, verifier: str) -> Dict[str, Any]:
    """Exchange authorization code for tokens."""
    token_data = _http_post_form(TOKEN_ENDPOINT, {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    })

    refresh_token = token_data.get("refresh_token")
    access_token = token_data.get("access_token")
    expires_in = int(token_data.get("expires_in", 3600))

    if not refresh_token or not access_token:
        raise ValueError(f"Missing tokens in response: {list(token_data.keys())}")

    # Get user email
    email = _fetch_email(access_token)

    # Get project ID
    project_id = _fetch_project_id(access_token)

    result = {
        "refresh_token": refresh_token,
        "access_token": access_token,
        "expires_at": time.time() + expires_in - 60,  # 60s safety margin
        "email": email,
        "project_id": project_id,
    }
    _save_stored(result)
    log.info("Antigravity auth: logged in as %s (project: %s)", email, project_id)
    return result


def refresh_access_token(refresh_token: Optional[str] = None) -> str:
    """Refresh access token. Uses stored refresh_token if not provided."""
    stored = _load_stored()
    rt = refresh_token or stored.get("refresh_token", "")
    if not rt:
        raise RuntimeError("No refresh_token. Run antigravity_auth.login() first.")

    token_data = _http_post_form(TOKEN_ENDPOINT, {
        "grant_type": "refresh_token",
        "refresh_token": rt,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })

    access_token = token_data["access_token"]
    expires_in = int(token_data.get("expires_in", 3600))

    # Update stored tokens
    stored["access_token"] = access_token
    stored["expires_at"] = time.time() + expires_in - 60
    if token_data.get("refresh_token"):
        stored["refresh_token"] = token_data["refresh_token"]
    _save_stored(stored)

    return access_token


def get_access_token() -> str:
    """Get a valid access token, refreshing if expired."""
    stored = _load_stored()
    if not stored.get("refresh_token"):
        raise RuntimeError("Not logged in. Run antigravity_auth.login() first.")

    expires_at = stored.get("expires_at", 0)
    if stored.get("access_token") and time.time() < expires_at:
        return stored["access_token"]

    return refresh_access_token(stored["refresh_token"])


def get_project_id() -> str:
    """Get stored project ID."""
    stored = _load_stored()
    return stored.get("project_id", DEFAULT_PROJECT_ID)


# ---------------------------------------------------------------------------
# Helpers: email & project ID
# ---------------------------------------------------------------------------

def _fetch_email(access_token: str) -> str:
    import requests
    try:
        resp = requests.get(
            USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("email", "")
    except Exception:
        log.debug("Failed to fetch email", exc_info=True)
    return ""


def _fetch_project_id(access_token: str) -> str:
    """Resolve Antigravity project ID via loadCodeAssist."""
    import requests
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "Client-Metadata": '{"ideType":"ANTIGRAVITY","platform":"MACOS","pluginType":"GEMINI"}',
    }
    body = json.dumps({
        "metadata": {
            "ideType": "ANTIGRAVITY",
            "platform": "MACOS",
            "pluginType": "GEMINI",
        }
    })

    for endpoint in ENDPOINTS_LOAD:
        try:
            resp = requests.post(
                f"{endpoint}/v1internal:loadCodeAssist",
                headers=headers,
                data=body,
                timeout=10,
            )
            if not resp.ok:
                continue
            data = resp.json()
            pid = data.get("cloudaicompanionProject", "")
            if isinstance(pid, dict):
                pid = pid.get("id", "")
            if pid:
                return pid
        except Exception:
            continue

    log.warning("Could not resolve project ID, using default: %s", DEFAULT_PROJECT_ID)
    return DEFAULT_PROJECT_ID


# ---------------------------------------------------------------------------
# Login flows
# ---------------------------------------------------------------------------

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth redirect callback on localhost."""
    code: Optional[str] = None
    state: Optional[str] = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        _OAuthCallbackHandler.code = params.get("code", [None])[0]
        _OAuthCallbackHandler.state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>&#10004; Authenticated! You can close this tab.</h1>")

    def log_message(self, format, *args):
        pass  # suppress server logs


def login() -> Dict[str, Any]:
    """Interactive OAuth login. Opens browser, waits for callback."""
    import webbrowser

    url, verifier = get_auth_url()

    print("\n" + "=" * 60)
    print("ANTIGRAVITY LOGIN")
    print("=" * 60)
    print(f"\nOpen this URL in your browser:\n\n{url}\n")

    try:
        webbrowser.open(url)
        print("(Browser should open automatically)")
    except Exception:
        print("(Could not open browser — copy the URL manually)")

    print("\nWaiting for callback on localhost:%d ..." % CALLBACK_PORT)

    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _OAuthCallbackHandler)
    server.handle_request()
    server.server_close()

    if not _OAuthCallbackHandler.code:
        raise RuntimeError("No authorization code received. Try login_manual().")

    # Decode state to get verifier (in case it was modified)
    if _OAuthCallbackHandler.state:
        verifier, _ = _decode_state(_OAuthCallbackHandler.state)

    return exchange_code(_OAuthCallbackHandler.code, verifier)


def login_manual() -> Dict[str, Any]:
    """Manual login for headless environments (Colab, SSH).

    User copies the callback URL from the browser and pastes it here.
    """
    url, verifier = get_auth_url()

    print("\n" + "=" * 60)
    print("ANTIGRAVITY LOGIN (manual mode)")
    print("=" * 60)
    print(f"\n1. Open this URL in your browser:\n\n{url}\n")
    print("2. Log in with your Google account")
    print("3. After redirect, copy the FULL URL from the browser address bar")
    print("   (it will start with http://localhost:51121/oauth-callback?code=...)")
    print()

    callback_url = input("Paste the callback URL here: ").strip()
    if not callback_url:
        raise RuntimeError("Empty URL")

    parsed = urlparse(callback_url)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]

    if not code:
        raise RuntimeError(f"No 'code' parameter found in URL: {callback_url}")

    if state:
        verifier, _ = _decode_state(state)

    return exchange_code(code, verifier)


def is_logged_in() -> bool:
    """Check if we have stored credentials."""
    stored = _load_stored()
    return bool(stored.get("refresh_token"))
