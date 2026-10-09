#!/usr/bin/env python3
"""
stdio MCP proxy to THE ambient companion (PRJ-12).

There is one companion per lab: the Nomad service on the Dell (`ambient.home.arpa`). MCP clients that
only speak stdio (Antigravity, Claude Code) register this proxy instead of starting their own copy of the
server, so every client shares the same camera lock, voice, backend ladder and state.

Each JSON-RPC line on stdin is POSTed to the service; the response goes to stdout. Notifications produce no
output. If the service is down or rejects the request, a JSON-RPC error with the request id is returned.

  AMBIENT_MCP_URL    default http://127.0.0.1:8089/mcp (on the Dell; use http://ambient.home.arpa/mcp elsewhere)
  AMBIENT_API_TOKEN  bearer token; if unset it is read at start from the vault helper (never stored in a config)
"""

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

DEFAULT_URL = "http://127.0.0.1:8089/mcp"
REQUEST_TIMEOUT_SEC = 330          # a Tier 2 cycle on the CPU can take over two minutes
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def log(msg: str) -> None:
    print(f"[ambient-mcp-proxy] {msg}", file=sys.stderr, flush=True)


def resolve_token() -> str:
    token = os.environ.get("AMBIENT_API_TOKEN", "").strip()
    if token:
        return token
    helper = REPO_ROOT / "scripts" / "get_secret.py"
    try:
        res = subprocess.run([sys.executable, str(helper), "vault_ambient_api_token"],
                             capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""


def _error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_line(line: str, url: str, token: str) -> Optional[dict]:
    """Forward one JSON-RPC line. Returns the response to print, or None (notification / blank)."""
    line = line.strip()
    if not line:
        return None
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return _error(None, -32700, "Parse error")
    req_id = req.get("id") if isinstance(req, dict) else None
    is_notification = isinstance(req, dict) and "id" not in req

    http_req = urllib.request.Request(
        url, data=line.encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    try:
        with urllib.request.urlopen(http_req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            body = resp.read()
            if is_notification or resp.status == 202 or not body:
                return None
            return json.loads(body)
    except urllib.error.HTTPError as err:
        if is_notification:
            return None
        hint = {401: "missing token", 403: "token rejected"}.get(err.code, f"HTTP {err.code}")
        return _error(req_id, -32000, f"ambient-companion service unavailable: {hint}")
    except Exception as err:  # connection refused, DNS, timeout, bad JSON from the service
        if is_notification:
            return None
        return _error(req_id, -32000, f"ambient-companion service unavailable: {type(err).__name__}")


def main() -> int:
    url = os.environ.get("AMBIENT_MCP_URL", DEFAULT_URL)
    token = resolve_token()
    if not token:
        log("no API token found (set AMBIENT_API_TOKEN or fix scripts/get_secret.py); requests will be rejected")
    out_lock = threading.Lock()

    def work(line: str) -> None:
        resp = handle_line(line, url, token)
        if resp is not None:
            with out_lock:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for raw in sys.stdin:           # exits when the client closes stdin
            pool.submit(work, raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
