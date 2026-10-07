#!/usr/bin/env python3
"""
HTTP front-end for the ambient companion (PRJ-12 / ADR-42).

  GET  /health  liveness only, never touches ADB (used by Nomad check_restart)
  GET  /ready   live ADB / thermal / llama-server / TTS state as JSON
  POST /mcp     one JSON-RPC 2.0 request (same tools as the stdio server), bearer-token protected

Stdlib only. `python3 server.py serve` starts it; the stdio server is unchanged.
"""

import hmac
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import config
import housekeeping
import server

MAX_BODY_BYTES = 1024 * 1024


def _make_handler(token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ambient-companion/{server.SERVER_VERSION}"

        def log_message(self, fmt, *args):  # keep container logs short
            server.log_debug("%s %s" % (self.address_string(), fmt % args))

        def _send(self, status: int, payload: Optional[dict] = None):
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "version": server.SERVER_VERSION})
            elif self.path == "/ready":
                self._send(200, server.readiness())
            else:
                self._send(404, {"error": "not found"})

        def _authorized(self) -> Optional[int]:
            """None when allowed, else the HTTP status to answer with."""
            if not token:
                return None
            header = self.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return 401
            if not hmac.compare_digest(header[len("Bearer "):], token):
                return 403
            return None

        def do_POST(self):
            if self.path != "/mcp":
                self._send(404, {"error": "not found"})
                return
            denied = self._authorized()
            if denied:
                self._send(denied, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(400, {"error": "invalid body size"})
                return
            try:
                req = json.loads(self.rfile.read(length))
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
            except ValueError:
                self._send(400, {"error": "invalid JSON"})
                return
            try:
                resp = server.dispatch(req)
            except Exception as err:  # dispatch should not raise; never leak a traceback
                server.log_debug(f"dispatch error: {err!r}")
                self._send(500, {"jsonrpc": "2.0", "id": req.get("id"),
                                 "error": {"code": -32603, "message": "internal error"}})
                return
            if resp is None:
                self._send(202)
            else:
                self._send(200, resp)

    return Handler


def make_server(host: str, port: int, token: str = "") -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _make_handler(token))
    httpd.daemon_threads = True
    return httpd


def serve() -> None:
    host, port, token = config.SERVE_HOST, config.SERVE_PORT, config.API_TOKEN
    if not token and host not in ("127.0.0.1", "localhost", "::1"):
        server.log_debug("WARNING: AMBIENT_API_TOKEN is empty and the listener is not loopback-only")
    httpd = make_server(host, port, token)
    housekeeping.start_background()

    def _stop(signum, _frame):
        server.log_debug(f"signal {signum}: shutting down")
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    server.log_debug(f"HTTP service on {host}:{port} (auth {'on' if token else 'off'})")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    sys.exit(0)
