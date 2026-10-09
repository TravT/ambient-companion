#!/usr/bin/env python3
"""mcp_proxy: a stdio MCP server that forwards to THE companion service (one companion per lab)."""

import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import http_service
import mcp_proxy

TOKEN = "proxy-test-token"


class ServiceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http_service.make_server("127.0.0.1", 0, token=TOKEN)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/mcp"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()


class TestHandleLine(ServiceCase):
    def test_forwards_a_request_and_returns_the_service_response(self):
        resp = mcp_proxy.handle_line(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}), self.url, TOKEN)
        self.assertEqual(resp["id"], 7)
        self.assertEqual(len(resp["result"]["tools"]), 5)

    def test_notifications_produce_no_output(self):
        line = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertIsNone(mcp_proxy.handle_line(line, self.url, TOKEN))

    def test_wrong_token_becomes_a_jsonrpc_error_with_the_request_id(self):
        resp = mcp_proxy.handle_line(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}), self.url, "nope")
        self.assertEqual(resp["id"], 3)
        self.assertEqual(resp["error"]["code"], -32000)
        self.assertNotIn("nope", json.dumps(resp))               # never echo the token

    def test_service_down_becomes_a_clear_error(self):
        resp = mcp_proxy.handle_line(json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/list"}),
                                     "http://127.0.0.1:9/mcp", TOKEN)
        self.assertEqual(resp["id"], 4)
        self.assertIn("unavailable", resp["error"]["message"])

    def test_garbage_line_is_a_parse_error(self):
        resp = mcp_proxy.handle_line("{not json", self.url, TOKEN)
        self.assertEqual(resp["error"]["code"], -32700)
        self.assertIsNone(resp["id"])

    def test_blank_line_is_ignored(self):
        self.assertIsNone(mcp_proxy.handle_line("   ", self.url, TOKEN))


class TestToken(unittest.TestCase):
    def test_env_token_wins(self):
        with mock.patch.dict(os.environ, {"AMBIENT_API_TOKEN": "from-env"}):
            self.assertEqual(mcp_proxy.resolve_token(), "from-env")

    def test_falls_back_to_the_vault_helper(self):
        fake = mock.Mock(stdout="from-vault\n", returncode=0)
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(mcp_proxy.subprocess, "run", return_value=fake) as run:
            os.environ.pop("AMBIENT_API_TOKEN", None)
            self.assertEqual(mcp_proxy.resolve_token(), "from-vault")
        self.assertIn("vault_ambient_api_token", run.call_args.args[0])

    def test_no_token_anywhere_is_empty_not_a_crash(self):
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(mcp_proxy.subprocess, "run", side_effect=OSError("no helper")):
            os.environ.pop("AMBIENT_API_TOKEN", None)
            self.assertEqual(mcp_proxy.resolve_token(), "")


class TestStdioEndToEnd(ServiceCase):
    def test_a_real_subprocess_speaks_mcp_over_stdio(self):
        env = dict(os.environ, AMBIENT_MCP_URL=self.url, AMBIENT_API_TOKEN=TOKEN)
        proc = subprocess.Popen([sys.executable, str(PKG_DIR / "mcp_proxy.py")], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        try:
            for i, method in enumerate(("initialize", "tools/list"), 1):
                proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": {}}) + "\n")
                proc.stdin.flush()
                resp = json.loads(proc.stdout.readline())
                self.assertEqual(resp["id"], i)
                self.assertIn("result", resp)
            self.assertEqual(resp["result"]["tools"][0]["name"], "ambient_escalation_cycle")
        finally:
            proc.stdin.close()
            proc.wait(timeout=10)
        self.assertEqual(proc.returncode, 0)                      # exits cleanly when the client closes stdin


if __name__ == "__main__":
    unittest.main()
