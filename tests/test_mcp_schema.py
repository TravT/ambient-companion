#!/usr/bin/env python3
"""Tool definitions must follow the MCP spec (inputSchema), or clients such as Claude Code drop the whole tool list.

Checked on every path a client can reach: the dispatch core, the HTTP service and the stdio proxy.
"""

import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import http_service
import mcp_proxy
import server

TOKEN = "schema-test-token"
LIST_REQ = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


class TestToolSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http_service.make_server("127.0.0.1", 0, token=TOKEN)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/mcp"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _via_dispatch(self):
        return server.dispatch(LIST_REQ)["result"]["tools"]

    def _via_http(self):
        req = urllib.request.Request(self.url, data=json.dumps(LIST_REQ).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {TOKEN}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())["result"]["tools"]

    def _via_proxy(self):
        return mcp_proxy.handle_line(json.dumps(LIST_REQ), self.url, TOKEN)["result"]["tools"]

    def _each_path(self):
        for label, fetch in (("dispatch", self._via_dispatch), ("http", self._via_http),
                             ("proxy", self._via_proxy)):
            yield label, fetch()

    def test_every_tool_has_an_object_input_schema(self):
        for label, tools in self._each_path():
            self.assertEqual(len(tools), 5, label)
            for t in tools:
                with self.subTest(path=label, tool=t["name"]):
                    self.assertIn("inputSchema", t)
                    self.assertEqual(t["inputSchema"]["type"], "object")
                    self.assertIn("properties", t["inputSchema"])

    def test_parameters_alias_matches_input_schema(self):
        for label, tools in self._each_path():
            for t in tools:
                with self.subTest(path=label, tool=t["name"]):
                    self.assertEqual(t["parameters"], t["inputSchema"])

    def test_required_fields_exist_in_properties(self):
        for t in self._via_dispatch():
            schema = t["inputSchema"]
            for field in schema.get("required", []):
                self.assertIn(field, schema["properties"], (t["name"], field))

    def test_valid_for_the_official_sdk_when_installed(self):
        try:
            from mcp.types import Tool
        except ImportError:
            self.skipTest("mcp SDK not installed")
        for label, tools in self._each_path():
            for t in tools:
                with self.subTest(path=label, tool=t["name"]):
                    Tool.model_validate(t)   # exactly what clients receive, alias included


if __name__ == "__main__":
    unittest.main()
