#!/usr/bin/env python3
"""Tests for the JSON-RPC dispatch core and the HTTP front-end (no hardware)."""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import daemon
import http_service
import server

EXPECTED_TOOLS = {
    "ambient_escalation_cycle", "ambient_triage_scene", "ambient_ocr_and_grounding",
    "ambient_speak", "ambient_hardware_status",
}


class TestDispatch(unittest.TestCase):
    def test_initialize(self):
        resp = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(resp["id"], 1)
        self.assertEqual(resp["result"]["serverInfo"]["name"], "ambient-companion")

    def test_tools_list_has_five_tools(self):
        resp = server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_notification_returns_none(self):
        self.assertIsNone(server.dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_is_error(self):
        resp = server.dispatch({"jsonrpc": "2.0", "id": 3, "method": "nope"})
        self.assertEqual(resp["error"]["code"], -32601)

    def test_unknown_notification_is_silent(self):
        self.assertIsNone(server.dispatch({"jsonrpc": "2.0", "method": "nope"}))

    def test_unknown_tool_is_tool_error(self):
        resp = server.dispatch({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                "params": {"name": "bogus", "arguments": {}}})
        self.assertTrue(resp["result"]["isError"])

    def test_cycle_tool_routes_to_daemon(self):
        with mock.patch.object(daemon, "execute_ambient_cycle", return_value={"status": "ok"}) as cyc:
            resp = server.dispatch({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
                "name": "ambient_escalation_cycle",
                "arguments": {"query": "q", "language": "pt", "play_audio": False}}})
        cyc.assert_called_once()
        self.assertEqual(cyc.call_args.kwargs["lang"], "pt")
        self.assertIn('"status": "ok"', resp["result"]["content"][0]["text"])

    def test_busy_hardware_returns_tool_error(self):
        # Another call holds the camera/ADB lock: answer "busy" instead of piling up.
        server.HARDWARE_LOCK.acquire()
        try:
            with mock.patch.object(server.config, "HARDWARE_LOCK_TIMEOUT_SEC", 0.05):
                resp = server.dispatch({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
                    "name": "ambient_escalation_cycle", "arguments": {"query": "q"}}})
        finally:
            server.HARDWARE_LOCK.release()
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("busy", resp["result"]["content"][0]["text"].lower())

    def test_hardware_status_does_not_take_the_lock(self):
        server.HARDWARE_LOCK.acquire()
        try:
            with mock.patch.object(server, "readiness", return_value={"adb": False, "edge_temp_c": None,
                                                                      "llama_server": False,
                                                                      "tts_available": False,
                                                                      "battery_level": None,
                                                                      "thermal_breaker": False}):
                resp = server.dispatch({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
                    "name": "ambient_hardware_status", "arguments": {}}})
        finally:
            server.HARDWARE_LOCK.release()
        self.assertNotIn("isError", resp["result"])


class TestReadiness(unittest.TestCase):
    def test_unreachable_edge_is_reported_not_raised(self):
        with mock.patch.object(daemon, "get_edge_temperature", return_value=None), \
                mock.patch.object(daemon.subprocess, "run", side_effect=OSError("no adb")), \
                mock.patch.object(daemon.requests, "get", side_effect=OSError("down")):
            report = server.readiness()
        self.assertFalse(report["adb"])
        self.assertIsNone(report["edge_temp_c"])
        self.assertFalse(report["llama_server"])
        self.assertFalse(report["thermal_breaker"])

    def test_hot_edge_engages_breaker(self):
        with mock.patch.object(daemon, "get_edge_temperature", return_value=44.0), \
                mock.patch.object(daemon.subprocess, "run",
                                  return_value=mock.Mock(stdout="  level: 80", returncode=0)), \
                mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)):
            report = server.readiness()
        self.assertTrue(report["adb"])
        self.assertTrue(report["thermal_breaker"])
        self.assertEqual(report["battery_level"], "80%")
        self.assertTrue(report["llama_server"])

    def test_llama_health_url_derived_from_chat_endpoint(self):
        with mock.patch.object(daemon, "LLAMA_SERVER_URL", "http://10.0.0.5:9000/v1/chat/completions"):
            self.assertEqual(server.llama_health_url(), "http://10.0.0.5:9000/health")


class HttpCase(unittest.TestCase):
    TOKEN = "s3cret"

    @classmethod
    def setUpClass(cls):
        cls.httpd = http_service.make_server("127.0.0.1", 0, token=cls.TOKEN)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, method="GET", body=None, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method=method,
            data=body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None),
            headers=headers or {},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode()


class TestHttp(HttpCase):
    def test_health_is_cheap_and_never_touches_adb(self):
        with mock.patch.object(daemon, "get_edge_temperature", side_effect=AssertionError("ADB touched")), \
                mock.patch.object(daemon.subprocess, "run", side_effect=AssertionError("ADB touched")):
            status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_ready_returns_report(self):
        report = {"adb": True, "edge_temp_c": 30.1, "llama_server": True, "tts_available": False,
                  "battery_level": "90%", "thermal_breaker": False}
        with mock.patch.object(server, "readiness", return_value=report):
            status, body = self.request("/ready")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["edge_temp_c"], 30.1)

    def test_mcp_requires_token(self):
        status, _ = self.request("/mcp", "POST", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(status, 401)

    def test_mcp_rejects_wrong_token(self):
        status, _ = self.request("/mcp", "POST", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                 {"Authorization": "Bearer nope"})
        self.assertEqual(status, 403)

    def test_mcp_tools_list_with_token(self):
        status, body = self.request("/mcp", "POST", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                    {"Authorization": f"Bearer {self.TOKEN}",
                                     "Content-Type": "application/json"})
        self.assertEqual(status, 200)
        names = {t["name"] for t in json.loads(body)["result"]["tools"]}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_mcp_bad_json_is_400(self):
        status, _ = self.request("/mcp", "POST", b"{not json", {"Authorization": f"Bearer {self.TOKEN}"})
        self.assertEqual(status, 400)

    def test_mcp_notification_is_202(self):
        status, _ = self.request("/mcp", "POST", {"jsonrpc": "2.0", "method": "notifications/initialized"},
                                 {"Authorization": f"Bearer {self.TOKEN}"})
        self.assertEqual(status, 202)

    def test_unknown_path_is_404(self):
        status, _ = self.request("/nope")
        self.assertEqual(status, 404)


class TestHttpNoToken(unittest.TestCase):
    def test_empty_token_disables_auth(self):
        httpd = http_service.make_server("127.0.0.1", 0, token="")
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/mcp", method="POST",
                data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode())
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
