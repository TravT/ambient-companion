#!/usr/bin/env python3
"""Speech generated on the Dell (ambient-speech), played on the phone, and push-to-talk (no hardware, no models)."""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import config
import daemon
import server

RIFF = b"RIFF" + b"\x00" * 60


class FakeResp:
    def __init__(self, status=200, content=b"", payload=None):
        self.status_code, self.content, self._payload = status, content, payload

    def json(self):
        return self._payload


class TestSpeakOnDell(unittest.TestCase):
    def setUp(self):
        self.pushed, self.played, self.removed = [], [], []
        patches = [
            mock.patch.object(config, "TTS_MODE", "dell"),
            mock.patch.object(daemon, "speech_service_up", return_value=True),
            mock.patch.object(daemon, "ensure_edge_audio", return_value=True),
            mock.patch.object(daemon, "_push_wav_to_edge",
                              side_effect=lambda wav, name: self.pushed.append(name) or f"/data/local/tmp/{name}"),
            mock.patch.object(daemon, "_play_wav_on_edge",
                              side_effect=lambda remote: self.played.append(remote) or (True, "")),
            mock.patch.object(daemon, "_remove_on_edge", side_effect=lambda remote: self.removed.append(remote)),
            mock.patch.object(daemon.subprocess, "run"),                      # volume command
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_every_chunk_is_synthesized_on_the_dell_and_played_in_order(self):
        texts = []
        with mock.patch.object(daemon, "synthesize_on_dell", side_effect=lambda t, l: texts.append((t, l)) or RIFF):
            res = daemon.speak("Yes. The label says ibuprofen, five hundred milligrams, one tablet every eight hours.", "en")
        self.assertTrue(res["played"])
        self.assertEqual(res["where"], "dell")
        self.assertGreater(len(texts), 1)                                     # chunked: audio starts early
        self.assertTrue(all(lang == "en" for _, lang in texts))
        self.assertEqual(self.played, [f"/data/local/tmp/{n}" for n in self.pushed])
        self.assertEqual(sorted(self.removed), sorted(self.played))           # nothing left on the phone
        self.assertIn("first_audio_sec", res)

    def test_portuguese_goes_to_the_portuguese_voice(self):
        langs = []
        with mock.patch.object(daemon, "synthesize_on_dell", side_effect=lambda t, l: langs.append(l) or RIFF):
            daemon.speak("Sim, há um copo.", "pt")
        self.assertEqual(set(langs), {"pt"})

    def test_falls_back_to_phone_voice_when_the_service_is_down(self):
        with mock.patch.object(daemon, "speech_service_up", return_value=False), \
                mock.patch.object(daemon, "_speak_edge", return_value={"status": "success", "played": True,
                                                                       "where": "s20-warm"}) as edge:
            res = daemon.speak("Yes.", "en")
        edge.assert_called_once()
        self.assertTrue(res["played"])
        self.assertEqual(res["fallback_from"], "dell")

    def test_falls_back_to_phone_voice_when_the_first_chunk_fails(self):
        with mock.patch.object(daemon, "synthesize_on_dell", return_value=None), \
                mock.patch.object(daemon, "_speak_edge", return_value={"status": "success", "played": True,
                                                                       "where": "s20"}) as edge:
            res = daemon.speak("Yes, there is.", "en")
        edge.assert_called_once()
        self.assertEqual(res["fallback_from"], "dell")

    def test_phone_audio_down_is_reported_not_retried_elsewhere(self):
        with mock.patch.object(daemon, "synthesize_on_dell", return_value=RIFF), \
                mock.patch.object(daemon, "ensure_edge_audio", return_value=False), \
                mock.patch.object(daemon, "_speak_edge") as edge:
            res = daemon.speak("Yes, there is.", "en")
        edge.assert_not_called()
        self.assertFalse(res["played"])
        self.assertIn("PulseAudio", res["error"])

    def test_tts_available_follows_the_service_then_the_phone(self):
        self.assertTrue(daemon.tts_available())
        with mock.patch.object(daemon, "speech_service_up", return_value=False), \
                mock.patch.object(daemon, "run_edge_command", return_value=("/data/.../pocket-tts", 0.1)):
            self.assertTrue(daemon.tts_available())


class TestSpeechClient(unittest.TestCase):
    def test_synthesize_sends_text_language_and_token(self):
        with mock.patch.object(config, "SPEECH_TOKEN", "tok"), \
                mock.patch.object(daemon.requests, "post", return_value=FakeResp(200, RIFF)) as post:
            self.assertEqual(daemon.synthesize_on_dell("hello", "en"), RIFF)
        url, = post.call_args.args
        self.assertTrue(url.endswith("/tts"))
        self.assertEqual(post.call_args.kwargs["json"], {"text": "hello", "language": "en"})
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer tok")

    def test_synthesize_returns_none_on_errors(self):
        with mock.patch.object(daemon.requests, "post", return_value=FakeResp(500, b"{}")):
            self.assertIsNone(daemon.synthesize_on_dell("hello", "en"))
        with mock.patch.object(daemon.requests, "post", side_effect=OSError("refused")):
            self.assertIsNone(daemon.synthesize_on_dell("hello", "en"))

    def test_transcribe_posts_wav_and_returns_text(self):
        payload = {"text": "is there a cup", "language": "en", "language_probability": 0.97}
        with mock.patch.object(daemon.requests, "post", return_value=FakeResp(200, payload=payload)) as post:
            res = daemon.transcribe_on_dell(b"RIFFwav", "auto")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["text"], "is there a cup")
        self.assertIn("language=auto", post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs["data"], b"RIFFwav")

    def test_transcribe_error_is_reported(self):
        with mock.patch.object(daemon.requests, "post", side_effect=OSError("refused")):
            self.assertEqual(daemon.transcribe_on_dell(b"x")["status"], "error")


class TestListen(unittest.TestCase):
    def test_spoken_question_is_answered_about_the_camera_view(self):
        cap = {"wav": b"RIFF", "reason": "end_of_speech", "speech_sec": 1.4, "total_sec": 2.6}
        stt = {"status": "success", "text": "Is there a cup on the desk?", "language": "en"}
        with mock.patch("listen.capture_utterance", return_value=cap), \
                mock.patch.object(daemon, "transcribe_on_dell", return_value=stt), \
                mock.patch.object(daemon, "execute_ambient_cycle", return_value={"final_answer": "Yes."}) as cycle:
            res = daemon.listen_and_answer()
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["transcript"]["text"], "Is there a cup on the desk?")
        self.assertEqual(cycle.call_args.kwargs["query"], "Is there a cup on the desk?")
        self.assertEqual(cycle.call_args.kwargs["lang"], "en")
        self.assertNotIn("wav", res["capture"])                               # audio never leaves memory

    def test_transcript_only_when_answer_is_false(self):
        cap = {"wav": b"RIFF", "reason": "end_of_speech", "speech_sec": 1.0, "total_sec": 2.0}
        with mock.patch("listen.capture_utterance", return_value=cap), \
                mock.patch.object(daemon, "transcribe_on_dell", return_value={"status": "success", "text": "oi", "language": "pt"}), \
                mock.patch.object(daemon, "execute_ambient_cycle") as cycle:
            res = daemon.listen_and_answer(answer=False)
        cycle.assert_not_called()
        self.assertEqual(res["transcript"]["text"], "oi")

    def test_no_speech_heard(self):
        cap = {"wav": None, "reason": "no_speech", "speech_sec": 0.0, "total_sec": 4.0}
        with mock.patch("listen.capture_utterance", return_value=cap), \
                mock.patch.object(daemon, "transcribe_on_dell") as stt, \
                mock.patch.object(daemon, "speak") as spoken:
            res = daemon.listen_and_answer(play_audio=False)
        stt.assert_not_called()
        spoken.assert_not_called()
        self.assertEqual(res["status"], "no_speech")


class TestListenTool(unittest.TestCase):
    def test_tool_is_listed_with_a_schema_and_holds_the_hardware_lock(self):
        tools = {t["name"]: t for t in server.TOOLS}
        self.assertIn("ambient_listen", tools)
        self.assertEqual(tools["ambient_listen"]["inputSchema"]["type"], "object")
        self.assertIn("ambient_listen", server.HARDWARE_TOOLS)

    def test_tool_returns_the_result_as_json(self):
        with mock.patch.object(daemon, "listen_and_answer", return_value={"status": "success", "transcript": {"text": "hi"}}) as la:
            res = server.handle_tool_call("ambient_listen", {"answer": False, "language": "pt"})
        self.assertNotIn("isError", res)
        self.assertEqual(json.loads(res["content"][0]["text"])["transcript"]["text"], "hi")
        self.assertEqual(la.call_args.kwargs["language"], "pt")
        self.assertFalse(la.call_args.kwargs["answer"])


if __name__ == "__main__":
    unittest.main()


class TestListenHttp(unittest.TestCase):
    TOKEN = "listen-test-token"

    @classmethod
    def setUpClass(cls):
        import threading
        import http_service
        cls.httpd = http_service.make_server("127.0.0.1", 0, token=cls.TOKEN)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/listen"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body: bytes, token=None):
        import urllib.error
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_needs_the_token(self):
        self.assertEqual(self.post(b"{}")[0], 401)

    def test_runs_push_to_talk_and_returns_json(self):
        with mock.patch.object(daemon, "listen_and_answer",
                               return_value={"status": "success", "transcript": {"text": "tem um copo?"}}) as la:
            status, body = self.post(json.dumps({"language": "pt"}).encode(), self.TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(body["transcript"]["text"], "tem um copo?")
        self.assertEqual(la.call_args.kwargs["language"], "pt")

    def test_empty_body_uses_defaults(self):
        with mock.patch.object(daemon, "listen_and_answer", return_value={"status": "no_speech"}) as la:
            req_status, body = self.post(b"", self.TOKEN)
        self.assertEqual(req_status, 200)
        self.assertEqual(body["status"], "no_speech")
        self.assertTrue(la.call_args.kwargs["answer"])

    def test_busy_hardware_is_409(self):
        with mock.patch.object(server, "handle_tool_call",
                               return_value={"isError": True, "content": [{"type": "text", "text": "Companion hardware is busy with another request; retry shortly."}]}):
            status, body = self.post(b"{}", self.TOKEN)
        self.assertEqual(status, 409)
        self.assertIn("busy", body["error"])
