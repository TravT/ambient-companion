#!/usr/bin/env python3
"""Fast follow-ups: image-first requests (server prompt cache), last-frame reuse, shorter spoken answers."""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from PIL import Image

import config
import daemon
import server


class TestImageFirst(unittest.TestCase):
    def test_tier2_request_puts_the_image_before_the_text(self):
        sent = {}

        def fake_post(url, json=None, timeout=None):
            sent["payload"] = json
            return mock.Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(daemon, "tier2_endpoints", return_value=["http://x/v1/chat/completions"]), \
                mock.patch.object(daemon.requests, "post", fake_post):
            img = Path(d) / "a.jpg"
            Image.new("RGB", (32, 32)).save(img)
            daemon.query_tier2_qwen(img, "What color?")
        content = sent["payload"]["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["image_url", "text"])
        self.assertEqual(content[1]["text"], "What color?")


class TestLastFrame(unittest.TestCase):
    def setUp(self):
        daemon._last_frame = None
        self.tmp = tempfile.TemporaryDirectory()
        self.img = Path(self.tmp.name) / "f.jpg"
        Image.new("RGB", (64, 64)).save(self.img)

    def tearDown(self):
        daemon._last_frame = None
        self.tmp.cleanup()

    def test_default_always_captures(self):
        with mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.img) as acq:
            daemon.acquire_frame("camera")
            daemon.acquire_frame("camera")
        self.assertEqual(acq.call_count, 2)

    def test_reuse_returns_the_previous_frame_without_capturing(self):
        with mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.img) as acq:
            first = daemon.acquire_frame("camera")
            again = daemon.acquire_frame("camera", reuse_last=True)
        self.assertEqual(acq.call_count, 1)
        self.assertEqual(first, again)

    def test_reuse_with_nothing_cached_captures(self):
        with mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.img) as acq:
            daemon.acquire_frame("camera", reuse_last=True)
        self.assertEqual(acq.call_count, 1)

    def test_stale_frame_is_not_reused(self):
        with mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.img) as acq:
            daemon.acquire_frame("camera")
            path, ts = daemon._last_frame
            daemon._last_frame = (path, ts - config.LAST_FRAME_TTL_SEC - 1)
            daemon.acquire_frame("camera", reuse_last=True)
        self.assertEqual(acq.call_count, 2)

    def test_a_different_source_is_not_served_from_the_cache(self):
        with mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.img) as acq:
            daemon.acquire_frame("camera")
            daemon.acquire_frame("/some/other/file.jpg", reuse_last=True)
        self.assertEqual(acq.call_count, 2)

    def test_tools_accept_reuse_last_frame(self):
        props = {t["name"]: t["parameters"]["properties"] for t in server.TOOLS}
        for name in ("ambient_escalation_cycle", "ambient_triage_scene", "ambient_ocr_and_grounding"):
            self.assertIn("reuse_last_frame", props[name], name)

    def test_cycle_tool_passes_the_flag_through(self):
        with mock.patch.object(daemon, "execute_ambient_cycle", return_value={"status": "ok"}) as cyc:
            server._run_tool("ambient_escalation_cycle", {"query": "q", "reuse_last_frame": True, "play_audio": False})
        self.assertTrue(cyc.call_args.kwargs["reuse_last_frame"])


class TestSpeechExcerpt(unittest.TestCase):
    def test_first_sentence_only_when_it_is_long_enough(self):
        text = "The image shows a street scene at night with a yellow car. There is a fence in front. It is raining."
        self.assertEqual(daemon.speech_excerpt(text, max_words=28), "The image shows a street scene at night with a yellow car.")

    def test_a_very_short_first_sentence_gets_the_second(self):
        self.assertEqual(daemon.speech_excerpt("Yes. A yellow car is parked outside. More text.", max_words=28),
                         "Yes. A yellow car is parked outside.")

    def test_long_sentence_is_cut_at_a_word_boundary(self):
        text = " ".join(["word"] * 60) + "."
        out = daemon.speech_excerpt(text, max_words=25)
        self.assertEqual(len(out.rstrip(".").split()), 25)

    def test_markdown_noise_is_removed(self):
        self.assertEqual(daemon.speech_excerpt("**Answer:** `ok`. Done.", max_words=28), "Answer: ok. Done.")

    def test_empty_stays_empty(self):
        self.assertEqual(daemon.speech_excerpt("", max_words=28), "")


if __name__ == "__main__":
    unittest.main()


class TestFollowUpRouting(unittest.TestCase):
    def plan(self, q, reuse, sat=True):
        with mock.patch.object(daemon, "backend_up", side_effect=lambda n: n == "satellite" and sat):
            return daemon.plan_route(q, None, reuse_last=reuse)

    def test_yes_no_follow_up_skips_the_phone_because_tier2_has_the_image_cached(self):
        p = self.plan("Are there any buildings?", reuse=True)
        self.assertEqual(p["backend"], "satellite")

    def test_a_fresh_yes_no_still_starts_on_the_phone(self):
        self.assertEqual(self.plan("Are there any buildings?", reuse=False)["backend"], "edge")

    def test_follow_up_works_on_the_dell_too(self):
        self.assertEqual(self.plan("Are there any buildings?", reuse=True, sat=False)["backend"], "homelab")


class TestChunkedSpeech(unittest.TestCase):
    def test_split_makes_a_short_first_chunk_and_keeps_all_words(self):
        text = "The scene shows a street with a yellow car parked on the side of the road, near a tall building at night."
        chunks = daemon.split_for_speech(text)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].split()), 9)
        self.assertEqual(" ".join(chunks).split(), text.split())
        self.assertTrue(all(len(c.split()) <= 13 for c in chunks))

    def test_short_text_is_one_chunk(self):
        self.assertEqual(daemon.split_for_speech("Yes, there is a car."), ["Yes, there is a car."])

    def test_second_chunk_is_synthesized_while_the_first_is_playing(self):
        first_playing = threading.Event()
        order = []

        def fake_edge(cmd, timeout=45, as_root=False):
            if "/tts" in cmd and "--form-string" in cmd:                 # synthesis of some chunk
                n = len([o for o in order if o.startswith("synth")]) + 1
                if n == 2:
                    self.assertTrue(first_playing.wait(timeout=5), "chunk 2 was not synthesized during chunk 1 playback")
                order.append(f"synth{n}")
                return "SYNTH_OK", 0.1
            if "paplay" in cmd:
                order.append("play")
                first_playing.set()
                time.sleep(0.05)
                return "RC=0", 0.1
            return "", 0.1

        text = "The scene shows a street with a yellow car parked on the side of the road near a tall building."
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", True), \
                mock.patch.object(daemon, "ensure_edge_tts_server", return_value=True), \
                mock.patch.object(daemon, "ensure_edge_audio", return_value=True), \
                mock.patch.object(daemon, "run_edge_command", fake_edge), \
                mock.patch.object(daemon.subprocess, "run", return_value=mock.Mock(returncode=0)):
            res = daemon.speak(text, "en")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["played"])
        self.assertEqual(res["where"], "s20-warm-chunked")
        self.assertIn("first_audio_sec", res)
        self.assertEqual(order[0], "synth1")
        self.assertLess(order.index("play"), order.index("synth2"))

    def test_failed_first_chunk_falls_back_to_the_single_shot_path(self):
        with mock.patch.object(config, "TTS_MODE", "edge"), mock.patch.object(config, "TTS_WARM", True), \
                mock.patch.object(daemon, "ensure_edge_tts_server", return_value=True), \
                mock.patch.object(daemon, "run_edge_command", return_value=("curl: (7) refused", 0.1)), \
                mock.patch.object(daemon, "speak_on_edge", return_value={"status": "error", "played": False}) as single:
            res = daemon.speak("The scene shows a street with a yellow car parked on the side of the road.", "en")
        single.assert_called_once()
        self.assertEqual(res["status"], "error")


class TestExplicitEndpoint(unittest.TestCase):
    def test_query_can_target_one_backend_for_benchmarking(self):
        seen = []

        def fake_post(url, json=None, timeout=None):
            seen.append(url)
            return mock.Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}], "usage": {}})

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(daemon, "tier2_endpoints", side_effect=AssertionError("ladder must not be used")), \
                mock.patch.object(daemon.requests, "post", fake_post):
            img = Path(d) / "a.jpg"
            Image.new("RGB", (32, 32)).save(img)
            res = daemon.query_tier2_qwen(img, "q", endpoints=["http://only-this/v1/chat/completions"])
        self.assertEqual(seen, ["http://only-this/v1/chat/completions"])
        self.assertEqual(res["endpoint"], "http://only-this/v1/chat/completions")
