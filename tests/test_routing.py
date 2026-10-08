#!/usr/bin/env python3
"""Routing: GPU-first when awake, question-type routing otherwise, no self-critique pass."""

import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import config
import daemon


class TestClassify(unittest.TestCase):
    def check(self, kind, *queries):
        for q in queries:
            self.assertEqual(daemon.classify_query(q), kind, q)

    def test_ocr(self):
        self.check("ocr", "Read the dosage on the bottle", "Leia o nome do medicamento",
                   "What does the label say?", "Is the text readable?")

    def test_presence(self):
        self.check("presence", "Is there a keyboard in view?", "Do you see a glass beaker?",
                   "Are there any people?", "Is the desk empty?", "Is anyone sitting in the office chair?",
                   "Tem um copo na mesa?", "Há alguém na sala?")

    def test_locate(self):
        self.check("locate", "Where is the remote?", "Locate the keyboard with bounding boxes",
                   "Onde está o controle?")

    def test_describe(self):
        self.check("describe", "What is on the desk?", "Describe the scene.", "O que tem na mesa?",
                   "How many cups are there?")


class TestNeedsEscalation(unittest.TestCase):
    def test_presence_answers_that_are_clear_stay_at_the_edge(self):
        for text in ("Yes.", "No.", "No, there is no keyboard present.", "Sim.", "Não."):
            self.assertFalse(daemon.needs_escalation(text), text)

    def test_empty_hedged_or_off_form_answers_escalate(self):
        for text in ("", "No response parsed", "I'm not sure.", "Maybe a cup", "Cup, bottle, paper.",
                     "Nothing.", "Não sei", "Yes, but unclear"):
            self.assertTrue(daemon.needs_escalation(text), text)


class TestGpuProbe(unittest.TestCase):
    def setUp(self):
        daemon._gpu_probe = (0.0, False)

    def test_awake_when_health_is_200(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)) as get:
            self.assertTrue(daemon.gpu_awake())
        self.assertTrue(get.call_args.args[0].endswith("/health"))
        self.assertIn("100.77.169.15", get.call_args.args[0])

    def test_asleep_when_unreachable_and_never_raises(self):
        with mock.patch.object(daemon.requests, "get", side_effect=OSError("down")):
            self.assertFalse(daemon.gpu_awake())

    def test_result_is_cached_briefly(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)) as get:
            daemon.gpu_awake()
            daemon.gpu_awake()
        self.assertEqual(get.call_count, 1)

    def test_prefer_gpu_off_never_probes(self):
        with mock.patch.object(config, "PREFER_GPU", False), \
                mock.patch.object(daemon.requests, "get", side_effect=AssertionError("probed")):
            self.assertFalse(daemon.gpu_awake())

    def test_endpoints_put_gpu_first_only_when_awake(self):
        with mock.patch.object(daemon, "gpu_awake", return_value=True):
            eps = daemon.tier2_endpoints()
        self.assertEqual(eps[0], daemon.FALLBACK_LLAMA_SERVER_URL)
        self.assertEqual(eps[1], daemon.LLAMA_SERVER_URL)
        with mock.patch.object(daemon, "gpu_awake", return_value=False):
            self.assertEqual(daemon.tier2_endpoints(), [daemon.LLAMA_SERVER_URL])


class TestPlanRoute(unittest.TestCase):
    def plan(self, query, crop=None, gpu=False):
        with mock.patch.object(daemon, "gpu_awake", return_value=gpu):
            return daemon.plan_route(query, crop)

    def test_gpu_awake_sends_everything_to_the_gpu_at_full_budget(self):
        for q in ("Is there a cup?", "What is on the desk?", "Read the label"):
            p = self.plan(q, gpu=True)
            self.assertEqual((p["backend"], p["tier2_px"]), ("gpu", 1024), q)

    def test_describe_and_locate_go_to_homelab_at_512(self):
        for q in ("What is on the desk?", "Where is the remote?"):
            p = self.plan(q)
            self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", 512), q)

    def test_reading_uses_the_read_budget(self):
        p = self.plan("Read the dosage")
        self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", config.READ_PX))

    def test_crop_goes_to_homelab_at_1024(self):
        p = self.plan("Is there a cup?", crop=[100, 100, 400, 400])
        self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", 1024))

    def test_presence_starts_on_the_phone(self):
        p = self.plan("Is there a keyboard?")
        self.assertEqual(p["backend"], "edge")
        self.assertEqual(p["tier2_px"], 512)


def _frame(tmp):
    from PIL import Image
    path = Path(tmp) / "f.jpg"
    Image.new("RGB", (800, 600), (90, 120, 150)).save(path)
    return path


class TestCycleRouting(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.frame = _frame(self.tmp.name)
        self.patches = [
            mock.patch.object(daemon, "get_edge_temperature", return_value=30.0),
            mock.patch.object(daemon, "BENCHMARK_DIR", Path(self.tmp.name)),
            mock.patch.object(daemon.optical_ingestion, "acquire_image", return_value=self.frame),
            mock.patch.object(daemon, "push_frame_to_edge"),
        ]
        self.mocks = [p.start() for p in self.patches]
        self.push = self.mocks[3]
        self.t2 = mock.patch.object(daemon, "query_tier2_qwen", return_value={
            "status": "success", "content": "A desk.", "duration_sec": 1.0,
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}).start()
        self.edge = mock.patch.object(daemon, "query_edge_smolvlm",
                                      return_value={"text": "Yes.", "duration_sec": 1.0, "raw_output": ""}).start()
        self.speak = mock.patch.object(daemon, "speak", return_value={"status": "success", "played": True}).start()

    def tearDown(self):
        mock.patch.stopall()
        self.tmp.cleanup()

    def run_cycle(self, query, gpu=False, play_audio=False, crop=None):
        with mock.patch.object(daemon, "gpu_awake", return_value=gpu):
            return daemon.execute_ambient_cycle(query, source="camera", lang="en",
                                                play_audio=play_audio, crop_bbox=crop)

    def test_gpu_awake_skips_the_phone_model_entirely(self):
        res = self.run_cycle("Is there a cup?", gpu=True)
        self.edge.assert_not_called()
        self.push.assert_not_called()
        self.assertEqual(res["resolution_tier"], "Tier 2 (GPU)")
        self.assertEqual(res["route"]["backend"], "gpu")

    def test_describe_goes_straight_to_homelab_without_the_phone(self):
        res = self.run_cycle("What is on the desk?")
        self.edge.assert_not_called()
        self.push.assert_not_called()
        self.assertEqual(self.t2.call_count, 1)
        self.assertEqual(res["resolution_tier"], "Tier 2 (Direct)")
        self.assertEqual(res["final_answer"], "A desk.")

    def test_presence_resolves_on_the_phone_with_exactly_one_model_call(self):
        res = self.run_cycle("Is there a cup?")
        self.assertEqual(self.edge.call_count, 1)      # no self-critique pass
        self.t2.assert_not_called()
        self.assertEqual(res["resolution_tier"], "Tier 1 (Edge On-Device)")
        self.assertEqual(res["final_answer"], "Yes.")

    def test_presence_escalates_when_the_phone_answer_is_unusable(self):
        self.edge.return_value = {"text": "Nothing.", "duration_sec": 1.0, "raw_output": ""}
        res = self.run_cycle("Is there a cup?")
        self.assertEqual(self.edge.call_count, 1)
        self.assertEqual(self.t2.call_count, 1)
        self.assertEqual(res["resolution_tier"], "Tier 2 (Escalation via Homelab)")

    def test_tier2_uses_the_planned_image_budget(self):
        sizes = []

        def fake_prepare(src, max_dim, dest, crop_bbox=None):
            sizes.append(max_dim)
            return dest, {"original_width": 800, "original_height": 600, "crop_applied": False}

        with mock.patch.object(daemon.optical_ingestion, "prepare_budgeted_image", fake_prepare):
            self.run_cycle("What is on the desk?")
        self.assertEqual(sizes, [512])

    def test_cue_is_spoken_while_tier2_runs_and_final_answer_after(self):
        order = []
        gate = threading.Event()

        def slow_tier2(*a, **k):
            gate.wait(timeout=2)           # the cue must be spoken while this is still running
            order.append("tier2_done")
            return {"status": "success", "content": "A desk.", "duration_sec": 1.0,
                    "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        def fake_speak(text, lang):
            order.append("cue" if "look" in text.lower() or "olhar" in text.lower() else "answer")
            if order[-1] == "cue":
                gate.set()
            return {"status": "success", "played": True}

        self.t2.side_effect = slow_tier2
        self.speak.side_effect = fake_speak
        self.run_cycle("What is on the desk?", play_audio=True)
        self.assertEqual(order, ["cue", "tier2_done", "answer"])

    def test_no_cue_when_the_gpu_answers_fast(self):
        self.run_cycle("What is on the desk?", gpu=True, play_audio=True)
        spoken = [c.args[0] for c in self.speak.call_args_list]
        self.assertEqual(len(spoken), 1)               # only the final answer

    def test_audio_off_never_speaks(self):
        self.run_cycle("What is on the desk?", play_audio=False)
        self.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
