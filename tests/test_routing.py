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

    def test_locate_wins_over_incidental_reading_words(self):
        # "text" / "label" name the thing being found; they are not a request to read it.
        self.check("locate", "Where is the Username text input field?", "Where is the label?",
                   "Where is the 'Forgot password?' link?", "Onde está o texto de erro?")

    def test_reading_verbs_still_win_over_locate(self):
        self.check("ocr", "Read the label", "Where can I read the dosage?", "Transcribe the text",
                   "Leia o rótulo", "What does the label say?")

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


class TestBackendLadder(unittest.TestCase):
    def setUp(self):
        daemon._probe_cache.clear()

    def test_awake_when_health_is_200(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)) as get:
            self.assertTrue(daemon.backend_up("gpu"))
        self.assertTrue(get.call_args.args[0].endswith("/health"))
        self.assertIn("100.102.231.37", get.call_args.args[0])      # Omarchy side is probed first

    def test_gpu_list_defaults_to_omarchy_then_windows(self):
        self.assertEqual([u.split("//")[1].split(":")[0] for u in daemon.GPU_SERVER_URLS],
                         ["100.102.231.37", "100.77.169.15"])
        self.assertEqual(daemon.FALLBACK_LLAMA_SERVER_URL, daemon.GPU_SERVER_URLS[0])

    def test_windows_side_is_used_when_only_it_is_up(self):
        def fake_get(url, timeout=None):
            return mock.Mock(status_code=200 if "100.77.169.15" in url else 500)
        with mock.patch.object(daemon.requests, "get", fake_get):
            self.assertTrue(daemon.backend_up("gpu"))
            self.assertIn("100.77.169.15", daemon.active_gpu_url())
            self.assertEqual(daemon.tier2_endpoints()[0], daemon.GPU_SERVER_URLS[1])

    def test_omarchy_side_wins_when_both_answer(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)):
            daemon.backend_up("gpu")
            self.assertIn("100.102.231.37", daemon.active_gpu_url())

    def test_both_gpu_sides_down_is_not_up(self):
        with mock.patch.object(daemon.requests, "get", side_effect=OSError("down")):
            self.assertFalse(daemon.backend_up("gpu"))
            self.assertIsNone(daemon.active_gpu_url())

    def test_url_list_parsing(self):
        self.assertEqual(config.parse_url_list(" http://a/v1/x , ,http://b/v1/x,http://a/v1/x "),
                         ["http://a/v1/x", "http://b/v1/x"])
        self.assertEqual(config.parse_url_list(""), [])

    def test_satellite_probe_uses_its_own_url(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)) as get:
            self.assertTrue(daemon.backend_up("satellite"))
        self.assertIn("100.105.6.62:8090", get.call_args.args[0])

    def test_asleep_when_unreachable_and_never_raises(self):
        with mock.patch.object(daemon.requests, "get", side_effect=OSError("down")):
            self.assertFalse(daemon.backend_up("gpu"))
            self.assertFalse(daemon.backend_up("satellite"))

    def test_result_is_cached_per_backend(self):
        with mock.patch.object(daemon.requests, "get", return_value=mock.Mock(status_code=200)) as get:
            daemon.backend_up("gpu")
            daemon.backend_up("gpu")
            daemon.backend_up("satellite")
        self.assertEqual(get.call_count, 2)

    def test_prefer_gpu_off_never_probes_the_gpu(self):
        with mock.patch.object(config, "PREFER_GPU", False), \
                mock.patch.object(daemon.requests, "get", side_effect=AssertionError("probed")):
            self.assertFalse(daemon.backend_up("gpu"))

    def test_unset_satellite_url_disables_it(self):
        with mock.patch.object(daemon, "SATELLITE_LLAMA_SERVER_URL", ""), \
                mock.patch.object(daemon.requests, "get", side_effect=AssertionError("probed")):
            self.assertFalse(daemon.backend_up("satellite"))

    def _up(self, **state):
        return mock.patch.object(daemon, "backend_up", side_effect=lambda name: state.get(name, False))

    def test_endpoints_order_gpu_satellite_dell(self):
        with self._up(gpu=True, satellite=True):
            self.assertEqual(daemon.tier2_endpoints(), [
                daemon.FALLBACK_LLAMA_SERVER_URL, daemon.SATELLITE_LLAMA_SERVER_URL, daemon.LLAMA_SERVER_URL])
        with self._up(satellite=True):
            self.assertEqual(daemon.tier2_endpoints(), [daemon.SATELLITE_LLAMA_SERVER_URL, daemon.LLAMA_SERVER_URL])
        with self._up(gpu=True):
            self.assertEqual(daemon.tier2_endpoints(), [daemon.FALLBACK_LLAMA_SERVER_URL, daemon.LLAMA_SERVER_URL])
        with self._up():
            self.assertEqual(daemon.tier2_endpoints(), [daemon.LLAMA_SERVER_URL])

    def test_best_backend_names_the_first_that_is_up(self):
        with self._up(gpu=True, satellite=True):
            self.assertEqual(daemon.best_backend(), "gpu")
        with self._up(satellite=True):
            self.assertEqual(daemon.best_backend(), "satellite")
        with self._up():
            self.assertEqual(daemon.best_backend(), "homelab")

    def test_failing_backend_falls_through_to_the_next_endpoint(self):
        seen = []

        def fake_post(url, json=None, timeout=None):
            seen.append(url)
            if url == daemon.SATELLITE_LLAMA_SERVER_URL:
                raise daemon.requests.exceptions.ConnectionError("satellite went to sleep")
            return mock.Mock(status_code=200, json=lambda: {
                "choices": [{"message": {"content": "ok"}}], "usage": {}})

        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(daemon, "tier2_endpoints",
                                  return_value=[daemon.SATELLITE_LLAMA_SERVER_URL, daemon.LLAMA_SERVER_URL]), \
                mock.patch.object(daemon.requests, "post", fake_post):
            img = Path(d) / "x.jpg"
            Image.new("RGB", (64, 64)).save(img)
            res = daemon.query_tier2_qwen(img, "q")
        self.assertEqual(res["status"], "success")
        self.assertEqual(seen, [daemon.SATELLITE_LLAMA_SERVER_URL, daemon.LLAMA_SERVER_URL])
        self.assertEqual(res["endpoint"], daemon.LLAMA_SERVER_URL)


class TestPlanRoute(unittest.TestCase):
    def plan(self, query, crop=None, gpu=False, satellite=False):
        with mock.patch.object(daemon, "backend_up",
                               side_effect=lambda n: {"gpu": gpu, "satellite": satellite}.get(n, False)):
            return daemon.plan_route(query, crop)

    def test_gpu_awake_sends_everything_to_the_gpu_at_full_budget(self):
        for q in ("Is there a cup?", "What is on the desk?", "Read the label"):
            p = self.plan(q, gpu=True)
            self.assertEqual((p["backend"], p["tier2_px"]), ("gpu", 1024), q)

    def test_describe_goes_to_homelab_at_512(self):
        p = self.plan("What is on the desk?")
        self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", 512))

    def test_locate_needs_the_full_1024_px_for_accurate_boxes(self):
        # Measured 2026-10-08 on a mock page: click error 7-60 px at 512, 10-173 at 768, 0-6 at 1024.
        for q in ("Where is the remote?", "Locate the Sign in button"):
            for sat in (False, True):
                p = self.plan(q, satellite=sat)
                self.assertEqual(p["tier2_px"], config.GROUND_PX, q)
        self.assertEqual(config.GROUND_PX, 1024)

    def test_reading_uses_the_read_budget(self):
        p = self.plan("Read the dosage")
        self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", config.READ_PX))

    def test_crop_goes_to_homelab_at_1024(self):
        p = self.plan("Is there a cup?", crop=[100, 100, 400, 400])
        self.assertEqual((p["backend"], p["tier2_px"]), ("homelab", 1024))

    def test_satellite_awake_serves_scenes_reading_and_crops_but_not_the_gpu_shortcut(self):
        p = self.plan("What is on the desk?", satellite=True)
        self.assertEqual((p["backend"], p["tier2_px"]), ("satellite", 512))
        p = self.plan("Read the dosage", satellite=True)
        self.assertEqual((p["backend"], p["tier2_px"]), ("satellite", config.READ_PX))
        p = self.plan("Is there a cup?", crop=[1, 1, 9, 9], satellite=True)
        self.assertEqual((p["backend"], p["tier2_px"]), ("satellite", 1024))

    def test_yes_no_still_starts_on_the_phone_when_only_the_satellite_is_up(self):
        p = self.plan("Is there a cup?", satellite=True)
        self.assertEqual(p["backend"], "edge")
        self.assertEqual(p["tier2_backend"], "satellite")

    def test_gpu_beats_satellite(self):
        p = self.plan("What is on the desk?", gpu=True, satellite=True)
        self.assertEqual((p["backend"], p["tier2_px"]), ("gpu", 1024))

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

    def run_cycle(self, query, gpu=False, play_audio=False, crop=None, satellite=False):
        with mock.patch.object(daemon, "backend_up",
                               side_effect=lambda n: {"gpu": gpu, "satellite": satellite}.get(n, False)):
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
        with mock.patch.object(config, "CUE_DELAY_SEC", 0.05):
            self.run_cycle("What is on the desk?", play_audio=True)
        self.assertEqual(order, ["cue", "tier2_done", "answer"])

    def test_no_cue_when_tier2_answers_quickly(self):
        # the default Tier 2 mock returns at once: a cue would only delay the answer
        with mock.patch.object(config, "CUE_DELAY_SEC", 0.5):
            self.run_cycle("What is on the desk?", play_audio=True)
        spoken = [c.args[0] for c in self.speak.call_args_list]
        self.assertEqual(spoken, ["A desk."])

    def test_satellite_awake_labels_the_tier_and_skips_the_cue_when_it_answers_fast(self):
        res = self.run_cycle("What is on the desk?", satellite=True, play_audio=True)
        self.edge.assert_not_called()
        self.assertEqual(res["resolution_tier"], "Tier 2 (Satellite)")
        self.assertEqual(res["route"]["backend"], "satellite")
        spoken = [c.args[0] for c in self.speak.call_args_list]
        self.assertEqual(spoken, ["A desk."])              # the cue only plays if Tier 2 is slow

    def test_no_cue_when_the_gpu_answers_fast(self):
        self.run_cycle("What is on the desk?", gpu=True, play_audio=True)
        spoken = [c.args[0] for c in self.speak.call_args_list]
        self.assertEqual(len(spoken), 1)               # only the final answer

    def test_cycle_reports_where_the_time_went(self):
        res = self.run_cycle("What is on the desk?", play_audio=True)
        phases = res["phase_sec"]
        for key in ("thermal", "acquire", "prepare", "tier2", "speech", "final_temp"):
            self.assertIn(key, phases)
            self.assertGreaterEqual(phases[key], 0.0)

    def test_audio_off_never_speaks(self):
        self.run_cycle("What is on the desk?", play_audio=False)
        self.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestPromptFor(unittest.TestCase):
    def test_locate_asks_for_a_pixel_box(self):
        p = daemon.prompt_for("locate", "Where is the button?")
        self.assertTrue(p.startswith("Where is the button?"))
        self.assertIn("[x1, y1, x2, y2]", p)

    def test_reading_is_sent_as_asked(self):
        # No wording fixed the Portuguese-to-English drift (see prompt_for); only the image size did.
        self.assertEqual(daemon.prompt_for("ocr", "Leia todo o texto da imagem."), "Leia todo o texto da imagem.")

    def test_other_kinds_are_sent_unchanged(self):
        for kind in ("describe", "presence", "ocr"):
            self.assertEqual(daemon.prompt_for(kind, "What is this?"), "What is this?")

    def test_the_eval_harness_uses_the_same_wrapper(self):
        from eval import run_eval
        self.assertIs(run_eval.build_prompt, daemon.prompt_for) if hasattr(run_eval, "build_prompt") else None
        self.assertEqual(run_eval.question_prompt({"q": "Read it", "kind": "read"}), "Read it")
        self.assertEqual(run_eval.question_prompt({"q": "Where?", "kind": "locate"}), daemon.prompt_for("locate", "Where?"))
