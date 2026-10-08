#!/usr/bin/env python3
"""Qwen2.5-VL answers in absolute pixels [x1, y1, x2, y2] of the (28-multiple) image it saw.

Regression for the bug found 2026-10-08: the parser read those numbers as normalized 0-1000
[ymin, xmin, ymax, xmax], so click targets were hundreds of pixels off. Ground truth below comes
from a mock login page whose element positions are known, run through the real model.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from PIL import Image

import optical_ingestion as oi


class TestQwenInputSize(unittest.TestCase):
    """Sizes and prompt-token counts measured on 2026-10-07 (tokens = grid + 31 template tokens)."""

    def test_measured_cases(self):
        cases = {
            (288, 384): ((392, 532), 297),    # 384 px frame: below the 256-token floor, scaled up
            (384, 512): ((392, 532), 297),    # 512 px frame: same grid, 297 measured
            (576, 768): ((588, 756), 598),    # 768 px frame
            (768, 1024): ((756, 1036), 1030),  # 1024 px frame
        }
        for (w, h), (size, tokens) in cases.items():
            got = oi.qwen_input_size(w, h, min_tokens=256)
            self.assertEqual(got, size, (w, h))
            self.assertEqual((got[0] // 28) * (got[1] // 28) + 31, tokens, (w, h))

    def test_sizes_are_multiples_of_28_and_respect_the_floor(self):
        for w, h in [(100, 100), (1280, 800), (800, 1280), (4032, 3024), (64, 900)]:
            ow, oh = oi.qwen_input_size(w, h, min_tokens=256)
            self.assertEqual((ow % 28, oh % 28), (0, 0))
            self.assertGreaterEqual((ow // 28) * (oh // 28), 256)


class TestAbsoluteGrounding(unittest.TestCase):
    # Real model output on the mock 1280x800 login page, sent as a 1024x640 image.
    SENT = (1024, 640)
    CASES = [
        ("[382, 437, 654, 478]", (640, 568)),   # blue "Sign in" button
        ("[382, 289, 654, 325]", (640, 381)),   # Username input
        ("[380, 499, 491, 514]", (535, 630)),   # "Forgot password?" link
    ]

    def parse(self, raw, crop_info=None):
        size = oi.qwen_input_size(*self.SENT, min_tokens=256)
        return oi.parse_grounding_coordinates(raw, 1280, 800, crop_info, model_size=size)

    def test_click_targets_match_the_known_elements(self):
        for raw, (tx, ty) in self.CASES:
            (m,) = self.parse(raw)
            self.assertLessEqual(abs(m["click_x"] - tx), 6, raw)
            self.assertLessEqual(abs(m["click_y"] - ty), 6, raw)

    def test_explicit_xyxy_fields_are_consistent(self):
        (m,) = self.parse(self.CASES[0][0])
        x1, y1, x2, y2 = m["box_xyxy_pixels"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertEqual((m["click_x"], m["click_y"]), ((x1 + x2) // 2, (y1 + y2) // 2))
        # existing y-first fields keep their meaning
        self.assertEqual(m["box_2d_pixels"], [y1, x1, y2, x2])

    def test_json_bbox_is_x1_y1_x2_y2_and_point_is_x_y(self):
        (m,) = self.parse('{"label": "button", "bbox_2d": [382, 437, 654, 478]}')
        self.assertLessEqual(abs(m["click_x"] - 640), 6)
        self.assertEqual(m["label"], "button")
        (p,) = self.parse('{"label": "dot", "point": [518, 458]}')
        self.assertLessEqual(abs(p["center_pixels"][1] - 640), 12)   # center_pixels = [y, x]
        self.assertLessEqual(abs(p["center_pixels"][0] - 570), 12)

    def test_out_of_range_numbers_are_clamped_not_dropped(self):
        (m,) = self.parse("[0, 0, 1036, 644]")
        self.assertEqual(m["box_xyxy_pixels"], [0, 0, 1280, 800])

    def test_prose_around_the_box_is_ignored(self):
        res = self.parse("The button is at [382, 437, 654, 478], below the form.")
        self.assertEqual(len(res), 1)

    def test_legacy_normalized_mode_is_unchanged_without_model_size(self):
        (m,) = oi.parse_grounding_coordinates("[100, 200, 300, 400]", 1000, 1000)
        self.assertEqual(m["box_2d_norm"], [100, 200, 300, 400])

    def test_crop_remap_uses_the_crop_image_the_model_saw(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "s.jpg"
            Image.new("RGB", (1920, 1080), (120, 130, 140)).save(src)
            crop = [200, 300, 600, 700]   # ymin, xmin, ymax, xmax on the 0-1000 canvas
            _, meta = oi.prepare_budgeted_image(src, 1024, Path(d) / "c.jpg", crop_bbox=crop)
        mw, mh = oi.qwen_input_size(meta["processed_width"], meta["processed_height"], min_tokens=256)
        # a box covering the middle half of the model's view of the crop
        raw = f"[{mw // 4}, {mh // 4}, {3 * mw // 4}, {3 * mh // 4}]"
        (m,) = oi.parse_grounding_coordinates(raw, meta["original_width"], meta["original_height"],
                                              meta["crop_info"], model_size=(mw, mh))
        self.assertTrue(m["remapped_from_crop"])
        # middle half of a crop spanning y 200-600 and x 300-700 -> y 300-500, x 400-600 (+-2 rounding)
        for got, want in zip(m["box_2d_norm"], [300, 400, 500, 600]):
            self.assertLessEqual(abs(got - want), 3)


class TestToolWiring(unittest.TestCase):
    """The MCP tool and the cycle must pass the model-space size to the parser."""

    def test_ocr_tool_returns_a_correct_click_target(self):
        import daemon
        import server
        with tempfile.TemporaryDirectory() as d:
            page = Path(d) / "page.png"
            Image.new("RGB", (1280, 800), (245, 247, 250)).save(page)
            # What the model would answer for the Sign-in button (true centre 640,568), computed in
            # the model's resized space for whatever size the tool actually sends.
            sent = {}

            def fake_prepare(src, max_dim, dest, crop_bbox=None):
                w, h = (max_dim, round(max_dim * 800 / 1280))
                sent["size"] = (w, h)
                return dest, {"original_width": 1280, "original_height": 800, "processed_width": w,
                              "processed_height": h, "crop_applied": False, "crop_info": None}

            def fake_t2(img, prompt, max_tokens=150):
                mw, mh = oi.qwen_input_size(*sent["size"], min_tokens=256)
                x1, y1, x2, y2 = (470 / 1280 * mw, 540 / 800 * mh, 810 / 1280 * mw, 596 / 800 * mh)
                return {"status": "success", "content": f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]",
                        "duration_sec": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

            with mock.patch.object(daemon, "BENCHMARK_DIR", Path(d)), \
                    mock.patch.object(daemon, "backend_up", return_value=False), \
                    mock.patch.object(oi, "prepare_budgeted_image", fake_prepare), \
                    mock.patch.object(daemon, "query_tier2_qwen", fake_t2):
                res = server._run_tool("ambient_ocr_and_grounding",
                                       {"query": "Where is the Sign in button?", "source": str(page)})
        text = res["content"][0]["text"]
        import json, re
        payload = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
        match = payload["primary_match"]
        self.assertLessEqual(abs(match["click_x"] - 640), 6)
        self.assertLessEqual(abs(match["click_y"] - 568), 6)


class TestGroundingRetry(unittest.TestCase):
    def run_tool(self, replies):
        import daemon
        import server
        prompts = []

        def fake_t2(img, prompt, max_tokens=150):
            prompts.append(prompt)
            return {"status": "success", "content": replies[min(len(prompts), len(replies)) - 1],
                    "duration_sec": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        with tempfile.TemporaryDirectory() as d:
            page = Path(d) / "p.png"
            Image.new("RGB", (1280, 800)).save(page)
            with mock.patch.object(daemon, "BENCHMARK_DIR", Path(d)), \
                    mock.patch.object(daemon, "backend_up", return_value=False), \
                    mock.patch.object(daemon, "query_tier2_qwen", fake_t2):
                res = server._run_tool("ambient_ocr_and_grounding",
                                       {"query": "Where is the Username field?", "source": str(page)})
        return res, prompts

    def test_prose_reply_to_a_locate_question_triggers_one_stricter_retry(self):
        res, prompts = self.run_tool(["It is below the label.", "[200, 150, 400, 190]"])
        self.assertEqual(len(prompts), 2)
        self.assertIn("exactly four numbers", prompts[1])
        self.assertIn("click_x", res["content"][0]["text"])

    def test_no_retry_when_the_first_reply_has_a_box(self):
        _, prompts = self.run_tool(["[200, 150, 400, 190]"])
        self.assertEqual(len(prompts), 1)

    def test_retry_happens_at_most_once(self):
        res, prompts = self.run_tool(["no idea", "still no idea"])
        self.assertEqual(len(prompts), 2)
        self.assertFalse(res.get("isError", False))


class TestGroundingPrompt(unittest.TestCase):
    def prompt_sent(self, query):
        import daemon
        import server
        seen = {}

        def fake_t2(img, prompt, max_tokens=150):
            seen["prompt"] = prompt
            return {"status": "success", "content": "[1, 2, 3, 4]", "duration_sec": 1.0,
                    "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        with tempfile.TemporaryDirectory() as d:
            page = Path(d) / "p.png"
            Image.new("RGB", (640, 400)).save(page)
            with mock.patch.object(daemon, "BENCHMARK_DIR", Path(d)), \
                    mock.patch.object(daemon, "backend_up", return_value=False), \
                    mock.patch.object(daemon, "query_tier2_qwen", fake_t2):
                server._run_tool("ambient_ocr_and_grounding", {"query": query, "source": str(page)})
        return seen["prompt"]

    def test_locate_questions_ask_for_a_machine_readable_box(self):
        p = self.prompt_sent("Where is the Sign in button?")
        self.assertIn("Where is the Sign in button?", p)
        self.assertIn("[x1, y1, x2, y2]", p)
        self.assertIn("pixel", p.lower())

    def test_other_questions_are_sent_unchanged(self):
        self.assertEqual(self.prompt_sent("Read the dosage on the label"), "Read the dosage on the label")


if __name__ == "__main__":
    unittest.main()
