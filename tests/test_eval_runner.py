#!/usr/bin/env python3
"""Evaluation harness plumbing: the synthetic set and the runner, with a fake backend (no models)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import daemon
from eval import make_synthetic, run_eval


class TestSyntheticSet(unittest.TestCase):
    def test_builds_images_and_a_loadable_case_file(self):
        with tempfile.TemporaryDirectory() as d:
            cases_path = make_synthetic.build(Path(d))
            cases = run_eval.load_cases(cases_path)
            self.assertGreaterEqual(len(cases), 10)
            for c in cases:
                self.assertTrue((Path(d) / c["image"]).exists(), c["id"])
                self.assertTrue(c["questions"])
            kinds = {q["kind"] for c in cases for q in c["questions"]}
            self.assertEqual(kinds, {"count", "presence", "describe", "locate", "read"})


class TestScoreQuestion(unittest.TestCase):
    def test_each_kind_uses_its_rule(self):
        s = run_eval.score_question
        self.assertEqual(s({"kind": "presence", "expect": "yes"}, "Yes.", []), 1.0)
        self.assertEqual(s({"kind": "count", "expect": 3}, "three", []), 1.0)
        self.assertEqual(s({"kind": "read", "expect_text": ["500 mg"]}, "IBUPROFEN 500 mg", []), 1.0)
        self.assertEqual(s({"kind": "locate", "expect_click": [10, 10], "tolerance": 5}, "", [{"click_x": 12, "click_y": 11}]), 1.0)
        self.assertEqual(s({"kind": "describe", "any_of": ["yellow"]}, "a yellow car", []), 1.0)


class TestRunner(unittest.TestCase):
    def test_runs_only_awake_backends_and_scores_the_fake_answers(self):
        with tempfile.TemporaryDirectory() as d:
            cases_path = make_synthetic.build(Path(d) / "set")
            out = Path(d) / "res" / "results.jsonl"
            calls = []

            def fake_query(image, prompt, max_tokens=150, endpoints=None):
                calls.append(endpoints[0])
                return {"status": "success", "content": "Yes.", "prompt_tokens": 100, "duration_sec": 0.1}

            with mock.patch.object(run_eval, "backend_urls", return_value={"gpu": None, "satellite": "http://sat/v1", "dell": None}), \
                    mock.patch.object(daemon, "query_tier2_qwen", fake_query):
                rows = run_eval.run(cases_path, ["dell", "satellite", "gpu"], [512], limit=2, out=out)
            self.assertTrue(rows)
            self.assertEqual(set(calls), {"http://sat/v1"})                  # sleeping backends are skipped
            self.assertTrue(all(r["backend"] == "satellite" for r in rows))
            self.assertTrue(out.exists())
            # the very first question of each image/size/backend is flagged cold
            self.assertTrue(rows[0]["cold"])
            self.assertTrue(all("group" in r for r in rows))

    def test_rows_carry_the_case_group(self):
        with tempfile.TemporaryDirectory() as d:
            cases_path = make_synthetic.build(Path(d) / "set")
            cases = run_eval.load_cases(cases_path)[:1]
            cases[0]["group"] = "text"
            fake = {"status": "success", "content": "Yes.", "prompt_tokens": 1}
            with mock.patch.object(run_eval, "load_cases", return_value=cases), \
                    mock.patch.object(run_eval, "backend_urls", return_value={"gpu": None, "satellite": None, "dell": "http://d/v1"}), \
                    mock.patch.object(daemon, "query_tier2_qwen", return_value=fake):
                rows = run_eval.run(cases_path, ["dell"], [512], limit=None, out=Path(d) / "r.jsonl")
            self.assertEqual({r["group"] for r in rows}, {"text"})


class TestGroupSummary(unittest.TestCase):
    def test_accuracy_and_latency_per_group_backend_and_size(self):
        rows = [
            {"group": "text", "backend": "dell", "px": 768, "score": 1.0, "latency": 10.0},
            {"group": "text", "backend": "dell", "px": 768, "score": 0.0, "latency": 30.0},
            {"group": "text", "backend": "dell", "px": 1024, "score": 1.0, "latency": 50.0},
            {"group": "absence", "backend": "dell", "px": 768, "score": 1.0, "latency": 5.0},
        ]
        got = {(r["group"], r["px"]): r for r in run_eval.group_summary(rows)}
        self.assertEqual(got[("text", 768)]["n"], 2)
        self.assertAlmostEqual(got[("text", 768)]["accuracy"], 0.5)
        self.assertEqual(got[("text", 768)]["median_latency"], 20.0)
        self.assertEqual(got[("absence", 768)]["accuracy"], 1.0)
        self.assertEqual(list(got)[0][0], "absence")                       # sorted by group


if __name__ == "__main__":
    unittest.main()
