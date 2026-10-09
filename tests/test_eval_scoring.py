#!/usr/bin/env python3
"""Scoring rules for the evaluation harness (PRJ-12 Phase D): pure functions, no models, no hardware."""

import sys
import unittest
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from eval import scoring


class TestNormalize(unittest.TestCase):
    def test_accents_case_and_punctuation_are_ignored(self):
        self.assertEqual(scoring.normalize("  Ibuprofeno, 500 mg!  "), "ibuprofeno 500 mg")
        self.assertEqual(scoring.normalize("GRATIDÃO"), "gratidao")


class TestYesNo(unittest.TestCase):
    def test_answers(self):
        cases = {"Yes.": "yes", "No, there is no keyboard present.": "no", "Sim, há um copo.": "yes",
                 "Não.": "no", "There are no people in the image.": "no", "There is a cup on the desk.": "yes",
                 "Nothing.": None, "": None}
        for text, want in cases.items():
            self.assertEqual(scoring.answer_yes_no(text), want, text)

    def test_score(self):
        self.assertEqual(scoring.score_yes_no("Yes.", "yes"), 1.0)
        self.assertEqual(scoring.score_yes_no("No.", "yes"), 0.0)
        self.assertEqual(scoring.score_yes_no("Nothing.", "yes"), 0.0)


class TestReading(unittest.TestCase):
    def test_cer(self):
        self.assertEqual(scoring.cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(scoring.cer("abd", "abc"), 1 / 3)
        self.assertEqual(scoring.cer("", ""), 0.0)

    def test_phrases_found_exactly_or_with_small_errors(self):
        answer = "The label reads: IBUPROFENO 500 mg, take 1 tablet every 8 hours."
        r = scoring.score_text(answer, ["ibuprofeno", "500 mg", "8 hours"])
        self.assertEqual((r["found"], r["total"]), (3, 3))
        r = scoring.score_text("IBUPROFENNO 500 mg", ["ibuprofeno"])           # one inserted letter
        self.assertEqual(r["found"], 1)
        r = scoring.score_text("paracetamol 750 mg", ["ibuprofeno", "500 mg"])
        self.assertEqual(r["found"], 0)

    def test_empty_expectation_scores_one(self):
        self.assertEqual(scoring.score_text("anything", [])["score"], 1.0)


class TestCount(unittest.TestCase):
    def test_numbers_and_words_in_two_languages(self):
        self.assertEqual(scoring.answer_count("There are three red cups."), 3)
        self.assertEqual(scoring.answer_count("3 cups"), 3)
        self.assertEqual(scoring.answer_count("Há cinco livros na mesa."), 5)
        self.assertEqual(scoring.answer_count("I see twelve plants"), 12)
        self.assertIsNone(scoring.answer_count("Several cups"))

    def test_score(self):
        self.assertEqual(scoring.score_count("There are 3 cups", 3), 1.0)
        self.assertEqual(scoring.score_count("There are 4 cups", 3), 0.0)


class TestLocate(unittest.TestCase):
    def test_click_error_and_tolerance(self):
        boxes = [{"click_x": 640, "click_y": 570}]
        self.assertEqual(scoring.click_error(boxes, (640, 568)), 2)
        self.assertEqual(scoring.score_click(boxes, (640, 568), tolerance=40), 1.0)
        self.assertEqual(scoring.score_click([{"click_x": 100, "click_y": 100}], (640, 568), tolerance=40), 0.0)

    def test_no_box_scores_zero_and_error_none(self):
        self.assertIsNone(scoring.click_error([], (1, 1)))
        self.assertEqual(scoring.score_click([], (1, 1), tolerance=40), 0.0)


class TestKeywords(unittest.TestCase):
    def test_any_and_all_of(self):
        a = "A yellow car is parked at night near a building."
        self.assertEqual(scoring.score_keywords(a, all_of=["yellow", "car"], any_of=["night", "dark"]), 1.0)
        self.assertEqual(scoring.score_keywords(a, all_of=["yellow", "truck"]), 0.5)
        self.assertEqual(scoring.score_keywords(a, any_of=["boat", "plane"]), 0.0)

    def test_nothing_expected_is_a_pass(self):
        self.assertEqual(scoring.score_keywords("x"), 1.0)


class TestSummary(unittest.TestCase):
    def test_groups_by_kind_backend_and_size(self):
        rows = [
            {"kind": "presence", "backend": "dell", "px": 512, "score": 1.0, "latency": 20.0},
            {"kind": "presence", "backend": "dell", "px": 512, "score": 0.0, "latency": 22.0},
            {"kind": "read", "backend": "dell", "px": 768, "score": 1.0, "latency": 55.0},
        ]
        table = scoring.summarize(rows)
        row = next(r for r in table if r["kind"] == "presence")
        self.assertEqual((row["n"], row["accuracy"], row["median_latency"]), (2, 0.5, 21.0))
        self.assertEqual(len(table), 2)


if __name__ == "__main__":
    unittest.main()
