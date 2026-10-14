#!/usr/bin/env python3
"""STT check: word error rate and the round trip through the speech service (fake service, no models)."""

import sys
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import daemon
from eval import stt_eval


class TestWer(unittest.TestCase):
    def test_identical_after_normalization(self):
        self.assertEqual(stt_eval.wer("Is there a cup on the desk?", "is there a cup on the desk"), 0.0)

    def test_accents_and_punctuation_are_ignored(self):
        self.assertEqual(stt_eval.wer("Tem um copo na mesa?", "tem um copo na mesa"), 0.0)
        self.assertEqual(stt_eval.wer("O que está escrito?", "o que esta escrito"), 0.0)

    def test_substitution_insertion_deletion(self):
        self.assertAlmostEqual(stt_eval.wer("read the label", "read a label"), 1 / 3)
        self.assertAlmostEqual(stt_eval.wer("read the label", "read the label now"), 1 / 3)
        self.assertAlmostEqual(stt_eval.wer("read the label", "read label"), 1 / 3)

    def test_numbers_as_digits_or_words_count_the_same(self):
        self.assertEqual(stt_eval.wer("How many cups are there, three?", "how many cups are there 3"), 0.0)


class TestPhrases(unittest.TestCase):
    def test_twenty_phrases_half_portuguese(self):
        langs = [p["language"] for p in stt_eval.PHRASES]
        self.assertEqual(len(langs), 20)
        self.assertEqual(langs.count("pt"), 10)


class TestRun(unittest.TestCase):
    def test_direct_round_trip_scores_each_phrase(self):
        with mock.patch.object(daemon, "synthesize_on_dell", return_value=b"RIFF"), \
                mock.patch.object(daemon, "transcribe_on_dell",
                                  side_effect=lambda wav, lang: {"status": "success", "text": "wrong words", "language": lang,
                                                                 "stt_sec": 0.5}):
            rows = stt_eval.run(stt_eval.PHRASES[:2], mode="direct", language_hint="given")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["wer"] > 0 for r in rows))
        self.assertTrue(all("stt_sec" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
