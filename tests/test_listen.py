#!/usr/bin/env python3
"""Push-to-talk capture (PRJ-12 Phase E): end-of-speech detection and the tinycap stream parser. No phone needed."""

import io
import math
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import listen

RATE = 16000
WIN = RATE // 10                      # 100 ms windows


def pcm(seconds, amp=0, freq=440):
    n = int(RATE * seconds)
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / RATE))) for i in range(n))


def windows(data):
    return [data[i:i + WIN * 2] for i in range(0, len(data) - WIN * 2 + 1, WIN * 2)]


class TestLevel(unittest.TestCase):
    def test_silence_is_very_quiet_and_a_tone_is_loud(self):
        self.assertLess(listen.level_dbfs(pcm(0.1, 0)), -90)
        self.assertGreater(listen.level_dbfs(pcm(0.1, 8000)), -20)


class TestEndpointDetector(unittest.TestCase):
    def run_detector(self, audio, **kw):
        det = listen.EndpointDetector(**kw)
        for i, w in enumerate(windows(audio)):
            state = det.feed(w)
            if state != "listening":
                return state, i + 1, det
        return "listening", len(windows(audio)), det

    def test_speech_then_silence_ends_the_utterance(self):
        audio = pcm(0.5, 40) + pcm(1.0, 6000) + pcm(2.0, 40)           # room noise, speech, room noise
        state, n, det = self.run_detector(audio, silence_sec=0.8)
        self.assertEqual(state, "done")
        self.assertTrue(det.heard_speech)
        self.assertLess(n * 0.1, 0.5 + 1.0 + 0.8 + 0.3)                  # stops about 0.8 s after the speech ends

    def test_nothing_but_room_noise_gives_up(self):
        state, n, det = self.run_detector(pcm(6.0, 40), no_speech_sec=3.0)
        self.assertEqual(state, "no_speech")
        self.assertFalse(det.heard_speech)
        self.assertLessEqual(n * 0.1, 3.3)

    def test_long_speech_is_capped(self):
        state, n, _ = self.run_detector(pcm(0.3, 40) + pcm(20.0, 6000), max_sec=5.0)
        self.assertEqual(state, "max")
        self.assertLessEqual(n * 0.1, 5.2)

    def test_a_click_is_not_speech(self):
        audio = pcm(0.5, 40) + pcm(0.1, 9000) + pcm(4.0, 40)             # a single loud 100 ms window
        state, _, det = self.run_detector(audio, no_speech_sec=3.0)
        self.assertEqual(state, "no_speech")

    def test_a_pause_inside_a_sentence_does_not_end_it(self):
        audio = pcm(0.4, 40) + pcm(0.8, 6000) + pcm(0.5, 40) + pcm(0.8, 6000) + pcm(2.0, 40)
        state, n, det = self.run_detector(audio, silence_sec=0.9)
        self.assertEqual(state, "done")
        self.assertGreater(n * 0.1, 0.4 + 0.8 + 0.5 + 0.8)               # waited through the 0.5 s pause


class TestStreamParser(unittest.TestCase):
    def header(self, n_bytes):
        return (b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
                + b"data" + struct.pack("<I", n_bytes))

    def test_banner_and_wav_header_are_skipped(self):
        body = pcm(0.3, 5000)
        stream = io.BytesIO(b"Capturing sample: 1 ch, 16000 hz, 16 bit\n" + self.header(len(body)) + body)
        got = b"".join(listen.pcm_chunks(stream, chunk_bytes=WIN * 2))
        self.assertEqual(got, body)

    def test_headerless_stream_passes_through(self):
        body = pcm(0.3, 5000)
        got = b"".join(listen.pcm_chunks(io.BytesIO(body), chunk_bytes=WIN * 2))
        self.assertEqual(got, body)


class TestWav(unittest.TestCase):
    def test_wav_wrapper_is_a_valid_header(self):
        data = pcm(0.2, 3000)
        wav = listen.to_wav(data)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(struct.unpack("<I", wav[40:44])[0], len(data))
        self.assertEqual(wav[44:], data)


class TestCapture(unittest.TestCase):
    def test_capture_stops_at_end_of_speech_and_never_writes_audio_to_disk(self):
        audio = pcm(0.5, 40) + pcm(1.0, 6000) + pcm(3.0, 40)
        stream = io.BytesIO(b"Capturing sample: 1 ch, 16000 hz, 16 bit\n" + TestStreamParser().header(len(audio)) + audio)

        class FakeProc:
            def __init__(self):
                self.stdout = stream
                self.stderr = io.BytesIO()
                self.terminated = False

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                return 0

        fake = FakeProc()
        with mock.patch.object(listen.subprocess, "run", return_value=mock.Mock(returncode=0)), \
                mock.patch.object(listen.subprocess, "Popen", return_value=fake), \
                mock.patch("builtins.open", side_effect=AssertionError("audio must stay in memory")):
            res = listen.capture_utterance(max_sec=10, no_speech_sec=4, silence_sec=0.8)
        self.assertEqual(res["reason"], "done")
        self.assertTrue(fake.terminated)
        self.assertEqual(res["wav"][:4], b"RIFF")
        self.assertGreater(res["speech_sec"], 0.9)
        self.assertLess(res["total_sec"], 3.5)

    def test_no_speech_returns_an_empty_result(self):
        audio = pcm(6.0, 40)
        stream = io.BytesIO(b"Capturing sample: 1 ch, 16000 hz, 16 bit\n" + audio)
        proc = mock.Mock(stdout=stream, stderr=io.BytesIO())
        proc.wait.return_value = 0
        with mock.patch.object(listen.subprocess, "run", return_value=mock.Mock(returncode=0)), \
                mock.patch.object(listen.subprocess, "Popen", return_value=proc):
            res = listen.capture_utterance(max_sec=10, no_speech_sec=2.0)
        self.assertEqual(res["reason"], "no_speech")
        self.assertIsNone(res["wav"])


if __name__ == "__main__":
    unittest.main()
