#!/usr/bin/env python3
"""ambient-speech HTTP surface and model lifecycle, with fake engines (no torch, no models)."""

import io
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR / "speech"))

import speech_service as ss

TOKEN = "speech-test-token"


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def fake_models(clock, loads):
    def tts_loader(lang):
        def load():
            loads.append(f"tts-{lang}")
            return lambda text: b"RIFF" + f"{lang}:{text}".encode()
        return load

    def stt_load():
        loads.append("stt")
        return lambda wav, language: {"text": "is there a cup", "language": "en" if language == "auto" else language,
                                      "language_probability": 0.9, "audio_sec": 0.5}

    return {
        "tts-en": ss.LazyModel("tts-en", tts_loader("en"), resident=True, idle_sec=60, clock=clock),
        "tts-pt": ss.LazyModel("tts-pt", tts_loader("pt"), idle_sec=60, clock=clock),
        "stt": ss.LazyModel("stt", stt_load, idle_sec=60, clock=clock),
    }


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.loads = []
        self.httpd = ss.make_server("127.0.0.1", 0, token=TOKEN, models=fake_models(self.clock, self.loads))
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def call(self, path, body=None, token=TOKEN, ctype="application/json"):
        headers = {"Content-Type": ctype}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.base + path, data=body, headers=headers,
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()


class TestHttp(ServiceCase):
    def test_health_lists_loaded_models_without_auth(self):
        status, _, body = self.call("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["loaded"], [])

    def test_tts_returns_wav_for_each_language(self):
        for lang in ("en", "pt"):
            status, headers, body = self.call("/tts", json.dumps({"text": "olá", "language": lang}).encode())
            self.assertEqual(status, 200, body)
            self.assertEqual(headers["Content-Type"], "audio/wav")
            self.assertEqual(body, b"RIFF" + f"{lang}:olá".encode())

    def test_tts_rejects_bad_input(self):
        self.assertEqual(self.call("/tts", json.dumps({"text": "hi", "language": "fr"}).encode())[0], 400)
        self.assertEqual(self.call("/tts", json.dumps({"text": "  "}).encode())[0], 400)
        self.assertEqual(self.call("/tts", json.dumps({"text": "x" * (ss.MAX_TEXT_CHARS + 1)}).encode())[0], 400)
        self.assertEqual(self.call("/tts", b"not json")[0], 400)

    def test_token_is_required(self):
        body = json.dumps({"text": "hi"}).encode()
        self.assertEqual(self.call("/tts", body, token=None)[0], 401)
        self.assertEqual(self.call("/tts", body, token="nope")[0], 401)
        self.assertEqual(self.call("/stt?language=en", wav_bytes(), token=None, ctype="audio/wav")[0], 401)

    def test_stt_transcribes_wav(self):
        status, _, body = self.call("/stt?language=pt", wav_bytes(), ctype="audio/wav")
        self.assertEqual(status, 200, body)
        res = json.loads(body)
        self.assertEqual(res["text"], "is there a cup")
        self.assertEqual(res["language"], "pt")
        self.assertIn("stt_sec", res)

    def test_stt_rejects_bad_language_and_oversized_audio(self):
        self.assertEqual(self.call("/stt?language=fr", wav_bytes(), ctype="audio/wav")[0], 400)
        big = b"\x00" * (ss.MAX_AUDIO_BYTES + 1)
        self.assertEqual(self.call("/stt", big, ctype="audio/wav")[0], 413)

    def test_engine_errors_become_500_without_traceback(self):
        def boom():
            raise RuntimeError("model file missing")
        self.httpd.models["stt"] = ss.LazyModel("stt", boom)
        status, _, body = self.call("/stt", wav_bytes(), ctype="audio/wav")
        self.assertEqual(status, 500)
        self.assertNotIn("Traceback", body.decode())
        self.assertIn("model file missing", body.decode())

    def test_unknown_path_is_404(self):
        self.assertEqual(self.call("/nope", b"{}")[0], 404)


class TestLifecycle(ServiceCase):
    def test_models_load_on_first_use_only(self):
        self.call("/tts", json.dumps({"text": "a", "language": "pt"}).encode())
        self.call("/tts", json.dumps({"text": "b", "language": "pt"}).encode())
        self.assertEqual(self.loads, ["tts-pt"])
        self.assertEqual(json.loads(self.call("/health")[2])["loaded"], ["tts-pt"])

    def test_idle_non_resident_models_are_dropped_and_reload_later(self):
        self.call("/tts", json.dumps({"text": "a", "language": "en"}).encode())
        self.call("/tts", json.dumps({"text": "a", "language": "pt"}).encode())
        self.call("/stt", wav_bytes(), ctype="audio/wav")
        self.clock.t += 30
        self.assertFalse(any(m.unload_if_idle() for m in self.httpd.models.values()))   # not idle long enough
        self.clock.t += 31
        dropped = [n for n, m in self.httpd.models.items() if m.unload_if_idle()]
        self.assertEqual(sorted(dropped), ["stt", "tts-pt"])                             # English stays resident
        self.assertEqual(json.loads(self.call("/health")[2])["loaded"], ["tts-en"])
        self.call("/tts", json.dumps({"text": "c", "language": "pt"}).encode())
        self.assertEqual(self.loads.count("tts-pt"), 2)


class TestWav(unittest.TestCase):
    def test_wav_to_float32_resamples_and_downmixes_to_16k_mono(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        self.assertEqual(len(ss.wav_to_float32(wav_bytes(0.25))), 4000)
        self.assertEqual(len(ss.wav_to_float32(wav_bytes(0.25, rate=24000))), 4000)    # pocket-tts output
        self.assertEqual(len(ss.wav_to_float32(wav_bytes(0.25, rate=48000))), 4000)    # browser recording
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x10\x00\x30" * 1600)                               # L=4096, R=12288
        mono = ss.wav_to_float32(buf.getvalue())
        self.assertEqual(len(mono), 1600)
        self.assertAlmostEqual(float(mono[0]), (4096 + 12288) / 2 / 32768, places=4)
        with self.assertRaises(ValueError):
            ss.wav_to_float32(b"not a wav")


if __name__ == "__main__":
    unittest.main()


class TestWarm(ServiceCase):
    def test_warm_loads_in_the_background_and_answers_202(self):
        import time
        status, _, body = self.call("/warm", json.dumps({"model": "stt"}).encode())
        self.assertEqual(status, 202, body)
        for _ in range(50):
            if "stt" in self.loads:
                break
            time.sleep(0.02)
        self.assertEqual(self.loads, ["stt"])

    def test_warm_on_a_loaded_model_is_a_no_op(self):
        self.call("/tts", json.dumps({"text": "a", "language": "en"}).encode())
        status, _, body = self.call("/warm", json.dumps({"model": "tts-en"}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["state"], "loaded")
        self.assertEqual(self.loads, ["tts-en"])

    def test_warm_rejects_unknown_models_and_needs_the_token(self):
        self.assertEqual(self.call("/warm", json.dumps({"model": "gpt"}).encode())[0], 400)
        self.assertEqual(self.call("/warm", json.dumps({"model": "stt"}).encode(), token=None)[0], 401)


class TestWorkerProcess(unittest.TestCase):
    """Models run in a child process, so unloading really returns the memory (gc inside one process does not)."""

    def test_calls_go_to_a_child_process_that_is_killed_when_idle(self):
        import os
        clock = FakeClock()
        m = ss.WorkerModel("echo", ss._selftest_factory, idle_sec=60, clock=clock)
        self.assertFalse(m.loaded)
        pid, args = m.call("hi", 2)
        self.assertNotEqual(pid, os.getpid())                     # ran in a child
        self.assertEqual(args, ["hi", 2])
        self.assertTrue(m.loaded)
        clock.t += 61
        self.assertTrue(m.unload_if_idle())
        self.assertFalse(m.loaded)                                 # child gone, memory returned
        pid2, _ = m.call("again")
        self.assertNotEqual(pid2, pid)                             # reloaded in a fresh child
        m.close()

    def test_errors_in_the_child_become_exceptions(self):
        m = ss.WorkerModel("echo", ss._selftest_factory)
        with self.assertRaises(RuntimeError) as ctx:
            m.call("boom")
        self.assertIn("selftest failure", str(ctx.exception))
        m.close()

    def test_a_crashed_child_is_restarted_on_the_next_call(self):
        m = ss.WorkerModel("echo", ss._selftest_factory)
        m.call("x")
        m.proc.kill()
        m.proc.join(5)
        self.assertEqual(m.call("y")[1], ["y"])
        m.close()
