#!/usr/bin/env python3
"""
ambient-speech: text-to-speech and speech-to-text for the companion, on the Dell (PRJ-12 Phases E and F).

  GET  /health   liveness, plus which models are loaded right now
  POST /tts      JSON {"text": ..., "language": "en"|"pt"}  ->  audio/wav (pocket-tts, built-in default voices)
  POST /stt      body = 16-bit WAV (any rate/channels, resampled to 16 kHz mono); ?language=en|pt|auto -> JSON
  POST /warm     JSON {"model": "tts-en"|"tts-pt"|"stt"}: load it in the background now (202), so it is ready when used

Loopback only; bearer token when SPEECH_TOKEN is set. Memory budget: the Dell has about 1.5-2.5 GB free and each
voice costs about 1 GB, so nothing is resident by default (SPEECH_RESIDENT): models load on first use or on /warm
(the companion warms the voice when a question starts, while the image is analysed, and Whisper while the user
speaks) and are dropped after SPEECH_IDLE_UNLOAD_SEC without use. Models come from read-only mounts and are never
downloaded here.
"""

import io
import json
import os
import signal
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

VERSION = "1.0.1"
LANGUAGES = {"en": "english", "pt": "portuguese"}
MAX_TEXT_CHARS = int(os.getenv("SPEECH_MAX_TEXT_CHARS", "600"))
MAX_AUDIO_BYTES = int(os.getenv("SPEECH_MAX_AUDIO_BYTES", str(16000 * 2 * 30 + 44)))   # 30 s of 16 kHz 16-bit mono
IDLE_UNLOAD_SEC = float(os.getenv("SPEECH_IDLE_UNLOAD_SEC", "600"))
RESIDENT = {x.strip() for x in os.getenv("SPEECH_RESIDENT", "").split(",") if x.strip()}
WHISPER_DIR = os.getenv("SPEECH_WHISPER_DIR", "/models/faster-whisper-small")
THREADS = int(os.getenv("SPEECH_THREADS", "4"))


def log(msg: str) -> None:
    print(f"[ambient-speech] {msg}", flush=True)


class LazyModel:
    """A model loaded on first use; non-resident ones are dropped after `idle_sec` without use."""

    def __init__(self, name: str, loader: Callable[[], Any], resident: bool = False,
                 idle_sec: float = IDLE_UNLOAD_SEC, clock: Callable[[], float] = time.monotonic):
        self.name, self.loader, self.resident, self.idle_sec, self.clock = name, loader, resident, idle_sec, clock
        self.lock = threading.Lock()                 # one inference at a time per model
        self.obj: Any = None
        self.last_used = 0.0

    @property
    def loaded(self) -> bool:
        return self.obj is not None

    def use(self, fn: Callable[[Any], Any]) -> Any:
        with self.lock:
            if self.obj is None:
                t0 = time.time()
                self.obj = self.loader()
                log(f"loaded {self.name} in {time.time() - t0:.1f} s")
            try:
                return fn(self.obj)
            finally:
                self.last_used = self.clock()

    def unload_if_idle(self) -> bool:
        if self.resident or self.obj is None or self.clock() - self.last_used < self.idle_sec:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            self.obj = None
            log(f"unloaded {self.name} after {self.idle_sec:.0f} s idle")
            return True
        finally:
            self.lock.release()


# ---------------------------------------------------------------- real engines (imported lazily: heavy deps)

class _KeepOpen(io.BytesIO):
    """pocket-tts closes the file it writes to; keep the buffer readable afterwards."""

    def close(self) -> None:  # noqa: D401
        pass


def load_pocket_tts(language: str) -> Callable[[str], bytes]:
    from pocket_tts import TTSModel
    from pocket_tts.data.audio import stream_audio_chunks
    from pocket_tts.default_parameters import get_default_voice_for_language

    model_lang = LANGUAGES[language]
    model = TTSModel.load_model(language=model_lang, quantize=os.getenv("SPEECH_TTS_QUANTIZE", "1") == "1")
    state = model.get_state_for_audio_prompt(get_default_voice_for_language(model_lang, None, None))
    rate = model.config.mimi.sample_rate

    def synth(text: str) -> bytes:
        buf = _KeepOpen()
        stream_audio_chunks(buf, model.generate_audio_stream(model_state=state, text_to_generate=text), rate)
        return buf.getvalue()

    return synth


def wav_to_float32(data: bytes):
    """16-bit PCM WAV -> 16 kHz mono float32 for Whisper (pocket-tts writes 24 kHz, browsers often 48 kHz stereo)."""
    import numpy as np
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                raise ValueError("expected 16-bit PCM WAV")
            channels, rate = w.getnchannels(), w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype("float32") / 32768.0
    except (wave.Error, EOFError) as err:
        raise ValueError(f"not a WAV file: {err}") from err
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != 16000 and len(pcm):
        n_out = int(round(len(pcm) * 16000 / rate))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm).astype("float32")
    return pcm


def load_whisper() -> Callable[[bytes, str], Dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=THREADS)

    def transcribe(wav: bytes, language: str) -> Dict[str, Any]:
        segments, info = model.transcribe(wav_to_float32(wav), language=None if language == "auto" else language,
                                          beam_size=1, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": info.language,
                "language_probability": round(float(info.language_probability), 2),
                "audio_sec": round(float(info.duration), 2)}

    return transcribe


def default_models() -> Dict[str, LazyModel]:
    return {
        "tts-en": LazyModel("tts-en", lambda: load_pocket_tts("en"), resident="tts-en" in RESIDENT),
        "tts-pt": LazyModel("tts-pt", lambda: load_pocket_tts("pt"), resident="tts-pt" in RESIDENT),
        "stt": LazyModel("stt", load_whisper, resident="stt" in RESIDENT),
    }


# ---------------------------------------------------------------- HTTP

def make_handler(models: Dict[str, LazyModel], token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"ambient-speech/{VERSION}"

        def log_message(self, fmt, *args):  # keep container logs short
            pass

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return not token or self.headers.get("Authorization", "") == f"Bearer {token}"

        def _body(self, limit: int) -> Optional[bytes]:
            n = int(self.headers.get("Content-Length") or 0)
            if 0 < n <= limit:
                return self.rfile.read(n)
            if limit < n <= 8 * limit:              # drain it, so the client reads our answer instead of a reset
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 65536))
                    if not chunk:
                        break
                    left -= len(chunk)
            self.close_connection = True
            self._json(413 if n > limit else 400, {"error": f"body must be 1..{limit} bytes"})
            return None

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                return self._json(200, {"status": "ok", "version": VERSION,
                                        "loaded": sorted(k for k, m in models.items() if m.loaded)})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            if url.path not in ("/tts", "/stt", "/warm"):
                return self._json(404, {"error": "not found"})
            if not self._authorized():
                return self._json(401, {"error": "unauthorized"})
            try:
                if url.path == "/tts":
                    return self._tts()
                if url.path == "/warm":
                    return self._warm()
                return self._stt(parse_qs(url.query).get("language", ["auto"])[0])
            except Exception as err:  # never leak a traceback to the client
                log(f"{url.path} failed: {err!r}")
                return self._json(500, {"error": f"{type(err).__name__}: {err}"[:300]})

        def _tts(self):
            raw = self._body(MAX_TEXT_CHARS * 4 + 200)
            if raw is None:
                return
            try:
                req = json.loads(raw)
            except ValueError:
                return self._json(400, {"error": "JSON body expected"})
            text = str(req.get("text", "")).strip()
            lang = str(req.get("language", "en"))
            if lang not in LANGUAGES:
                return self._json(400, {"error": f"language must be one of {sorted(LANGUAGES)}"})
            if not text or len(text) > MAX_TEXT_CHARS:
                return self._json(400, {"error": f"text must be 1..{MAX_TEXT_CHARS} characters"})
            t0 = time.time()
            wav = models[f"tts-{lang}"].use(lambda synth: synth(text))
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("X-Synth-Sec", f"{time.time() - t0:.2f}")
            self.end_headers()
            self.wfile.write(wav)

        def _warm(self):
            raw = self._body(200)
            if raw is None:
                return
            try:
                name = str(json.loads(raw).get("model", ""))
            except (ValueError, AttributeError):
                return self._json(400, {"error": "JSON body expected"})
            if name not in models:
                return self._json(400, {"error": f"model must be one of {sorted(models)}"})
            m = models[name]
            if m.loaded:
                m.last_used = m.clock()                       # keep it a little longer
                return self._json(200, {"model": name, "state": "loaded"})
            threading.Thread(target=m.use, args=(lambda _obj: None,), daemon=True).start()
            self._json(202, {"model": name, "state": "loading"})

        def _stt(self, language: str):
            if language not in ("auto", *LANGUAGES):
                return self._json(400, {"error": "language must be auto, en or pt"})
            wav = self._body(MAX_AUDIO_BYTES)
            if wav is None:
                return
            t0 = time.time()
            result = models["stt"].use(lambda transcribe: transcribe(wav, language))
            result["stt_sec"] = round(time.time() - t0, 2)
            self._json(200, result)

    return Handler


def make_server(host: str, port: int, token: str = "", models: Optional[Dict[str, LazyModel]] = None):
    models = models if models is not None else default_models()
    httpd = ThreadingHTTPServer((host, port), make_handler(models, token))
    httpd.models = models
    return httpd


def _reaper(models: Dict[str, LazyModel], stop: threading.Event) -> None:
    while not stop.wait(30):
        for m in models.values():
            m.unload_if_idle()


def main() -> None:
    host = os.getenv("SPEECH_HOST", "127.0.0.1")
    port = int(os.getenv("SPEECH_PORT", "8091"))
    httpd = make_server(host, port, os.getenv("SPEECH_TOKEN", ""))
    stop = threading.Event()
    threading.Thread(target=_reaper, args=(httpd.models, stop), daemon=True).start()
    for name, m in httpd.models.items():          # resident models load before /health turns green
        if m.resident:
            m.use(lambda _obj: None)

    def _stop(signum, _frame):
        stop.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log(f"v{VERSION} on {host}:{port}, resident={sorted(RESIDENT)}, idle unload {IDLE_UNLOAD_SEC:.0f} s")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
