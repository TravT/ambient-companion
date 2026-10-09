#!/usr/bin/env python3
"""
Push-to-talk capture from the phone's microphone (PRJ-12 Phase E).

On demand only: no wake word, no always-on listening. The mic is open from the trigger until the end of the
utterance (or a time cap). Audio is streamed from the S20 FE over ADB with the recipe PRJ-15 proved
(`tinymix` capture routes, then `tinycap`), judged for end of speech on the Dell with a simple energy detector, and
kept in memory only: it is never written to disk here.
"""

import math
import struct
import subprocess
import time
from typing import Iterator, Optional

import config

RATE = 16000                      # Whisper's native rate
WINDOW_SEC = 0.1
WINDOW_BYTES = int(RATE * WINDOW_SEC) * 2

CAPTURE_ROUTES = [
    ("MultiMedia1 Mixer TX_CDC_DMA_TX_3", "1"),
    ("TX_AIF1_CAP Mixer DEC0", "1"),
    ("TX DMIC MUX0", "DMIC1"),
    ("TX DEC0 MUX", "MSM_DMIC"),
    ("TX_CDC_DMA_TX_3 Channels", "One"),
]


def level_dbfs(pcm16: bytes) -> float:
    """RMS level of 16-bit little-endian mono PCM in dB relative to full scale."""
    n = len(pcm16) // 2
    if n == 0:
        return -120.0
    samples = struct.unpack(f"<{n}h", pcm16[: n * 2])
    rms = math.sqrt(sum(s * s for s in samples) / n)
    return 20 * math.log10(rms / 32768) if rms > 0 else -120.0


class EndpointDetector:
    """Energy-based start/end-of-speech detector over 100 ms windows.

    The noise floor is learned from the first 0.3 s; speech needs two consecutive windows well above it (a click
    is not speech); the utterance ends after `silence_sec` of quiet once speech was heard.
    """

    def __init__(self, max_sec: float = 12.0, no_speech_sec: float = 4.0, silence_sec: float = 1.0,
                 margin_db: float = 14.0, floor_dbfs: float = -55.0):
        self.max_sec, self.no_speech_sec, self.silence_sec = max_sec, no_speech_sec, silence_sec
        self.margin_db, self.floor_dbfs = margin_db, floor_dbfs
        self.elapsed = 0.0
        self.heard_speech = False
        self.speech_sec = 0.0
        self._calibration: list = []
        self._noise_db = floor_dbfs
        self._loud_run = 0
        self._quiet_for = 0.0

    @property
    def threshold_dbfs(self) -> float:
        return max(self.floor_dbfs, self._noise_db + self.margin_db)

    def feed(self, window: bytes) -> str:
        """Returns 'listening', 'done', 'no_speech' or 'max'."""
        level = level_dbfs(window)
        self.elapsed += WINDOW_SEC
        if len(self._calibration) < 3 and not self.heard_speech:
            self._calibration.append(level)
            self._noise_db = sum(self._calibration) / len(self._calibration)
            return "listening"
        loud = level > self.threshold_dbfs
        if loud:
            self._loud_run += 1
            self._quiet_for = 0.0
            if not self.heard_speech and self._loud_run >= 2:
                self.heard_speech = True
                self.speech_sec += self._loud_run * WINDOW_SEC       # count the windows that proved it
            elif self.heard_speech:
                self.speech_sec += WINDOW_SEC
        else:
            self._loud_run = 0
            if self.heard_speech:
                self._quiet_for += WINDOW_SEC
        if self.elapsed >= self.max_sec:
            return "max"
        if self.heard_speech and self._quiet_for >= self.silence_sec:
            return "done"
        if not self.heard_speech and self.elapsed >= self.no_speech_sec:
            return "no_speech"
        return "listening"


def to_wav(pcm16: bytes, rate: int = RATE) -> bytes:
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm16)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm16)))
    return header + pcm16


def pcm_chunks(stream, chunk_bytes: int = WINDOW_BYTES) -> Iterator[bytes]:
    """PCM from tinycap's stdout: skips the text banner line and the WAV header when present."""
    head = stream.read(64)
    if not head:
        return
    data = head
    if head[:9] == b"Capturing":
        nl = head.find(b"\n")
        if nl < 0:
            data = head + stream.read(64)
            nl = data.find(b"\n")
        data = data[nl + 1:]
    if data[:4] == b"RIFF":
        need = 44 - len(data)
        if need > 0:
            data += stream.read(need)
        data = data[44:]
    buf = data
    while True:
        while len(buf) >= chunk_bytes:
            yield buf[:chunk_bytes]
            buf = buf[chunk_bytes:]
        more = stream.read(chunk_bytes)
        if not more:
            break
        buf += more
    if buf:
        yield buf


def capture_utterance(max_sec: float = 12.0, no_speech_sec: float = 4.0, silence_sec: float = 1.0) -> dict:
    """Record one utterance from the phone mic. Returns {wav|None, reason, speech_sec, total_sec}."""
    t0 = time.time()
    routes = "; ".join(f'tinymix "{name}" {value}' for name, value in CAPTURE_ROUTES)
    subprocess.run([*config.adb_base(), "shell", f"su -c '{routes}'"], capture_output=True, timeout=20, check=False)
    cmd = (f"su -c 'tinycap /dev/stdout -D 0 -d 0 -c 1 -r {RATE} -b 16 -T {int(math.ceil(max_sec)) + 1}'")
    proc = subprocess.Popen([*config.adb_base(), "exec-out", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    det = EndpointDetector(max_sec=max_sec, no_speech_sec=no_speech_sec, silence_sec=silence_sec)
    captured = bytearray()
    reason = "eof"
    try:
        for window in pcm_chunks(proc.stdout):
            if len(window) < WINDOW_BYTES:
                break
            captured += window
            state = det.feed(window)
            if state != "listening":
                reason = state
                break
    finally:
        proc.terminate()               # closing the pipe also stops tinycap on the phone
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    wav: Optional[bytes] = to_wav(bytes(captured)) if det.heard_speech else None
    return {"wav": wav, "reason": reason, "speech_sec": round(det.speech_sec, 2),
            "total_sec": round(max(time.time() - t0, det.elapsed), 2) if reason == "eof" else round(det.elapsed, 2)}
