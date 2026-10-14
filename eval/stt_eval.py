#!/usr/bin/env python3
"""
Speech-to-text check for push-to-talk (PRJ-12 Phase E), with the default pocket-tts voices instead of recordings.

  direct    : synthesize each phrase on the Dell (ambient-speech /tts), transcribe the WAV (/stt). Tests Whisper.
  acoustic  : play each phrase on the S20 FE speaker and record it with the S20 FE mic (listen.py), then /stt.
              Tests the real capture path; it is audible, so run it when sound is fine.

  python3 -m eval.stt_eval [--mode direct|acoustic] [--language given|auto] [--out results.jsonl]

Prints word error rate (lowercase, no accents or punctuation, digits as words) and STT latency per language.
"""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

import daemon  # noqa: E402

PHRASES: List[Dict[str, str]] = [
    {"language": "en", "text": "Is there a cup on the desk?"},
    {"language": "en", "text": "What does the label on the bottle say?"},
    {"language": "en", "text": "How many books are on the shelf?"},
    {"language": "en", "text": "Where is the remote control?"},
    {"language": "en", "text": "Is the window open or closed?"},
    {"language": "en", "text": "Read the text on the screen."},
    {"language": "en", "text": "What color is the chair next to the table?"},
    {"language": "en", "text": "Did I leave my keys here?"},
    {"language": "en", "text": "Is the light in the kitchen on?"},
    {"language": "en", "text": "What time does the clock show?"},
    {"language": "pt", "text": "Tem um copo em cima da mesa?"},
    {"language": "pt", "text": "O que está escrito no rótulo do remédio?"},
    {"language": "pt", "text": "Quantos livros tem na estante?"},
    {"language": "pt", "text": "Onde está o controle remoto?"},
    {"language": "pt", "text": "A janela está aberta ou fechada?"},
    {"language": "pt", "text": "Leia o texto da tela, por favor."},
    {"language": "pt", "text": "Qual é a cor da cadeira ao lado da mesa?"},
    {"language": "pt", "text": "Eu esqueci minhas chaves aqui?"},
    {"language": "pt", "text": "A luz da cozinha está acesa?"},
    {"language": "pt", "text": "Que horas o relógio está mostrando?"},
]

_NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
            "8": "eight", "9": "nine", "10": "ten"}


def normalize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", text)
    return [_NUMBERS.get(w, w) for w in words]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + insertions + deletions) / reference words."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def _acoustic_capture(wav: bytes) -> bytes:
    """Play `wav` on the phone while its mic records; returns the recorded WAV (b'' when nothing was heard)."""
    import threading

    import listen
    holder: Dict[str, Any] = {}
    rec = threading.Thread(target=lambda: holder.update(listen.capture_utterance(max_sec=10.0, silence_sec=1.2)))
    rec.start()
    time.sleep(0.8)                                    # let tinycap open before the phrase starts
    remote = daemon._push_wav_to_edge(wav, f"stt_eval_{int(time.time() * 1000)}.wav")
    if remote:
        daemon.ensure_edge_audio()
        daemon._play_wav_on_edge(remote)
        daemon._remove_on_edge(remote)
    rec.join(timeout=20)
    return holder.get("wav") or b""


def run(phrases: List[Dict[str, str]], mode: str = "direct", language_hint: str = "given") -> List[Dict[str, Any]]:
    rows = []
    for p in phrases:
        wav = daemon.synthesize_on_dell(p["text"], p["language"])
        if not wav:
            rows.append({**p, "error": "tts failed", "wer": 1.0})
            continue
        audio = wav if mode == "direct" else _acoustic_capture(wav)
        if not audio:
            rows.append({**p, "error": "nothing recorded", "wer": 1.0})
            continue
        stt = daemon.transcribe_on_dell(audio, p["language"] if language_hint == "given" else "auto")
        heard = stt.get("text", "") if stt.get("status") == "success" else ""
        row = {**p, "heard": heard, "wer": round(wer(p["text"], heard), 3), "stt_sec": stt.get("stt_sec"),
               "detected_language": stt.get("language"), "mode": mode}
        if stt.get("status") != "success":
            row["error"] = stt.get("error")
        rows.append(row)
        print(f"  {p['language']} wer={row['wer']:.2f} {str(row.get('stt_sec')):>5}s  {p['text']!r} -> {heard!r}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["direct", "acoustic"], default="direct")
    ap.add_argument("--language", choices=["given", "auto"], default="given",
                    help="pass the phrase language to Whisper, or let it detect (tests language detection)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = run(PHRASES, args.mode, args.language)
    for lang in ("en", "pt"):
        sub = [r for r in rows if r["language"] == lang]
        lat = [r["stt_sec"] for r in sub if r.get("stt_sec") is not None]
        exact = sum(1 for r in sub if r["wer"] == 0)
        print(f"{lang}: mean WER {statistics.mean(r['wer'] for r in sub):.1%}, exact {exact}/{len(sub)}, "
              f"median STT {statistics.median(lat) if lat else '-'} s"
              + (f", language detected right {sum(1 for r in sub if r.get('detected_language') == lang)}/{len(sub)}"
                 if args.language == "auto" else ""))
    if args.out:
        args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
