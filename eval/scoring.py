#!/usr/bin/env python3
"""Scoring rules for the ambient-companion evaluation harness (PRJ-12 Phase D). Pure functions."""

import re
import statistics
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize(text: str) -> str:
    """Lowercase, no accents, no punctuation, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", plain.lower())).strip()


# ---- yes / no --------------------------------------------------------------------------------------------
_NEGATIVE = re.compile(r"\b(there (is|are) no|no one|nobody|not (visible|present|any)|nao (ha|existe|tem|ve))\b")
_POSITIVE = re.compile(r"\b(there (is|are)|ha (um|uma|uns|umas)|existe|tem (um|uma))\b")


def answer_yes_no(text: str) -> Optional[str]:
    n = normalize(text)
    if not n:
        return None
    first = n.split()[0]
    if first in ("yes", "sim"):
        return "yes"
    if first in ("no", "nao"):
        return "no"
    if _NEGATIVE.search(n):
        return "no"
    if _POSITIVE.search(n):
        return "yes"
    return None


def score_yes_no(answer: str, expected: str) -> float:
    return 1.0 if answer_yes_no(answer) == expected.strip().lower() else 0.0


# ---- reading ---------------------------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate of `hypothesis` against `reference` (0.0 = identical)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis, reference) / len(reference)


def _phrase_found(answer_norm: str, phrase: str, max_cer: float = 0.2) -> bool:
    p = normalize(phrase)
    if not p:
        return True
    if p in answer_norm:
        return True
    words, n = answer_norm.split(), len(p.split())
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(words) - size + 1)):
            window = " ".join(words[i:i + size])
            if window and cer(window, p) <= max_cer:
                return True
    return False


def score_text(answer: str, expected: Sequence[str]) -> Dict[str, Any]:
    """Fraction of the expected phrases found (exactly, or within 20% character error)."""
    if not expected:
        return {"found": 0, "total": 0, "score": 1.0}
    a = normalize(answer)
    found = sum(1 for phrase in expected if _phrase_found(a, phrase))
    return {"found": found, "total": len(expected), "score": found / len(expected)}


# ---- counting --------------------------------------------------------------------------------------------
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5, "seis": 6,
    "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12,
}


def answer_count(text: str) -> Optional[int]:
    for token in normalize(text).split():
        if token.isdigit():
            return int(token)
        if token in _NUMBER_WORDS:
            return _NUMBER_WORDS[token]
    return None


def score_count(answer: str, expected: int) -> float:
    return 1.0 if answer_count(answer) == expected else 0.0


# ---- locating --------------------------------------------------------------------------------------------
def click_error(boxes: Iterable[Dict[str, Any]], truth: Tuple[int, int]) -> Optional[int]:
    """Manhattan pixel error of the first box's click target against the true centre, or None without boxes."""
    for box in boxes:
        return abs(int(box["click_x"]) - truth[0]) + abs(int(box["click_y"]) - truth[1])
    return None


def score_click(boxes: Iterable[Dict[str, Any]], truth: Tuple[int, int], tolerance: int) -> float:
    err = click_error(boxes, truth)
    return 1.0 if err is not None and err <= tolerance else 0.0


# ---- descriptions ----------------------------------------------------------------------------------------
def score_keywords(answer: str, all_of: Sequence[str] = (), any_of: Sequence[str] = ()) -> float:
    """Mean of: fraction of all_of words present, and 1/0 for at least one any_of word present."""
    a = set(normalize(answer).split())
    parts: List[float] = []
    if all_of:
        parts.append(sum(1 for w in all_of if normalize(w) in a) / len(all_of))
    if any_of:
        parts.append(1.0 if any(normalize(w) in a for w in any_of) else 0.0)
    return sum(parts) / len(parts) if parts else 1.0


# ---- summary ---------------------------------------------------------------------------------------------
def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (kind, backend, px): count, mean score, median latency."""
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["backend"], r["px"]), []).append(r)
    out = []
    for (kind, backend, px), items in sorted(groups.items()):
        out.append({
            "kind": kind, "backend": backend, "px": px, "n": len(items),
            "accuracy": round(sum(i["score"] for i in items) / len(items), 3),
            "median_latency": round(statistics.median(i["latency"] for i in items), 2),
        })
    return out
