#!/usr/bin/env python3
"""
Evaluation harness for the ambient companion (PRJ-12 Phase D).

Runs every question of a cases.yaml against each awake Tier 2 backend at several image sizes, scores the
answers with eval/scoring.py, and prints accuracy and latency per (kind, backend, size).

  python3 -m eval.run_eval --cases data/eval/synthetic/cases.yaml --backends dell,satellite,gpu --px 512,768,1024

Case file (YAML list); image paths are relative to the cases file:
  - id: desk-01
    image: desk-01.jpg
    questions:
      - {q: "Is there a laptop?", kind: presence, expect: "yes"}
      - {q: "How many cups?", kind: count, expect: 3}
      - {q: "What does the label say?", kind: read, expect_text: ["500 mg"]}
      - {q: "Where is the remote?", kind: locate, expect_click: [640, 360], tolerance: 80}
      - {q: "Describe the scene.", kind: describe, all_of: ["desk"], any_of: ["lamp", "laptop"]}
Backends are only used if their /health answers; nothing is ever woken.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import config
import daemon
import optical_ingestion as oi
from eval import scoring

KIND_TO_ROUTE = {"read": "ocr", "locate": "locate"}


def question_prompt(q: Dict[str, Any]) -> str:
    """Exactly what the companion would send for this question kind."""
    return daemon.prompt_for(KIND_TO_ROUTE.get(q.get("kind", "describe"), q.get("kind", "describe")), q["q"])


def load_cases(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def backend_urls() -> Dict[str, Optional[str]]:
    """Chat URL per backend name, only for the ones that answer /health right now."""
    found: Dict[str, Optional[str]] = {}
    daemon._probe_cache.clear()
    found["gpu"] = daemon.active_gpu_url() if daemon.backend_up("gpu") else None
    found["satellite"] = daemon.SATELLITE_LLAMA_SERVER_URL if daemon.backend_up("satellite") else None
    found["dell"] = daemon.LLAMA_SERVER_URL if daemon._health_ok(daemon.LLAMA_SERVER_URL) else None
    return found


def score_question(q: Dict[str, Any], answer: str, boxes: List[Dict[str, Any]]) -> float:
    kind = q.get("kind", "describe")
    if kind == "presence":
        return scoring.score_yes_no(answer, str(q["expect"]))
    if kind == "count":
        return scoring.score_count(answer, int(q["expect"]))
    if kind == "read":
        return scoring.score_text(answer, q.get("expect_text", []))["score"]
    if kind == "locate":
        return scoring.score_click(boxes, tuple(q["expect_click"]), int(q.get("tolerance", 80)))
    return scoring.score_keywords(answer, q.get("all_of", []), q.get("any_of", []))


def run(cases_file: Path, backends: List[str], sizes: List[int], limit: Optional[int], out: Path,
        only: Optional[str] = None) -> List[Dict[str, Any]]:
    urls = backend_urls()
    live = [b for b in backends if urls.get(b)]
    print(f"backends asked: {backends} | answering now: {live} | skipped: {[b for b in backends if b not in live]}", flush=True)
    cases = [c for c in load_cases(cases_file) if not only or only in c["id"]][: limit or None]
    scratch = out.parent / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with out.open("w") as sink:
        for case in cases:
            src = cases_file.parent / case["image"]
            for px in sizes:
                prepared, meta = oi.prepare_budgeted_image(src, px, scratch / f"{case['id']}_{px}.jpg")
                model_size = oi.qwen_input_size(meta["processed_width"], meta["processed_height"])
                for backend in live:
                    for n, q in enumerate(case["questions"]):
                        prompt = question_prompt(q)
                        t0 = time.time()
                        res = daemon.query_tier2_qwen(prepared, prompt, max_tokens=150, endpoints=[urls[backend]])
                        latency = time.time() - t0
                        answer = res.get("content", "") if res.get("status") == "success" else ""
                        boxes = oi.parse_grounding_coordinates(
                            answer, meta["original_width"], meta["original_height"], None, model_size=model_size
                        ) if q.get("kind") == "locate" else []
                        row = {"case": case["id"], "group": case.get("group", ""), "q": q["q"], "kind": q.get("kind", "describe"), "backend": backend,
                               "px": px, "score": score_question(q, answer, boxes), "latency": round(latency, 2),
                               "cold": n == 0, "tokens": res.get("prompt_tokens", 0), "answer": answer,
                               "click_error": scoring.click_error(boxes, tuple(q["expect_click"])) if q.get("kind") == "locate" else None,
                               "status": res.get("status")}
                        if res.get("status") != "success":
                            row["error"] = str(res.get("error", ""))[:300]
                        rows.append(row)
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        sink.flush()
                        print(f"  {case['id']:14} {backend:9} {px:5}px {row['kind']:9} score={row['score']:.1f} {latency:6.1f}s  {answer[:50]!r}", flush=True)
    return rows


def answered(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows where the backend answered; failed calls (server down, timeout) say nothing about the model."""
    return [r for r in rows if r.get("status", "success") == "success"]


def group_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accuracy and median latency per (photo group, backend, size), sorted by group; failed calls counted apart."""
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault((r.get("group", ""), r["backend"], r["px"]), []).append(r)
    out = []
    for (g, b, px), rs in sorted(buckets.items()):
        ok = answered(rs)
        out.append({"group": g, "backend": b, "px": px, "n": len(ok), "errors": len(rs) - len(ok),
                    "accuracy": sum(r["score"] for r in ok) / len(ok) if ok else 0.0,
                    "median_latency": round(statistics.median(r["latency"] for r in ok), 1) if ok else None})
    return out


def rescore(cases_file: Path, results: Path) -> List[Dict[str, Any]]:
    """Recompute scores (and groups) of a results JSONL from its stored answers, after a rubric or scorer fix."""
    questions = {}
    for case in load_cases(cases_file):
        for q in case["questions"]:
            questions[(case["id"], q["q"])] = (q, case.get("group", ""))
    rows = []
    for line in results.read_text().splitlines():
        r = json.loads(line)
        q, group = questions.get((r["case"], r["q"]), (None, r.get("group", "")))
        r["group"] = group
        if q is not None and r.get("status", "success") == "success":
            if q.get("kind") == "locate":
                err = r.get("click_error")
                r["score"] = 1.0 if err is not None and err <= int(q.get("tolerance", 80)) else 0.0
            else:
                r["score"] = score_question(q, r.get("answer", ""), [])
        rows.append(r)
    return rows


def print_summary(rows: List[Dict[str, Any]]) -> None:
    def table(title: str, subset: List[Dict[str, Any]]) -> None:
        print(f"\n{title}\n\n| kind | backend | px | n | accuracy | median latency |\n| --- | --- | ---: | ---: | ---: | ---: |")
        for r in scoring.summarize(subset):
            print(f"| {r['kind']} | {r['backend']} | {r['px']} | {r['n']} | {r['accuracy']:.0%} | {r['median_latency']} s |")
    failed = len(rows) - len(answered(rows))
    if failed:
        print(f"\n{failed} of {len(rows)} calls failed (backend error/timeout) and are left out of the accuracy below.")
    rows = answered(rows)
    table("All questions", rows)
    table("First question per image/size/backend only (cold image, no cache help)", [r for r in rows if r["cold"]])
    if any(r.get("group") for r in rows):
        print("\nPer photo group\n\n| group | backend | px | n | accuracy | median latency |\n| --- | --- | ---: | ---: | ---: | ---: |")
        for r in group_summary(rows):
            print(f"| {r['group']} | {r['backend']} | {r['px']} | {r['n']} | {r['accuracy']:.0%} | {r['median_latency']} s |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--backends", default="dell,satellite,gpu")
    ap.add_argument("--px", default="512,768,1024")
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ap.add_argument("--only", default=None, help="only cases whose id contains this text (e.g. text-)")
    ap.add_argument("--out", type=Path, default=None, help="results JSONL (default: next to the cases file)")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="re-score an existing results JSONL with the current cases/scorer (no model calls)")
    args = ap.parse_args()
    if args.rescore:
        rows = rescore(args.cases, args.rescore)
        out = args.rescore.with_suffix(".rescored.jsonl")
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print_summary(rows)
        print(f"\nrescored: {out}")
        return 0
    out = args.out or args.cases.parent / f"results-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rows = run(args.cases, [b.strip() for b in args.backends.split(",") if b.strip()],
               [int(p) for p in args.px.split(",")], args.limit, out, args.only)
    print_summary(rows)
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
