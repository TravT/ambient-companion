#!/usr/bin/env python3
"""
Ambient Multimodal Desktop Companion & Escalation Daemon (PRJ-12 / ADR-40 / ADR-42)
Orchestrates Tier 1 Edge (Galaxy S20 FE) and Tier 2 Homelab Core (Dell Latitude 7390)
with Pluggable Optical Ingestion (Termux Camera, DroidCamX, RTSP, Local Files).

Two-Pass Verification & Escalation Protocol (2P-VEP):
  [Optical Ingestion (Camera/RTSP/Dropzone)]
                  │
                  ▼
  [Pass 0: Intent Filter]  ──(Dense OCR / Fine Reading)──► [Tier 2 Homelab llama-server]
                  │                                                    │
              (General)                                                │
                  ▼                                                    │
  [Pass 1: Edge SmolVLM-256M Triage]                                 │
                  │                                                    │
                  ▼                                                    │
  [Pass 2: Edge Self-Critique]                                       │
                  │                                                    │
           (Confident?) ──Yes──► [Resolved at Edge]                    │
                  │                     │                              │
                 No                     │                              │
                  ▼                     │                              │
        [Spoken Escalation Cue]         │                              │
                  │                     │                              │
                  ▼                     │                              │
        [Tier 2 Escalation] ────────────┼──────────────────────────────┘
                                        │
                                        ▼
                           [Pocket-TTS Voice Synthesis]
                     (Option B 25s User Clone / Rafael PT)
                                        │
                                        ▼
                       [S20 FE Hardware Speaker Playback]
                         (PulseAudio AAudio Sink @ 30%)
"""

import os
import sys
import time
import json
import base64
import re
import shlex
import threading
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import config
import optical_ingestion

# --- Cluster Endpoints & Hardware Addresses (see config.py; env-overridable) ---
CONTAINER_IP = config.ADB_GATEWAY_HOST
DEVICE_TARGET = config.DEVICE_TARGET
LLAMA_SERVER_URL = config.LLAMA_SERVER_URL
FALLBACK_LLAMA_SERVER_URL = config.FALLBACK_LLAMA_SERVER_URL
MAX_SAFE_TEMP_C = config.MAX_SAFE_TEMP_C

# --- Edge Paths & Models ---
EDGE_TERMUX_HOME = config.EDGE_TERMUX_HOME
EDGE_MODEL = config.EDGE_MODEL
EDGE_MMPROJ = config.EDGE_MMPROJ

# --- Host Assets & Voice Profiles ---
BENCHMARK_DIR = config.BENCHMARK_DIR
POCKET_TTS_BIN = config.POCKET_TTS_BIN
VOICE_PROFILE_EN = config.VOICE_PROFILE_EN
VOICE_PROFILE_PT = config.VOICE_PROFILE_PT


def safe_remote_name(name: str) -> str:
    """Make a file name safe to embed in an edge shell command."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def adb_args(*args: str) -> List[str]:
    """adb argv for the edge device (uses the module-level gateway/target)."""
    return ["adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET, *args]


def get_edge_temperature() -> Optional[float]:
    """Reads S20 FE battery temperature in Celsius via ADB.

    Returns None when the device cannot be read, so callers fail closed
    instead of treating "unreachable" as a cool 0.0 degrees.
    """
    inner = (
        "cat /sys/class/power_supply/battery/temp 2>/dev/null || "
        "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null"
    )
    cmd = adb_args("shell", f"su -c {shlex.quote(inner)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        raw = res.stdout.strip()
        if raw and raw.lstrip("-").isdigit():
            val = float(raw)
            if val > 1000:
                return val / 1000.0
            elif val > 100:
                return val / 10.0
            return val
    except Exception:
        pass
    return None


def run_edge_command(cmd: str, timeout: int = 45, as_root: bool = False) -> tuple[str, float]:
    """Runs a shell command inside Termux on the S20 FE.

    `cmd` is quoted as a single argument for `su -c`, so quotes and shell
    metacharacters inside it reach the device shell unchanged.
    """
    if as_root:
        full_cmd = adb_args("shell", f"su -c {shlex.quote(cmd)}")
    else:
        inner = (
            f"export HOME={EDGE_TERMUX_HOME}; export PREFIX=/data/data/com.termux/files/usr; "
            f"export TMPDIR=$PREFIX/tmp; export PATH=$PREFIX/bin:$PATH; cd {EDGE_TERMUX_HOME}; {cmd}"
        )
        full_cmd = adb_args(
            "shell",
            f"su {config.EDGE_TERMUX_UID} -g 3003 -G 9997 -G 1015 -G 1077 -c {shlex.quote(inner)}",
        )
    t0 = time.time()
    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        dur = time.time() - t0
        return proc.stdout.strip(), dur
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0


def push_frame_to_edge(local_path: Path, remote_name: str) -> None:
    """Copy a frame to the S20 FE and make it readable by the Termux user."""
    remote_name = safe_remote_name(remote_name)
    subprocess.run(
        adb_args("push", str(local_path), f"/sdcard/Download/{remote_name}"),
        capture_output=True, check=False,
    )
    dest = f"{EDGE_TERMUX_HOME}/{remote_name}"
    run_edge_command(
        f"cp /sdcard/Download/{remote_name} {dest} && "
        f"chown {config.EDGE_TERMUX_UID}:{config.EDGE_TERMUX_UID} {dest} && chmod 644 {dest}",
        as_root=True,
    )


def detect_ocr_intent(query: str) -> bool:
    """Pass 0: Evaluates whether query requires fine text recognition (OCR)."""
    patterns = [
        r"\bread\b", r"\btranscribe\b", r"\bocr\b", r"\btext\b", r"\bwords\b",
        r"\bingredient\b", r"\bdosage\b", r"\bbrand\b", r"\blabel\b",
        r"\bleia\b", r"\btranscreva\b", r"\btexto\b", r"\bescrito\b",
        r"\bpalavras\b", r"\brótulo\b", r"\bbula\b", r"\bingredientes\b"
    ]
    q = query.lower()
    return any(re.search(p, q) for p in patterns)


_LOCATE_RE = re.compile(r"\b(where|locate|bounding|bbox|onde|localiz\w*)\b", re.IGNORECASE)
_PRESENCE_RE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|can|could|has|have|any|tem|t[eê]m|h[aá]|existe|existem|est[aá]|voc[eê] v[eê])\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(not sure|unsure|unclear|uncertain|maybe|might|possibly|cannot|can't|unable|n[aã]o sei|talvez)\b",
    re.IGNORECASE,
)
_YESNO_RE = re.compile(r"\b(yes|no|sim|n[aã]o)\b", re.IGNORECASE)


def classify_query(query: str) -> str:
    """Question shape: ocr | locate | presence (yes/no) | describe. Decides the tier, not a model."""
    if detect_ocr_intent(query):
        return "ocr"
    if _LOCATE_RE.search(query):
        return "locate"
    if _PRESENCE_RE.search(query):
        return "presence"
    return "describe"


def needs_escalation(text: str) -> bool:
    """True when the phone's answer to a yes/no question is empty, hedged or not a yes/no at all."""
    t = (text or "").strip()
    if not t or t.lower() == "no response parsed":
        return True
    if _HEDGE_RE.search(t):
        return True
    return not _YESNO_RE.search(t)


def detect_language(query: str) -> str:
    """Infers whether user query is Portuguese or English."""
    pt_keywords = ["o que", "tem", "mesa", "leia", "você", "xarope", "frasco", "sim", "não", "está", "onde"]
    q = query.lower()
    if any(k in q for k in pt_keywords):
        return "pt"
    return "en"


def build_smolvlm_command(image_remote_name: str, prompt: str, max_tokens: int) -> str:
    """Shell command for llama-mtmd-cli with every user-controlled value quoted."""
    formatted_prompt = (
        f"User:<image>{prompt}<end_of_utterance>\n"
        f"Assistant:"
    )
    argv = [
        "llama-mtmd-cli", "-m", EDGE_MODEL, "--mmproj", EDGE_MMPROJ,
        "--image", safe_remote_name(image_remote_name), "-p", formatted_prompt,
        "-n", str(int(max_tokens)), "-t", "4", "-Cr", "4-7", "--temp", "0.2",
    ]
    return " ".join(shlex.quote(a) for a in argv) + " 2>&1"


def query_edge_smolvlm(image_remote_name: str, prompt: str, max_tokens: int = 30) -> dict:
    """
    Runs on-device SmolVLM-256M inference via llama-mtmd-cli on S20 FE CPU.
    Uses official SmolVLM/Idefics3 prompt template: User:<image>{prompt}<end_of_utterance>\\nAssistant:
    Pins 4 threads strictly to Cortex-A77 Gold & Prime performance cores (-Cr 4-7).
    """
    cmd = build_smolvlm_command(image_remote_name, prompt, max_tokens)
    output, duration = run_edge_command(cmd, timeout=35)
    
    # Isolate generated completion block (occurs after the last vision batch encoding log)
    if "mtmd batch encoding done in" in output:
        raw_completion = output.rsplit("mtmd batch encoding done in", 1)[-1]
        if "\n" in raw_completion:
            raw_completion = raw_completion.split("\n", 1)[1]
    else:
        raw_completion = output
        
    cleaned_text = ""
    for line in raw_completion.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        if re.match(r"^\d+\.\d+\.\d+\s+[IWE]\s+", line_clean):
            continue
        if any(ign in line_clean.lower() for ign in [
            "llama_", "ggml_", "clip_", "system_info", "sampling:", "main:",
            "load_image:", "encode_image:", "total time =", "prompt eval time =",
            "eval time =", "model_load_time", "smolvlm", "warmup", "mtmd",
            "warn:", "init:", "example:"
        ]):
            continue
        cleaned_text += line_clean + " "
        
    final_text = (
        cleaned_text
        .replace("<end_of_utterance>", "")
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .strip()
    )
    return {
        "text": final_text or "No response parsed",
        "duration_sec": round(duration, 2),
        "raw_output": output
    }


_gpu_probe = (0.0, False)


def health_url(chat_url: str) -> str:
    """llama-server health endpoint derived from its chat-completions URL."""
    return chat_url.split("/v1/", 1)[0].rstrip("/") + "/health"


def gpu_awake() -> bool:
    """True when the RTX 5070 desktop's llama-server answers right now. Never wakes it; cached briefly."""
    global _gpu_probe
    if not config.PREFER_GPU or not FALLBACK_LLAMA_SERVER_URL or FALLBACK_LLAMA_SERVER_URL == LLAMA_SERVER_URL:
        return False
    now = time.time()
    probed_at, last = _gpu_probe
    if now - probed_at < config.GPU_PROBE_TTL_SEC:
        return last
    try:
        awake = requests.get(health_url(FALLBACK_LLAMA_SERVER_URL), timeout=(0.6, 1.5)).status_code == 200
    except Exception:
        awake = False
    _gpu_probe = (now, awake)
    return awake


def tier2_endpoints() -> List[str]:
    """GPU first when it is awake, then the Dell CPU llama-server."""
    return [FALLBACK_LLAMA_SERVER_URL, LLAMA_SERVER_URL] if gpu_awake() else [LLAMA_SERVER_URL]


def plan_route(query: str, crop_bbox: Optional[List[int]] = None) -> dict:
    """Pick the backend and the Tier 2 image budget for this question."""
    kind = classify_query(query)
    if gpu_awake():
        return {"kind": kind, "backend": "gpu", "tier2_px": 1024}
    if crop_bbox:
        return {"kind": kind, "backend": "homelab", "tier2_px": 1024}
    if kind == "ocr":
        return {"kind": kind, "backend": "homelab", "tier2_px": config.READ_PX}
    if kind in ("locate", "describe"):
        return {"kind": kind, "backend": "homelab", "tier2_px": config.SCENE_PX}
    return {"kind": kind, "backend": "edge", "tier2_px": config.SCENE_PX}


def query_tier2_qwen(image_path: Path, prompt: str, max_tokens: int = 150) -> dict:
    """
    Queries Tier 2 llama-server hosting Qwen2.5-VL-3B.
    Supports cold-start 503 retry resilience, token usage tracking, and automatic endpoint fallback
    between homelab core (127.0.0.1:8085) and desktop GPU node (100.77.169.15:8085).
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "model": config.VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    
    endpoints = tier2_endpoints()
        
    last_err = None
    t0 = time.time()
    
    for endpoint in endpoints:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 3s connect timeout for fast fallback, 180s read timeout for host CPU prefill
                resp = requests.post(endpoint, json=payload, timeout=(3.0, 180.0))
                dur = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    return {
                        "status": "success",
                        "content": content,
                        "duration_sec": round(dur, 2),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "endpoint": endpoint
                    }
                elif resp.status_code == 503:
                    # Model loading or slot busy, exponential backoff
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    last_err = f"HTTP {resp.status_code}: {resp.text}"
                    break
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                # Network unreachable / connection refused: break to try next endpoint immediately
                break
                
    dur = time.time() - t0
    return {
        "status": "error",
        "error": last_err or "Unknown VLM query error",
        "duration_sec": round(dur, 2),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }


def synthesize_speech(text: str, language: str, out_wav_path: Path) -> dict:
    """Synthesizes speech using Kyutai Pocket-TTS (`pocket-tts generate`)."""
    if not POCKET_TTS_BIN:
        return {
            "status": "error",
            "error": "TTS unavailable: pocket-tts is not installed (set POCKET_TTS_BIN).",
            "duration_sec": 0.0,
        }
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if language == "en":
        voice_flag = ["--voice", str(VOICE_PROFILE_EN)]
    else:
        voice_flag = ["--voice", VOICE_PROFILE_PT]

    cmd = [
        str(POCKET_TTS_BIN), "generate", "--quiet",
        "--text", text,
        *voice_flag,
        "--output-path", str(out_wav_path)
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        dur = time.time() - t0
        if proc.returncode == 0 and out_wav_path.exists():
            return {
                "status": "success",
                "wav_path": str(out_wav_path),
                "duration_sec": round(dur, 2),
                "language": language
            }
        return {"status": "error", "error": proc.stderr.strip() or proc.stdout.strip(), "duration_sec": round(dur, 2)}
    except Exception as e:
        return {"status": "error", "error": str(e), "duration_sec": round(time.time() - t0, 2)}


def play_audio_on_edge(wav_path: Path) -> bool:
    """Pushes audio payload to S20 FE and plays it aloud via PulseAudio AAudio sink."""
    remote_wav = f"/sdcard/Download/{safe_remote_name(wav_path.name)}"
    subprocess.run(adb_args("push", str(wav_path), remote_wav), capture_output=True, check=False)

    # Ensure media volume is at 30%
    subprocess.run(adb_args("shell", "cmd media_session volume --stream 3 --set 5"),
                   capture_output=True, check=False)

    # Play via paplay with dynamic timeout
    play_cmd = f"timeout 10 paplay {shlex.quote(remote_wav)}"
    out, dur = run_edge_command(play_cmd, timeout=12)
    return "TIMEOUT" not in out and "ERROR" not in out


def tts_available() -> bool:
    """True when the configured voice backend can speak right now."""
    mode = config.TTS_MODE
    if mode == "off":
        return False
    if mode == "host":
        return bool(POCKET_TTS_BIN)
    out, _ = run_edge_command(f"command -v {shlex.quote(config.EDGE_TTS_BIN)}", timeout=10)
    return out.startswith("/")


def ensure_edge_tts_server() -> bool:
    """Make sure `pocket-tts serve` is answering on the phone, starting it when it is down."""
    port = config.EDGE_TTS_PORT
    health = f"curl -fsS -m 2 http://127.0.0.1:{port}/health >/dev/null 2>&1"
    start = (
        f"HF_HUB_DISABLE_XET=1 nohup {shlex.quote(config.EDGE_TTS_BIN)} serve --host 127.0.0.1 "
        f"--port {port} --default-voice {shlex.quote(config.EDGE_VOICE_EN)} --quantize "
        f"> {shlex.quote(config.EDGE_TTS_LOG)} 2>&1 < /dev/null &"
    )
    cmd = (
        f"{health} || {{ {start} for i in $(seq 1 40); do {health} && break; sleep 1; done; }}; "
        f"{health} && echo UP || echo DOWN"
    )
    out, _ = run_edge_command(cmd, timeout=60)
    lines = out.strip().splitlines()
    return bool(lines) and lines[-1].strip() == "UP"


def _warm_synth_command(text: str, language: str, remote_wav: str) -> str:
    """curl against the resident server. --form-string keeps "@file" / "<file" in text literal."""
    argv = ["curl", "-sS", "-m", "90", "-X", "POST", f"http://127.0.0.1:{config.EDGE_TTS_PORT}/tts",
            "--form-string", f"text={text}"]
    if language != "en":
        argv += ["--form-string", f"voice_url={config.EDGE_VOICE_PT}"]
    quoted = " ".join(shlex.quote(a) for a in argv)
    return f"{quoted} -o {shlex.quote(remote_wav)} && test -s {shlex.quote(remote_wav)} && echo SYNTH_OK"


def speak_on_edge(text: str, language: str) -> dict:
    """Synthesize with pocket-tts on the S20 FE (resident server, CLI as fallback) and play it with paplay."""
    voice = config.EDGE_VOICE_EN if language == "en" else config.EDGE_VOICE_PT
    remote_wav = f"{EDGE_TERMUX_HOME}/tts_{int(time.time() * 1000)}.wav"
    t0 = time.time()
    where = None
    out = ""
    if config.TTS_WARM and ensure_edge_tts_server():
        out, _ = run_edge_command(_warm_synth_command(text, language, remote_wav), timeout=90)
        where = "s20-warm" if "SYNTH_OK" in out else None
    if where is None:
        # HF_HUB_DISABLE_XET: the Rust hf_xet downloader is not built in Termux; plain HTTP works.
        synth = (
            f"HF_HUB_DISABLE_XET=1 {shlex.quote(config.EDGE_TTS_BIN)} generate --quiet --text {shlex.quote(text)} "
            f"--voice {shlex.quote(voice)} --output-path {shlex.quote(remote_wav)} "
            f"&& test -s {shlex.quote(remote_wav)} && echo SYNTH_OK"
        )
        out, _ = run_edge_command(synth, timeout=90)
        if "SYNTH_OK" not in out:
            return {"status": "error", "played": False, "language": language,
                    "duration_sec": round(time.time() - t0, 2),
                    "error": (out or "no output from pocket-tts on the S20")[-300:]}
        where = "s20"
    synth_dur = time.time() - t0

    subprocess.run(adb_args("shell", "cmd media_session volume --stream 3 --set 5"),
                   capture_output=True, check=False)
    play_out, _ = run_edge_command(f"timeout 15 paplay {shlex.quote(remote_wav)}", timeout=20)
    played = "TIMEOUT" not in play_out and "ERROR" not in play_out
    run_edge_command(f"rm -f {shlex.quote(remote_wav)}", timeout=10)
    return {"status": "success", "played": played, "language": language,
            "duration_sec": round(synth_dur, 2), "where": where}


def speak(text: str, language: str) -> dict:
    """Speak `text` according to AMBIENT_TTS_MODE (edge | host | off)."""
    mode = config.TTS_MODE
    if mode == "off":
        return {"status": "disabled", "played": False, "language": language}
    if mode == "host":
        wav = BENCHMARK_DIR / f"ambient_speech_{int(time.time() * 1000)}.wav"
        res = synthesize_speech(text, language, wav)
        res["played"] = bool(res.get("status") == "success" and play_audio_on_edge(wav))
        return res
    return speak_on_edge(text, language)


def execute_ambient_cycle(
    query: str,
    source: str = "camera",
    lang: str = None,
    play_audio: bool = True,
    crop_bbox: Optional[List[int]] = None
) -> dict:
    """
    Executes an end-to-end ambient companion cycle:
    1. Thermal / reachability check of the S20 FE
    2. Route (no model involved): GPU desktop when awake; otherwise by question shape.
       yes/no questions start on the phone (SmolVLM, one pass); reading, locating and
       open-ended questions go straight to the Dell (512 px scenes, 768 px reading).
    3. Optical ingestion, RoI crop and visual token budgeting
    4. Phone answer, escalated only when it is empty, hedged or not a yes/no
    5. Tier 2 call (the "let me look closer" cue is spoken while it runs)
    6. Voice on the S20 FE
    """
    cycle_start = time.time()
    if lang is None:
        lang = detect_language(query)

    print(f"\n=======================================================")
    print(f" Ambient Multimodal Companion Cycle")
    print(f"=======================================================")
    print(f"• Query      : \"{query}\"")
    print(f"• Source     : {source}")
    print(f"• Language   : {lang.upper()}")
    if crop_bbox:
        print(f"• RoI Crop   : {crop_bbox} (Optical Macro Zoom)")

    # Step 0: Check Edge Device Thermal State
    temp_c = get_edge_temperature()
    if temp_c is None:
        print("🚨 EDGE UNREACHABLE: cannot read S20 FE temperature over ADB. Aborting (fail closed).")
        return {"status": "aborted_edge_unreachable", "adb_target": f"{CONTAINER_IP} -> {DEVICE_TARGET}"}
    print(f"• Edge Temp  : {temp_c:.1f}°C (Threshold: {MAX_SAFE_TEMP_C}°C)")
    if temp_c >= MAX_SAFE_TEMP_C:
        print(f"🚨 THERMAL CIRCUIT BREAKER: Edge temperature {temp_c:.1f}°C exceeds threshold! Aborting.")
        return {"status": "aborted_thermal", "temp_c": temp_c}

    plan = plan_route(query, crop_bbox)
    print(f"• Route      : {plan['kind']} -> {plan['backend']} (Tier 2 image {plan['tier2_px']}px)")

    # Step 1: Optical Acquisition
    print(f"\n📷 [Optical Ingestion] Acquiring frame from source: '{source}'...")
    t0_opt = time.time()
    raw_img = optical_ingestion.acquire_image(source)
    opt_dur = round(time.time() - t0_opt, 2)
    print(f"   ✓ Acquired: {raw_img.name} ({opt_dur}s)")

    tier2_img_local = BENCHMARK_DIR / f"daemon_tier2_{plan['tier2_px']}px_{raw_img.stem}.jpg"
    _, tier2_meta = optical_ingestion.prepare_budgeted_image(
        raw_img, plan["tier2_px"], tier2_img_local, crop_bbox=crop_bbox)

    telemetry = {
        "query": query,
        "language": lang,
        "route": plan,
        "optical_source": source,
        "image_file": str(raw_img),
        "optical_acquisition_sec": opt_dur,
        "temp_start_c": temp_c,
        "crop_applied": tier2_meta.get("crop_applied", False),
        "crop_info": tier2_meta.get("crop_info"),
        "canvas_size": [tier2_meta.get("original_width"), tier2_meta.get("original_height")],
        "grounding_matches": [],
        "steps": [{"step": "Route", "decision": plan}],
        "resolution_tier": "",
        "final_answer": "",
        "total_latency_sec": 0.0
    }

    final_answer = ""
    escalated = False

    if plan["backend"] == "edge":
        # Yes/no question: one SmolVLM pass on the phone. No self-critique call; the answer is
        # checked by needs_escalation() (empty, hedged, or not a yes/no), which costs nothing.
        edge_img_local = BENCHMARK_DIR / f"daemon_edge_384px_{raw_img.stem}.jpg"
        optical_ingestion.prepare_budgeted_image(raw_img, 384, edge_img_local, crop_bbox=crop_bbox)
        edge_remote_name = safe_remote_name(f"edge_frame_{raw_img.stem}.jpg")
        push_frame_to_edge(edge_img_local, edge_remote_name)

        print("\n🔍 [Edge] Running SmolVLM-256M on Snapdragon 865...")
        p1_res = query_edge_smolvlm(edge_remote_name, query, max_tokens=25)
        print(f"   ✓ Edge Output: \"{p1_res['text']}\" ({p1_res['duration_sec']}s)")
        telemetry["steps"].append({"step": "Edge_Answer", "result": p1_res})
        if needs_escalation(p1_res["text"]):
            escalated = True
            print("\n⚠️  [Escalating to Tier 2] Edge answer was empty, hedged or not a yes/no.")
        else:
            print("\n✅ [Resolved at Edge] Zero homelab egress.")
            telemetry["resolution_tier"] = "Tier 1 (Edge On-Device)"
            final_answer = p1_res["text"]

    if not final_answer:
        if plan["backend"] == "gpu":
            telemetry["resolution_tier"] = "Tier 2 (GPU)"
        elif escalated:
            telemetry["resolution_tier"] = "Tier 2 (Escalation via Homelab)"
        else:
            telemetry["resolution_tier"] = "Tier 2 (Direct)"

        # The CPU path takes tens of seconds: say something while it runs. The GPU answers
        # in about a second, so a cue would only delay the answer.
        cue_thread = None
        if play_audio and plan["backend"] != "gpu":
            cue_text = "Let me look closer." if lang == "en" else "Deixe-me olhar com mais atenção."
            print(f"   🗣️  [Voice Cue]: \"{cue_text}\"")
            cue_thread = threading.Thread(target=speak, args=(cue_text, lang), daemon=True)
            cue_thread.start()

        print(f"   🏠 Querying Tier 2 llama-server ({plan['tier2_px']}px, backend {plan['backend']})...")
        t2_res = query_tier2_qwen(tier2_img_local, query, max_tokens=150)
        if cue_thread is not None:
            cue_thread.join(timeout=30)
        final_answer = t2_res.get("content", "")
        print(f"   ✓ Tier 2 Output: \"{final_answer}\" ({t2_res.get('duration_sec')}s | {t2_res.get('prompt_tokens', 0)} in, {t2_res.get('completion_tokens', 0)} out)")
        telemetry["steps"].append({"step": "Tier_2_Inference", "result": t2_res})

        telemetry["grounding_matches"] = optical_ingestion.parse_grounding_coordinates(
            final_answer,
            orig_w=tier2_meta.get("original_width", 1000),
            orig_h=tier2_meta.get("original_height", 1000),
            crop_info=tier2_meta.get("crop_info")
        )

    telemetry["final_answer"] = final_answer
    
    # Step 3: Pocket-TTS Audio Synthesis
    if play_audio and final_answer:
        clean_speech = re.sub(r"[\*#_`]", "", final_answer).strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean_speech)
        speech_text = " ".join(sentences[:2]) if len(sentences) > 2 else clean_speech
        
        print(f"\n🎵 [Voice] Speaking via Pocket-TTS ({lang.upper()}, mode={config.TTS_MODE})...")
        print(f"   Speech Prompt: \"{speech_text}\"")
        speech = speak(speech_text, lang)
        telemetry["steps"].append({"step": "Speech", "result": speech})
        if speech.get("played"):
            print("   ✓ Spoken on the S20 FE at ~30% volume.")

    telemetry["total_latency_sec"] = round(time.time() - cycle_start, 2)
    telemetry["temp_end_c"] = get_edge_temperature()
    end_temp = f"{telemetry['temp_end_c']:.1f}°C" if telemetry["temp_end_c"] is not None else "n/a"
    print(f"\n🎉 [Cycle Complete] Total Latency: {telemetry['total_latency_sec']}s | End Temp: {end_temp}")
    return telemetry


def main():
    parser = argparse.ArgumentParser(description="Ambient Multimodal Desktop Companion Daemon (PRJ-12 / ADR-42)")
    parser.add_argument("--query", "-q", type=str, default="Describe what is on the desk in 15 words.", help="User question or trigger prompt")
    parser.add_argument("--source", "-s", type=str, default="camera", help="Optical source: 'camera', 'droidcam', 'http://...', 'rtsp://...', or '/path/to/file'")
    parser.add_argument("--lang", "-l", type=str, choices=["en", "pt"], default=None, help="Language override (en or pt)")
    parser.add_argument("--crop-bbox", "--crop", nargs=4, type=int, default=None, metavar=("YMIN", "XMIN", "YMAX", "XMAX"), help="Normalized RoI crop [ymin xmin ymax xmax] (0-1000)")
    parser.add_argument("--no-audio", action="store_true", help="Skip speech synthesis and playback")
    parser.add_argument("--json", action="store_true", help="Output telemetry in JSON format")
    args = parser.parse_args()
    
    res = execute_ambient_cycle(
        query=args.query,
        source=args.source,
        lang=args.lang,
        play_audio=not args.no_audio,
        crop_bbox=args.crop_bbox
    )
    
    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
