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
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import optical_ingestion

# --- Cluster Endpoints & Hardware Addresses ---
CONTAINER_IP = os.getenv("ADB_GATEWAY_HOST", "172.17.0.2")
DEVICE_TARGET = os.getenv("S20_DEVICE_TARGET", "100.115.165.41:5555")
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8085/v1/chat/completions")
FALLBACK_LLAMA_SERVER_URL = os.getenv("FALLBACK_LLAMA_SERVER_URL", "http://100.77.169.15:8085/v1/chat/completions")
MAX_SAFE_TEMP_C = 40.0

# --- Edge Paths & Models ---
EDGE_TERMUX_HOME = "/data/data/com.termux/files/home"
EDGE_MODEL = "models/SmolVLM-256M-Instruct-Q8_0.gguf"
EDGE_MMPROJ = "models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

# --- Host Assets & Voice Profiles ---
WORKSPACE_DIR = Path("/home/tlima/Enterprise_Hub")
BENCHMARK_DIR = WORKSPACE_DIR / "data/media/merged/vision/benchmark"
GDRIVE_TTS_DIR = WORKSPACE_DIR / "data/media/cloud_drive/VoiceRecordingsTTS/SynthesizedOutput"
DROPZONE_TTS_DIR = WORKSPACE_DIR / "data/dropzone/files/tts_cloned_output"
VENV_PYTHON = WORKSPACE_DIR / ".venv/bin/python3"
POCKET_TTS_BIN = WORKSPACE_DIR / ".venv/bin/pocket-tts"

VOICE_PROFILE_EN = BENCHMARK_DIR / "voice_profile_user_optionB_full25s.safetensors"
VOICE_PROFILE_PT = "rafael"


def get_edge_temperature() -> float:
    """Reads S20 FE battery temperature in Celsius via ADB."""
    cmd = [
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "shell", "su -c 'cat /sys/class/power_supply/battery/temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null'"
    ]
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
    return 0.0


def run_edge_command(cmd: str, timeout: int = 45, as_root: bool = False) -> tuple[str, float]:
    """Runs a shell command inside Termux on the S20 FE."""
    if as_root:
        full_cmd = ["adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET, "shell", f"su -c '{cmd}'"]
    else:
        full_cmd = [
            "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
            "shell",
            f"su u0_a356 -g 3003 -G 9997 -G 1015 -G 1077 -c "
            f"'export HOME={EDGE_TERMUX_HOME}; export PREFIX=/data/data/com.termux/files/usr; "
            f"export TMPDIR=$PREFIX/tmp; export PATH=$PREFIX/bin:$PATH; cd {EDGE_TERMUX_HOME}; {cmd}'"
        ]
    t0 = time.time()
    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        dur = time.time() - t0
        return proc.stdout.strip(), dur
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0


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


def detect_language(query: str) -> str:
    """Infers whether user query is Portuguese or English."""
    pt_keywords = ["o que", "tem", "mesa", "leia", "você", "xarope", "frasco", "sim", "não", "está", "onde"]
    q = query.lower()
    if any(k in q for k in pt_keywords):
        return "pt"
    return "en"


def query_edge_smolvlm(image_remote_name: str, prompt: str, max_tokens: int = 30) -> dict:
    """
    Runs on-device SmolVLM-256M inference via llama-mtmd-cli on S20 FE CPU.
    Uses official SmolVLM/Idefics3 prompt template: User:<image>{prompt}<end_of_utterance>\\nAssistant:
    Pins 4 threads strictly to Cortex-A77 Gold & Prime performance cores (-Cr 4-7).
    """
    formatted_prompt = (
        f"User:<image>{prompt}<end_of_utterance>\n"
        f"Assistant:"
    )
    safe_prompt = formatted_prompt.replace('"', '\\"').replace("'", "'\\''")
    cmd = (
        f"llama-mtmd-cli -m {EDGE_MODEL} --mmproj {EDGE_MMPROJ} "
        f"--image {image_remote_name} -p \"{safe_prompt}\" "
        f"-n {max_tokens} -t 4 -Cr 4-7 --temp 0.2 2>&1"
    )
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


def query_tier2_qwen(image_path: Path, prompt: str, max_tokens: int = 150) -> dict:
    """
    Queries Tier 2 llama-server hosting Qwen2.5-VL-3B.
    Supports cold-start 503 retry resilience, token usage tracking, and automatic endpoint fallback
    between homelab core (127.0.0.1:8085) and desktop GPU node (100.77.169.15:8085).
    """
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "model": "qwen2.5vl:3b",
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
    
    endpoints = [LLAMA_SERVER_URL]
    if FALLBACK_LLAMA_SERVER_URL and FALLBACK_LLAMA_SERVER_URL != LLAMA_SERVER_URL:
        endpoints.append(FALLBACK_LLAMA_SERVER_URL)
        
    last_err = None
    t0 = time.time()
    
    for endpoint in endpoints:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(endpoint, json=payload, timeout=60)
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
    """Synthesizes speech using Kyutai Pocket-TTS."""
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    if language == "en":
        voice_flag = ["--voice", str(VOICE_PROFILE_EN)]
    else:
        voice_flag = ["--voice", VOICE_PROFILE_PT]
        
    cmd = [
        str(POCKET_TTS_BIN),
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
    remote_wav = f"/sdcard/Download/{wav_path.name}"
    subprocess.run([
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "push", str(wav_path), remote_wav
    ], capture_output=True, check=False)
    
    # Ensure media volume is at 30%
    subprocess.run([
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "shell", "cmd media_session volume --stream 3 --set 5"
    ], capture_output=True, check=False)
    
    # Play via paplay with dynamic timeout
    play_cmd = f"timeout 10 paplay {remote_wav}"
    out, dur = run_edge_command(play_cmd, timeout=12)
    return "TIMEOUT" not in out and "ERROR" not in out


def execute_ambient_cycle(
    query: str,
    source: str = "camera",
    lang: str = None,
    play_audio: bool = True,
    crop_bbox: Optional[List[int]] = None
) -> dict:
    """
    Executes an end-to-end ambient companion cycle:
    1. Optical Ingestion (Termux Camera, DroidCam, RTSP, or File)
    2. RoI Crop-on-Demand & Visual Token Budgeting
    3. Pass 0: Intent Filter
    4. Pass 1: Edge Triage (SmolVLM-256M)
    5. Pass 2: Edge Self-Critique
    6. Tier 2 Homelab Escalation & Coordinate Remapping (if needed)
    7. Pocket-TTS Voice Synthesis
    8. Galaxy S20 FE Audio Playback
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
    print(f"• Edge Temp  : {temp_c:.1f}°C (Threshold: {MAX_SAFE_TEMP_C}°C)")
    if temp_c >= MAX_SAFE_TEMP_C:
        print(f"🚨 THERMAL CIRCUIT BREAKER: Edge temperature {temp_c:.1f}°C exceeds threshold! Aborting.")
        return {"status": "aborted_thermal", "temp_c": temp_c}
        
    # Step 1: Optical Acquisition
    print(f"\n📷 [Optical Ingestion] Acquiring frame from source: '{source}'...")
    t0_opt = time.time()
    raw_img = optical_ingestion.acquire_image(source)
    opt_dur = round(time.time() - t0_opt, 2)
    print(f"   ✓ Acquired: {raw_img.name} ({opt_dur}s)")
    
    # Budget frames (384px for Edge SmolVLM, 1024px budgeted for Tier 2 Qwen2.5-VL)
    edge_img_local = BENCHMARK_DIR / f"daemon_edge_384px_{raw_img.stem}.jpg"
    tier2_img_local = BENCHMARK_DIR / f"daemon_tier2_1024px_{raw_img.stem}.jpg"
    _, edge_meta = optical_ingestion.prepare_budgeted_image(raw_img, 384, edge_img_local, crop_bbox=crop_bbox)
    _, tier2_meta = optical_ingestion.prepare_budgeted_image(raw_img, 1024, tier2_img_local, crop_bbox=crop_bbox)
    
    telemetry = {
        "query": query,
        "language": lang,
        "optical_source": source,
        "image_file": str(raw_img),
        "optical_acquisition_sec": opt_dur,
        "temp_start_c": temp_c,
        "crop_applied": tier2_meta.get("crop_applied", False),
        "crop_info": tier2_meta.get("crop_info"),
        "canvas_size": [tier2_meta.get("original_width"), tier2_meta.get("original_height")],
        "grounding_matches": [],
        "steps": [],
        "resolution_tier": "",
        "final_answer": "",
        "total_latency_sec": 0.0
    }
    
    # Push edge frame to Termux on S20 FE
    edge_remote_name = f"edge_frame_{raw_img.stem}.jpg"
    subprocess.run([
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "push", str(edge_img_local), f"/sdcard/Download/{edge_remote_name}"
    ], capture_output=True, check=False)
    run_edge_command(f"cp /sdcard/Download/{edge_remote_name} {EDGE_TERMUX_HOME}/{edge_remote_name} && chown u0_a356:u0_a356 {EDGE_TERMUX_HOME}/{edge_remote_name} && chmod 644 {EDGE_TERMUX_HOME}/{edge_remote_name}", as_root=True)
    
    # Pass 0: Intent Filter
    is_ocr = detect_ocr_intent(query)
    final_answer = ""
    
    if is_ocr or crop_bbox:
        reason = "RoI crop zoom active" if crop_bbox else "OCR match"
        print(f"\n⚡ [Pass 0: Intent Filter] High-precision vision intent ({reason})! Bypassing Edge VLM directly to Tier 2.")
        telemetry["steps"].append({"step": "Pass_0_Intent_Filter", "decision": "Bypass to Tier 2", "reason": reason})
        telemetry["resolution_tier"] = "Tier 2 (Direct Escalation)"
        
        cue_text = "Reading text with the homelab server..." if lang == "en" else "Lendo o texto com o servidor do homelab..."
        print(f"   🗣️  [Voice Cue]: \"{cue_text}\"")
        
        print(f"   🏠 Querying Homelab llama-server (1024px budgeted)...")
        t2_res = query_tier2_qwen(tier2_img_local, query, max_tokens=150)
        final_answer = t2_res.get("content", "")
        print(f"   ✓ Homelab Output: \"{final_answer}\" ({t2_res.get('duration_sec')}s | {t2_res.get('prompt_tokens', 0)} in, {t2_res.get('completion_tokens', 0)} out)")
        telemetry["steps"].append({"step": "Tier_2_Homelab_Inference", "result": t2_res})
        
        boxes = optical_ingestion.parse_grounding_coordinates(
            final_answer,
            orig_w=tier2_meta.get("original_width", 1000),
            orig_h=tier2_meta.get("original_height", 1000),
            crop_info=tier2_meta.get("crop_info")
        )
        telemetry["grounding_matches"] = boxes
        
    else:
        # Pass 1: Edge Triage via SmolVLM-256M
        print("\n🔍 [Pass 1: Edge Triage] Running SmolVLM-256M on Snapdragon 865...")
        p1_res = query_edge_smolvlm(edge_remote_name, query, max_tokens=25)
        print(f"   ✓ Edge Pass 1 Output: \"{p1_res['text']}\" ({p1_res['duration_sec']}s)")
        telemetry["steps"].append({"step": "Pass_1_Edge_Triage", "result": p1_res})
        
        # Pass 2: Self-Critique Verification
        critique_prompt = f"Is this observation completely clear and certain: '{p1_res['text']}'? Answer YES or NO."
        print(f"   🧐 [Pass 2: Edge Self-Critique] Verifying certainty: \"{critique_prompt}\"...")
        p2_res = query_edge_smolvlm(edge_remote_name, critique_prompt, max_tokens=10)
        print(f"   ✓ Edge Pass 2 Output: \"{p2_res['text']}\" ({p2_res['duration_sec']}s)")
        telemetry["steps"].append({"step": "Pass_2_Edge_Critique", "result": p2_res})
        
        # Evaluate Decision
        p1_up = p1_res["text"].upper()
        p2_up = p2_res["text"].upper()
        is_confident = ("YES" in p2_up or "SIM" in p2_up) and not ("UNCLEAR" in p2_up or "NO" in p2_up)
        
        if is_confident and len(p1_res["text"].strip()) > 3:
            print("\n✅ [Resolved at Edge] High confidence verified! Zero homelab egress.")
            telemetry["resolution_tier"] = "Tier 1 (Edge On-Device)"
            final_answer = p1_res["text"]
        else:
            print("\n⚠️  [Escalating to Tier 2] Ambiguity or fine-detail flagged by self-critique.")
            telemetry["resolution_tier"] = "Tier 2 (Escalation via Homelab)"
            
            cue_text = "Let me check with the homelab server..." if lang == "en" else "Deixe-me verificar com o servidor do homelab..."
            print(f"   🗣️  [Voice Cue]: \"{cue_text}\"")
            
            print(f"   🏠 Querying Homelab llama-server (1024px budgeted)...")
            t2_res = query_tier2_qwen(tier2_img_local, query, max_tokens=150)
            final_answer = t2_res.get("content", "")
            print(f"   ✓ Homelab Output: \"{final_answer}\" ({t2_res.get('duration_sec')}s | {t2_res.get('prompt_tokens', 0)} in, {t2_res.get('completion_tokens', 0)} out)")
            telemetry["steps"].append({"step": "Tier_2_Homelab_Inference", "result": t2_res})
            
            boxes = optical_ingestion.parse_grounding_coordinates(
                final_answer,
                orig_w=tier2_meta.get("original_width", 1000),
                orig_h=tier2_meta.get("original_height", 1000),
                crop_info=tier2_meta.get("crop_info")
            )
            telemetry["grounding_matches"] = boxes
            
    telemetry["final_answer"] = final_answer
    
    # Step 3: Pocket-TTS Audio Synthesis
    if play_audio and final_answer:
        clean_speech = re.sub(r"[\*#_`]", "", final_answer).strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean_speech)
        speech_text = " ".join(sentences[:2]) if len(sentences) > 2 else clean_speech
        
        timestamp = int(time.time())
        out_wav = BENCHMARK_DIR / f"ambient_resp_{timestamp}.wav"
        print(f"\n🎵 [Voice Synthesis] Synthesizing speech via Pocket-TTS ({lang.upper()})...")
        print(f"   Speech Prompt: \"{speech_text}\"")
        tts_res = synthesize_speech(speech_text, lang, out_wav)
        telemetry["steps"].append({"step": "Pocket_TTS_Synthesis", "result": tts_res})
        
        if tts_res.get("status") == "success":
            print(f"\n🔊 [Edge Speaker] Dispatching audio to Galaxy S20 FE hardware speaker...")
            play_ok = play_audio_on_edge(out_wav)
            telemetry["steps"].append({"step": "Edge_Audio_Playback", "played": play_ok})
            if play_ok:
                print("   ✓ Hardware audio playback confirmed at ~30% volume.")
                
    telemetry["total_latency_sec"] = round(time.time() - cycle_start, 2)
    telemetry["temp_end_c"] = get_edge_temperature()
    print(f"\n🎉 [Cycle Complete] Total Latency: {telemetry['total_latency_sec']}s | End Temp: {telemetry['temp_end_c']:.1f}°C")
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
