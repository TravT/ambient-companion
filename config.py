#!/usr/bin/env python3
"""
Environment-driven settings for the ambient companion (PRJ-12 / ADR-42).

Defaults reproduce the original host behaviour, so the CLI keeps working on the
Dell unchanged. The container overrides them through the Nomad job `env` block.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

# --- Data layout (host: /home/tlima/Enterprise_Hub/data, container: /data) ---
DATA_DIR = Path(os.getenv("AMBIENT_DATA_DIR", "/home/tlima/Enterprise_Hub/data"))
BENCHMARK_DIR = Path(os.getenv("AMBIENT_BENCHMARK_DIR", str(DATA_DIR / "media/merged/vision/benchmark")))
VALIDATION_DIR = Path(os.getenv("AMBIENT_VALIDATION_DIR", str(DATA_DIR / "media/merged/vision/validation")))
DROPZONE_DIR = Path(os.getenv("AMBIENT_DROPZONE_DIR", str(DATA_DIR / "dropzone/files")))

# --- Edge device (S20 FE) reached through the ws-scrcpy ADB server ---
ADB_GATEWAY_HOST = os.getenv("ADB_GATEWAY_HOST", "127.0.0.1")
DEVICE_TARGET = os.getenv("S20_DEVICE_TARGET", "100.115.165.41:5555")
EDGE_TERMUX_UID = os.getenv("EDGE_TERMUX_UID", "u0_a356")
EDGE_TERMUX_HOME = "/data/data/com.termux/files/home"
EDGE_MODEL = os.getenv("EDGE_MODEL", "models/SmolVLM-256M-Instruct-Q8_0.gguf")
EDGE_MMPROJ = os.getenv("EDGE_MMPROJ", "models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
MAX_SAFE_TEMP_C = float(os.getenv("MAX_SAFE_TEMP_C", "40.0"))
DROIDCAM_URL = os.getenv("DROIDCAM_URL", "http://100.115.165.41:4747/cam/1/frame.jpg")

# --- Tier 2 (llama-server hosting Qwen2.5-VL) ---
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8085/v1/chat/completions")
FALLBACK_LLAMA_SERVER_URL = os.getenv(
    "FALLBACK_LLAMA_SERVER_URL", "http://100.77.169.15:8085/v1/chat/completions"
)
VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")
# The fallback URL is the RTX 5070 desktop. It is used FIRST when it answers /health (it is awake),
# and never woken by the companion (the owner decides when it runs).
PREFER_GPU = os.getenv("AMBIENT_PREFER_GPU", "1") == "1"
# MateBook satellite (omarchy-station): native llama-server on the tailnet (satellite_vlm role), about 2x
# the Dell's speed (512 px in 14 s against 31 s). Used before the Dell when its /health answers; set
# SATELLITE_LLAMA_SERVER_URL="" to disable. It is never woken by the companion.
SATELLITE_LLAMA_SERVER_URL = os.getenv(
    "SATELLITE_LLAMA_SERVER_URL", "http://100.105.6.62:8090/v1/chat/completions"
)
GPU_PROBE_TTL_SEC = float(os.getenv("AMBIENT_GPU_PROBE_TTL", "15"))
# Longest side sent to Tier 2 on the Dell CPU (prefill is ~10 visual tokens/s, so cost ~ pixels):
# scenes 512 px (~30 s), reading 768 px (~55 s), crops up to 1024 px native. The GPU always gets 1024.
SCENE_PX = int(os.getenv("AMBIENT_SCENE_PX", "512"))
# llama-server --image-min-tokens on every Tier 2 backend (Dell job, satellite service): it decides how the
# image is resized before the model sees it, which the grounding coordinates are relative to.
VLM_IMAGE_MIN_TOKENS = int(os.getenv("VLM_IMAGE_MIN_TOKENS", "256"))
READ_PX = int(os.getenv("AMBIENT_READ_PX", "768"))
# Locate questions (boxes, click targets) need the full 1024 px: measured click error on a mock page with known
# positions was 7-60 px at 512, 10-173 px at 768 and 0-6 px at 1024.
GROUND_PX = int(os.getenv("AMBIENT_GROUND_PX", "1024"))

# --- Voice ---
# edge: pocket-tts runs on the S20 FE (Termux) and plays through paplay (default, PRJ-12 Phase 10)
# host: synthesize here with POCKET_TTS_BIN, push the WAV to the phone (dev CLI on the Dell)
# off:  never speak
TTS_MODE = os.getenv("AMBIENT_TTS_MODE", "edge").lower()
EDGE_TTS_BIN = os.getenv("EDGE_TTS_BIN", "pocket-tts")
EDGE_VOICE_EN = os.getenv(
    "EDGE_VOICE_EN", f"{EDGE_TERMUX_HOME}/voices/voice_profile_user_optionB_full25s.safetensors"
)
EDGE_VOICE_PT = os.getenv("EDGE_VOICE_PT", "rafael")
# Keep the pocket-tts model resident on the phone (about 0.9 GB RAM): a short phrase takes ~2 s
# instead of ~8 s because torch and the weights are not reloaded on every call.
TTS_WARM = os.getenv("AMBIENT_TTS_WARM", "1") == "1"
EDGE_TTS_PORT = int(os.getenv("EDGE_TTS_PORT", "8765"))
EDGE_TTS_LOG = os.getenv("EDGE_TTS_LOG", f"{EDGE_TERMUX_HOME}/tts-serve.log")
# Host mode only (the slim container image ships without pocket-tts):
POCKET_TTS_BIN: Optional[str] = (
    os.getenv("POCKET_TTS_BIN")
    or shutil.which("pocket-tts")
    or (str(DATA_DIR.parent / ".venv/bin/pocket-tts")
        if (DATA_DIR.parent / ".venv/bin/pocket-tts").exists() else None)
)
VOICE_PROFILE_EN = Path(os.getenv(
    "VOICE_PROFILE_EN", str(BENCHMARK_DIR / "voice_profile_user_optionB_full25s.safetensors")
))
VOICE_PROFILE_PT = os.getenv("VOICE_PROFILE_PT", "rafael")

# --- HTTP service ---
SERVE_HOST = os.getenv("AMBIENT_HOST", "127.0.0.1")
SERVE_PORT = int(os.getenv("AMBIENT_PORT", "8089"))
# Bearer token for POST /mcp. Empty disables auth (loopback-only use).
API_TOKEN = os.getenv("AMBIENT_API_TOKEN", "")
# Seconds a tool call waits for the camera/ADB before answering "busy".
HARDWARE_LOCK_TIMEOUT_SEC = float(os.getenv("AMBIENT_LOCK_TIMEOUT", "240"))

# --- Safety and housekeeping ---
RETENTION_DAYS = float(os.getenv("AMBIENT_RETENTION_DAYS", "7"))


def allowed_dirs() -> Optional[List[Path]]:
    """Directories the `file` optical source may read.

    None means unrestricted (host CLI default). The container sets
    AMBIENT_ALLOWED_DIRS (colon separated) so agents cannot read other paths.
    """
    raw = os.getenv("AMBIENT_ALLOWED_DIRS", "").strip()
    if not raw:
        return None
    return [Path(p).expanduser().resolve() for p in raw.split(":") if p.strip()]


def adb_base() -> List[str]:
    """Argument prefix for every adb call against the edge device."""
    return ["adb", "-H", ADB_GATEWAY_HOST, "-s", DEVICE_TARGET]
