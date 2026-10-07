#!/usr/bin/env python3
"""Retention for frames and WAVs the companion leaves behind (host cache and S20 FE)."""

import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import config

LOCAL_PATTERNS = [
    "daemon_edge_*", "daemon_tier2_*", "mcp_edge_*", "mcp_tier2_*",
    "ambient_resp_*.wav", "mcp_speech_*.wav", "acquired_optical_*",
]
EDGE_PATTERNS = ["edge_frame_*", "mcp_frame_*", "ambient_resp_*", "mcp_speech_*"]


def purge_local(directory: Path, patterns: Iterable[str], max_age_days: float,
                now: Optional[float] = None) -> int:
    """Delete files matching `patterns` older than `max_age_days`. Returns the count removed."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - max_age_days * 86400
    removed = 0
    for pattern in patterns:
        for path in directory.glob(pattern):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    return removed


def purge_edge(max_age_days: float) -> bool:
    """Delete stale companion files from the S20 FE download folder and Termux home."""
    days = int(max(1, round(max_age_days)))
    names = " -o ".join(f"-name '{p}'" for p in EDGE_PATTERNS)
    find_dl = f"find /sdcard/Download -maxdepth 1 -type f \\( {names} \\) -mtime +{days} -delete"
    try:
        res = subprocess.run([*config.adb_base(), "shell", find_dl],
                             capture_output=True, text=True, timeout=15)
        return res.returncode == 0
    except Exception:
        return False


def run_once(max_age_days: Optional[float] = None) -> dict:
    """One retention pass over the local caches and the edge device."""
    days = config.RETENTION_DAYS if max_age_days is None else max_age_days
    removed = 0
    for directory in {config.BENCHMARK_DIR, config.VALIDATION_DIR}:
        removed += purge_local(directory, LOCAL_PATTERNS, days)
    return {"local_removed": removed, "edge_ok": purge_edge(days)}


def start_background(interval_sec: float = 6 * 3600) -> threading.Thread:
    """Daemon thread that runs a retention pass at start and then every `interval_sec`."""
    def loop():
        while True:
            try:
                run_once()
            except Exception:
                pass
            time.sleep(interval_sec)

    thread = threading.Thread(target=loop, name="ambient-housekeeping", daemon=True)
    thread.start()
    return thread
