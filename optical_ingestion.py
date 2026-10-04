#!/usr/bin/env python3
"""
Pluggable Optical Ingestion Engine for Ambient Multimodal Desktop Companion.
Supports multiple visual sources:
  1. 'termux_camera': Live 12MP optical capture via Galaxy S20 FE Termux API & ADB.
  2. 'droidcam': Sub-second HTTP JPEG frame capture from DroidCamX (port 4747).
  3. 'http://...' or 'https://...': Arbitrary IP camera snapshot or MJPEG stream.
  4. 'rtsp://...': Network RTSP camera keyframe stream.
  5. 'file:...': Local image path on disk or Dropzone asset.
  6. 'screen' / 'active_window': Desktop screen capture.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageOps

CONTAINER_IP = os.getenv("ADB_GATEWAY_HOST", "172.17.0.2")
DEVICE_TARGET = os.getenv("S20_DEVICE_TARGET", "100.115.165.41:5555")
DROIDCAM_DEFAULT_URL = os.getenv("DROIDCAM_URL", "http://100.115.165.41:4747/cam/1/frame.jpg")
DEFAULT_CACHE_DIR = Path("/home/tlima/Enterprise_Hub/data/media/merged/vision/validation")


def wake_and_unlock_edge() -> bool:
    """Wakes the S20 FE screen and dismisses keyguard via ADB."""
    cmd = [
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "shell", "input keyevent KEYCODE_WAKEUP && wm dismiss-keyguard"
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=5, check=False)
        return True
    except Exception:
        return False


def capture_termux_camera(dest_path: Path, camera_id: int = 0) -> Path:
    """Captures native optical photo via Termux API on Galaxy S20 FE."""
    wake_and_unlock_edge()
    remote_tmp = "/sdcard/live_snap.jpg"
    
    # Trigger camera photo via Termux API
    snap_cmd = [
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "shell", f"su -c '/data/data/com.termux/files/usr/bin/termux-camera-photo -c {camera_id} {remote_tmp}'"
    ]
    res = subprocess.run(snap_cmd, capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(f"Termux camera capture failed: {res.stderr.strip() or res.stdout.strip()}")

    # Pull capture to homelab host
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    pull_cmd = [
        "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
        "pull", remote_tmp, str(dest_path)
    ]
    pull_res = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=15)
    if pull_res.returncode != 0:
        raise RuntimeError(f"Failed to pull photo from S20 FE: {pull_res.stderr.strip()}")

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Captured photo is empty or missing: {dest_path}")

    return dest_path


def capture_droidcam_http(dest_path: Path, url: str = DROIDCAM_DEFAULT_URL, timeout: int = 5) -> Path:
    """
    Captures a single JPEG frame from DroidCamX or an IP Webcam HTTP endpoint.
    Supports direct snapshot endpoints and MJPEG streams.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try direct JPEG snapshot
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "").lower()
            if "multipart" in content_type:
                # MJPEG stream: parse first JPEG frame
                bytes_buf = b""
                for chunk in resp.iter_content(chunk_size=4096):
                    bytes_buf += chunk
                    a = bytes_buf.find(b"\xff\xd8")
                    b = bytes_buf.find(b"\xff\xd9")
                    if a != -1 and b != -1 and b > a:
                        jpg_data = bytes_buf[a:b+2]
                        with open(dest_path, "wb") as f:
                            f.write(jpg_data)
                        return dest_path
            else:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return dest_path
    except requests.RequestException as e:
        # If DroidCam is not running, attempt launching via ADB intent
        try:
            subprocess.run([
                "adb", "-H", CONTAINER_IP, "-s", DEVICE_TARGET,
                "shell", "monkey -p com.dev47apps.droidcamx -c android.intent.category.LAUNCHER 1"
            ], capture_output=True, timeout=5, check=False)
            time.sleep(1.5)
            # Retry request once
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return dest_path
        except Exception:
            pass
        raise RuntimeError(f"DroidCam capture failed from {url}: {e}")

    raise RuntimeError(f"Unexpected response from DroidCam: {resp.status_code}")


def capture_rtsp_stream(dest_path: Path, rtsp_url: str, timeout: int = 8) -> Path:
    """Extracts a single keyframe from an RTSP stream using ffmpeg or socket."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vframes", "1",
        "-q:v", "2",
        str(dest_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path
    except FileNotFoundError:
        pass  # ffmpeg not available
    except Exception as e:
        raise RuntimeError(f"RTSP capture failed: {e}")

    raise RuntimeError("RTSP capture requires ffmpeg installed on host.")


def capture_desktop_screen(dest_path: Path) -> Path:
    """Captures desktop screen via PIL ImageGrab or X11 utility."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.save(dest_path, format="JPEG", quality=92)
        return dest_path
    except Exception:
        # Fallback to scrot or xwd on Linux if available
        for util in ["scrot", "import"]:
            try:
                res = subprocess.run([util, "-window", "root", str(dest_path)], capture_output=True)
                if res.returncode == 0 and dest_path.exists():
                    return dest_path
            except Exception:
                pass
    raise RuntimeError("Desktop screen capture failed (no display or capture utility available).")


def acquire_image(source: str, dest_path: Optional[Path] = None) -> Path:
    """
    Master optical acquisition router.
    Resolves image from any supported source:
      - 'termux_camera' / 's20_camera': Optical snapshot via S20 FE Termux API.
      - 'droidcam': HTTP snapshot from DroidCamX stream.
      - 'http://...' / 'https://...': Remote HTTP camera endpoint.
      - 'rtsp://...': RTSP live stream.
      - 'screen' / 'desktop': Desktop display capture.
      - 'file:<path>' / '<path>': Existing local file path.
    """
    if dest_path is None:
        timestamp = int(time.time() * 1000)
        dest_path = DEFAULT_CACHE_DIR / f"acquired_optical_{timestamp}.jpg"

    src_lower = source.strip().lower()

    if src_lower in ("termux_camera", "s20_camera", "phone_camera", "camera"):
        return capture_termux_camera(dest_path)

    elif src_lower in ("droidcam", "droidcamx"):
        return capture_droidcam_http(dest_path, DROIDCAM_DEFAULT_URL)

    elif src_lower.startswith("http://") or src_lower.startswith("https://"):
        return capture_droidcam_http(dest_path, source)

    elif src_lower.startswith("rtsp://"):
        return capture_rtsp_stream(dest_path, source)

    elif src_lower in ("screen", "desktop", "display"):
        return capture_desktop_screen(dest_path)

    else:
        # Local file path
        clean_path = source
        if clean_path.startswith("file://"):
            clean_path = clean_path[7:]
        elif clean_path.startswith("file:"):
            clean_path = clean_path[5:]
        
        path_obj = Path(clean_path).expanduser().resolve()
        if path_obj.exists() and path_obj.is_file():
            return path_obj
        else:
            raise FileNotFoundError(f"Optical source file '{source}' does not exist.")


def prepare_budgeted_image(src_path: Path, max_dim: int, dest_path: Path) -> Path:
    """
    Downsamples an image preserving aspect ratio with Lanczos filtering,
    applying EXIF auto-rotation to ensure vertical smartphone captures stay upright.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if max(w, h) > max_dim:
            if w >= h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest_path, format="JPEG", quality=88, optimize=True)
    return dest_path
