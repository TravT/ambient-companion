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
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from PIL import Image, ImageOps

import config

CONTAINER_IP = config.ADB_GATEWAY_HOST
DEVICE_TARGET = config.DEVICE_TARGET
DROIDCAM_DEFAULT_URL = config.DROIDCAM_URL
DEFAULT_CACHE_DIR = config.VALIDATION_DIR


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
        roots = config.allowed_dirs()
        if roots is not None and not any(path_obj.is_relative_to(r) for r in roots):
            raise PermissionError(f"Optical source '{source}' is outside the allowed directories.")
        if path_obj.exists() and path_obj.is_file():
            return path_obj
        else:
            raise FileNotFoundError(f"Optical source file '{source}' does not exist.")


def prepare_budgeted_image(
    src_path: Path,
    max_dim: int,
    dest_path: Path,
    crop_bbox: Optional[List[int]] = None
) -> Tuple[Path, Dict[str, Any]]:
    """
    Downsamples an image preserving aspect ratio with Lanczos filtering,
    applying EXIF auto-rotation to ensure vertical smartphone captures stay upright.
    Supports RoI Crop-on-Demand (crop_bbox: [ymin, xmin, ymax, xmax] normalized to 1000)
    to perform optical macro zoom directly on uncompressed camera frames before downsampling.
    Returns (dest_path, metadata_dict).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)
        orig_w, orig_h = img.size
        meta: Dict[str, Any] = {
            "original_width": orig_w,
            "original_height": orig_h,
            "crop_applied": False,
            "crop_info": None
        }

        # 1. RoI Crop-on-Demand (Digital Optical Macro Zoom)
        if crop_bbox and len(crop_bbox) == 4:
            ymin, xmin, ymax, xmax = crop_bbox
            if 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000:
                c_ymin = int((ymin / 1000.0) * orig_h)
                c_xmin = int((xmin / 1000.0) * orig_w)
                c_ymax = int((ymax / 1000.0) * orig_h)
                c_xmax = int((xmax / 1000.0) * orig_w)

                c_w = max(1, c_xmax - c_xmin)
                c_h = max(1, c_ymax - c_ymin)

                img = img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
                meta["crop_applied"] = True
                meta["crop_info"] = {
                    "crop_bbox": crop_bbox,
                    "crop_pixels": (c_xmin, c_ymin, c_xmax, c_ymax),
                    "crop_size": (c_w, c_h),
                    "original_size": (orig_w, orig_h)
                }

        # 2. Aspect-ratio preserving downsampling
        w, h = img.size
        if max(w, h) > max_dim:
            if w >= h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            new_w, new_h = w, h

        meta["processed_width"] = new_w
        meta["processed_height"] = new_h

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest_path, format="JPEG", quality=88, optimize=True)

    return dest_path, meta


def parse_grounding_coordinates(
    raw_text: str,
    orig_w: int,
    orig_h: int,
    crop_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Parses normalized 2D bounding boxes and points [ymin, xmin, ymax, xmax] from VLM output.
    If crop_info is present, seamlessly translates local crop coordinates back to global
    coordinates of the original high-resolution optical camera frame.
    Computes exact pixel bounds, center coordinates, and spatial relation descriptors.
    """
    results: List[Dict[str, Any]] = []
    if not raw_text:
        return results

    # 1. Attempt to parse JSON structures: [{"bbox_2d": [ymin, xmin, ymax, xmax], "label": "..."}]
    json_candidates = []
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    for block in fenced:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                json_candidates.extend(parsed)
            elif isinstance(parsed, dict):
                json_candidates.append(parsed)
        except Exception:
            pass

    if not json_candidates:
        try:
            parsed = json.loads(raw_text.strip())
            if isinstance(parsed, list):
                json_candidates.extend(parsed)
            elif isinstance(parsed, dict):
                json_candidates.append(parsed)
        except Exception:
            pass

    for item in json_candidates:
        if not isinstance(item, dict):
            continue
        label = item.get("label", item.get("name", "object"))
        box = item.get("bbox_2d") or item.get("box_2d") or item.get("bbox")
        point = item.get("point")

        if box and len(box) == 4:
            ymin, xmin, ymax, xmax = [int(v) for v in box]
            results.append(_format_grounding_result(ymin, xmin, ymax, xmax, label, orig_w, orig_h, crop_info))
        elif point and len(point) == 2:
            py, px = [int(v) for v in point]
            ymin, xmin = max(0, py - 10), max(0, px - 10)
            ymax, xmax = min(1000, py + 10), min(1000, px + 10)
            results.append(_format_grounding_result(ymin, xmin, ymax, xmax, label, orig_w, orig_h, crop_info))

    # 2. Regex fallback for bracketed coordinates: [ymin, xmin, ymax, xmax]
    if not results:
        pattern = r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]"
        matches = re.finditer(pattern, raw_text)
        for m in matches:
            ymin, xmin, ymax, xmax = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if ymin <= 1000 and xmin <= 1000 and ymax <= 1000 and xmax <= 1000:
                results.append(_format_grounding_result(ymin, xmin, ymax, xmax, "detected_item", orig_w, orig_h, crop_info))

    return results


def _format_grounding_result(
    ymin_raw: int,
    xmin_raw: int,
    ymax_raw: int,
    xmax_raw: int,
    label: str,
    orig_w: int,
    orig_h: int,
    crop_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Helper to remap, compute pixels, and determine spatial descriptors."""
    remapped = False

    if crop_info and crop_info.get("crop_bbox"):
        c_ymin, c_xmin, c_ymax, c_xmax = crop_info["crop_bbox"]
        c_h_norm = c_ymax - c_ymin
        c_w_norm = c_xmax - c_xmin

        global_ymin = int(c_ymin + (ymin_raw / 1000.0) * c_h_norm)
        global_xmin = int(c_xmin + (xmin_raw / 1000.0) * c_w_norm)
        global_ymax = int(c_ymin + (ymax_raw / 1000.0) * c_h_norm)
        global_xmax = int(c_xmin + (xmax_raw / 1000.0) * c_w_norm)
        remapped = True
    else:
        global_ymin, global_xmin, global_ymax, global_xmax = ymin_raw, xmin_raw, ymax_raw, xmax_raw

    global_ymin = max(0, min(1000, global_ymin))
    global_xmin = max(0, min(1000, global_xmin))
    global_ymax = max(0, min(1000, global_ymax))
    global_xmax = max(0, min(1000, global_xmax))

    p_ymin = int((global_ymin / 1000.0) * orig_h)
    p_xmin = int((global_xmin / 1000.0) * orig_w)
    p_ymax = int((global_ymax / 1000.0) * orig_h)
    p_xmax = int((global_xmax / 1000.0) * orig_w)

    center_norm_y = (global_ymin + global_ymax) // 2
    center_norm_x = (global_xmin + global_xmax) // 2
    center_pixel_y = (p_ymin + p_ymax) // 2
    center_pixel_x = (p_xmin + p_xmax) // 2

    v_pos = "top" if center_norm_y < 350 else ("bottom" if center_norm_y > 650 else "center")
    h_pos = "left" if center_norm_x < 350 else ("right" if center_norm_x > 650 else "center")
    if v_pos == "center" and h_pos == "center":
        spatial_desc = "center of view"
    elif v_pos == "center":
        spatial_desc = f"center-{h_pos}"
    elif h_pos == "center":
        spatial_desc = f"{v_pos}-center"
    else:
        spatial_desc = f"{v_pos}-{h_pos}"

    return {
        "label": label,
        "box_2d_norm": [global_ymin, global_xmin, global_ymax, global_xmax],
        "box_2d_pixels": [p_ymin, p_xmin, p_ymax, p_xmax],
        "center_norm": [center_norm_y, center_norm_x],
        "center_pixels": [center_pixel_y, center_pixel_x],
        "spatial_location": spatial_desc,
        "remapped_from_crop": remapped
    }

