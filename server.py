#!/usr/bin/env python3
"""
Ambient Multimodal Desktop Companion Model Context Protocol (MCP) Server (PRJ-12 / ADR-42)
Exposes 5 typed JSON-RPC 2.0 tools with Pluggable Optical Ingestion:
  1. ambient_escalation_cycle: Autonomous 2-Pass Verification & Escalation Protocol (2P-VEP).
  2. ambient_triage_scene: Tier 1 Edge (Snapdragon 865) ultra-fast triage (0 cloud tokens).
  3. ambient_ocr_and_grounding: Tier 2 Homelab (Dell 7390) dense text & spatial bounding boxes.
  4. ambient_speak: Personalized Kyutai Pocket-TTS voice synthesis + S20 FE hardware playback.
  5. ambient_hardware_status: Thermal circuit breaker, battery, and cluster connectivity.
"""

import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import config
import daemon
import optical_ingestion

SERVER_NAME = "ambient-companion"
SERVER_VERSION = "2.1.0"
PROTOCOL_VERSION = "2024-11-05"


# Serializes everything that drives the camera / ADB session / edge CPU.
HARDWARE_LOCK = threading.Lock()
HARDWARE_TOOLS = {
    "ambient_escalation_cycle", "ambient_triage_scene",
    "ambient_ocr_and_grounding", "ambient_speak",
}


def log_debug(msg: str):
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def llama_health_url() -> str:
    """Health endpoint of the Tier 2 llama-server derived from its chat URL."""
    base = daemon.LLAMA_SERVER_URL.split("/v1/", 1)[0].rstrip("/")
    return f"{base}/health"


def readiness() -> dict:
    """Live hardware and dependency state. Never raises; unreachable parts report False/None."""
    temp_c = daemon.get_edge_temperature()
    battery = None
    adb_ok = temp_c is not None
    try:
        res = daemon.subprocess.run(
            daemon.adb_args("shell", "dumpsys battery | grep level"),
            capture_output=True, text=True, timeout=5,
        )
        for line in res.stdout.split("\n"):
            if "level" in line:
                battery = line.split(":")[-1].strip() + "%"
                adb_ok = True
    except Exception:
        pass

    llama_ok = False
    try:
        llama_ok = daemon.requests.get(llama_health_url(), timeout=3).status_code == 200
    except Exception:
        pass

    return {
        "adb": adb_ok,
        "edge_temp_c": temp_c,
        "battery_level": battery,
        "thermal_breaker": temp_c is not None and temp_c >= daemon.MAX_SAFE_TEMP_C,
        "llama_server": llama_ok,
        "tts_available": bool(daemon.POCKET_TTS_BIN),
    }


TOOLS = [
    {
        "name": "ambient_escalation_cycle",
        "description": (
            "Autonomous Two-Pass Verification & Escalation Protocol (2P-VEP). "
            "Ingests a frame from an optical source ('camera', 'droidcam', 'http://...', 'rtsp://...', or a local file), "
            "evaluates intent, attempts Tier 1 Edge triage on Snapdragon 865, executes Pass 2 self-critique, "
            "escalates to Tier 2 homelab server if ambiguous or dense OCR is required, and speaks the verified answer aloud."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Visual question or instruction (e.g. 'Is there medicine on the desk?', 'Read the dosage on the bottle')."
                },
                "source": {
                    "type": "string",
                    "default": "camera",
                    "description": "Optical source: 'camera' (S20 FE 12MP capture), 'droidcam' (port 4747 HTTP frame), 'http://...' (IP camera), 'rtsp://...', or local file path."
                },
                "language": {
                    "type": "string",
                    "enum": ["auto", "en", "pt"],
                    "default": "auto",
                    "description": "Voice output language: 'auto' (detect from query), 'en' (Option B 25s user cloned voice), or 'pt' (Rafael Portuguese studio voice)."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] normalized bounding box (0-1000) to crop into a micro-region at 100% uncompressed optical fidelity."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                },
                "play_audio": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to speak the answer aloud via S20 FE hardware speaker."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ambient_triage_scene",
        "description": (
            "Tier 1 Edge scene triage executed strictly on the Galaxy S20 FE Snapdragon 865 CPU. "
            "Fastest turnaround (<8s), zero homelab CPU load, and zero cloud tokens. "
            "CANNOT read fine text or extract dosages (use ambient_ocr_and_grounding for reading)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Macroscopic visual question (e.g. 'Is the desk empty?', 'Do you see a water glass?')."
                },
                "source": {
                    "type": "string",
                    "default": "camera",
                    "description": "Optical source: 'camera', 'droidcam', 'http://...', 'rtsp://...', or file path."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ambient_ocr_and_grounding",
        "description": (
            "Tier 2 Homelab high-precision vision inference using Qwen2.5-VL-3B on Dell Latitude 7390 CPU. "
            "Resolves dense text, medication names, active ingredients, fine print, and 2D spatial coordinates [ymin, xmin, ymax, xmax]. "
            "Supports RoI Crop-on-Demand (crop_bbox) to inspect micro-regions at 100% native optical resolution with automatic coordinate remapping."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Fine-text reading or grounding prompt (e.g. 'Read the text on the yellow label', 'Locate the keyboard with bounding boxes')."
                },
                "source": {
                    "type": "string",
                    "default": "camera",
                    "description": "Optical source: 'camera', 'droidcam', 'http://...', 'rtsp://...', or file path."
                },
                "crop_bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [ymin, xmin, ymax, xmax] normalized bounding box (0-1000) to crop into a micro-region at 100% uncompressed optical fidelity."
                },
                "roi_crop": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alias for crop_bbox [ymin, xmin, ymax, xmax]."
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 150,
                    "description": "Maximum tokens to generate."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ambient_speak",
        "description": (
            "Synthesizes natural speech using Kyutai Pocket-TTS and plays it aloud via the Galaxy S20 FE hardware stereo speaker. "
            "Uses the user's personal cloned voice for English (Option B 25s sample) or the Rafael studio voice for Brazilian Portuguese."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text message to speak aloud (concise, conversational sentences)."
                },
                "language": {
                    "type": "string",
                    "enum": ["en", "pt"],
                    "default": "en",
                    "description": "Language and voice profile: 'en' (Option B cloned voice) or 'pt' (Rafael Portuguese)."
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "ambient_hardware_status",
        "description": (
            "Returns live hardware health, battery level, thermal circuit breaker status, "
            "ADB gateway connectivity, and llama-server health for the ambient companion cluster."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


def _run_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if tool_name == "ambient_escalation_cycle":
            query = arguments.get("query") or arguments.get("prompt", "")
            source = arguments.get("source") or arguments.get("image_path", "camera")
            lang = arguments.get("language", "auto")
            lang_param = None if lang == "auto" else lang
            play_audio = arguments.get("play_audio", True)
            crop_param = arguments.get("crop_bbox") or arguments.get("roi_crop")
            
            telemetry = daemon.execute_ambient_cycle(
                query=query,
                source=source,
                lang=lang_param,
                play_audio=play_audio,
                crop_bbox=crop_param
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(telemetry, indent=2)
                    }
                ]
            }

        elif tool_name == "ambient_triage_scene":
            query = arguments.get("query") or arguments.get("prompt", "")
            source = arguments.get("source") or arguments.get("image_path", "camera")
            crop_param = arguments.get("crop_bbox") or arguments.get("roi_crop")
            
            raw_img = optical_ingestion.acquire_image(source)
            edge_img = daemon.BENCHMARK_DIR / f"mcp_edge_384px_{raw_img.stem}.jpg"
            optical_ingestion.prepare_budgeted_image(raw_img, 384, edge_img, crop_bbox=crop_param)
            
            edge_remote = daemon.safe_remote_name(f"mcp_frame_{raw_img.stem}.jpg")
            daemon.push_frame_to_edge(edge_img, edge_remote)

            triage_res = daemon.query_edge_smolvlm(edge_remote, query, max_tokens=30)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tier 1 Edge Triage Result:\n- Observation: {triage_res['text']}\n- Latency: {triage_res['duration_sec']}s\n- Source: {source} ({raw_img.name})"
                    }
                ]
            }

        elif tool_name == "ambient_ocr_and_grounding":
            query = arguments.get("query") or arguments.get("prompt", "")
            source = arguments.get("source") or arguments.get("image_path", "camera")
            max_tokens = arguments.get("max_tokens", 150)
            crop_param = arguments.get("crop_bbox") or arguments.get("roi_crop")
            
            raw_img = optical_ingestion.acquire_image(source)
            budget_max_dim = 1024 if crop_param else 512
            tier2_img = daemon.BENCHMARK_DIR / f"mcp_tier2_{budget_max_dim}px_{raw_img.stem}.jpg"
            _, meta = optical_ingestion.prepare_budgeted_image(raw_img, budget_max_dim, tier2_img, crop_bbox=crop_param)
            
            qwen_res = daemon.query_tier2_qwen(tier2_img, query, max_tokens=max_tokens)
            if qwen_res.get("status") != "success":
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Tier 2 Homelab Error: {qwen_res.get('error')}"}]
                }
                
            boxes = optical_ingestion.parse_grounding_coordinates(
                qwen_res.get("content", ""),
                orig_w=meta.get("original_width", 1000),
                orig_h=meta.get("original_height", 1000),
                crop_info=meta.get("crop_info")
            )
            
            crop_note = ""
            if meta.get("crop_applied"):
                ci = meta.get("crop_info", {})
                crop_note = f"- **RoI Zoom Active**: {ci.get('crop_bbox')} (Native pixel crop: {ci.get('crop_size')})\n"
                
            result_data = {
                "source": source,
                "image_file": str(raw_img),
                "canvas_size": [meta.get("original_width"), meta.get("original_height")],
                "roi_crop_applied": meta.get("crop_applied", False),
                "crop_info": meta.get("crop_info"),
                "matches": boxes,
                "primary_match": boxes[0] if boxes else None,
                "text": qwen_res.get("content", ""),
                "tokens": {
                    "prompt": qwen_res.get("prompt_tokens", 0),
                    "completion": qwen_res.get("completion_tokens", 0),
                    "total": qwen_res.get("total_tokens", 0)
                },
                "latency_sec": qwen_res.get("duration_sec", 0),
                "endpoint": qwen_res.get("endpoint", "")
            }
            
            out_text = (
                f"### Ambient OCR & Visual Grounding (Tier 2)\n"
                f"- **Source**: {source} ({meta.get('original_width')}x{meta.get('original_height')})\n"
                f"{crop_note}"
                f"- **Processed Canvas**: {meta.get('processed_width')}x{meta.get('processed_height')}\n"
                f"- **Response**:\n{qwen_res.get('content')}\n\n"
                f"```json\n{json.dumps(result_data, indent=2)}\n```\n\n"
                f"- **Latency**: {qwen_res.get('duration_sec')}s | **Tokens**: {qwen_res.get('prompt_tokens', 0)} in, {qwen_res.get('completion_tokens', 0)} out"
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": out_text
                    }
                ]
            }

        elif tool_name == "ambient_speak":
            text = arguments.get("text", "")
            lang = arguments.get("language", "en")
            
            timestamp = int(time.time())
            wav_path = daemon.BENCHMARK_DIR / f"mcp_speech_{timestamp}.wav"
            tts_res = daemon.synthesize_speech(text, lang, wav_path)
            
            played = False
            if tts_res.get("status") == "success":
                played = daemon.play_audio_on_edge(wav_path)
                
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Speech Playback Status:\n- Text: \"{text}\"\n- Language: {lang}\n- Synthesized in: {tts_res.get('duration_sec', 0)}s\n- S20 FE Speaker Playback: {'Success' if played else 'Failed'}"
                    }
                ]
            }

        elif tool_name == "ambient_hardware_status":
            r = readiness()
            temp = f"{r['edge_temp_c']:.1f}°C" if r["edge_temp_c"] is not None else "UNKNOWN (edge unreachable)"
            breaker = "YES - OVERHEAT" if r["thermal_breaker"] else ("UNKNOWN" if r["edge_temp_c"] is None else "NO (Normal)")
            status_report = (
                f"Ambient Companion Hardware Status:\n"
                f"- S20 FE Battery Temp: {temp} (Circuit breaker limit: {daemon.MAX_SAFE_TEMP_C}°C)\n"
                f"- S20 FE Battery Level: {r['battery_level'] or 'Unknown'}\n"
                f"- Thermal Breaker Engaged: {breaker}\n"
                f"- ADB Gateway: {'CONNECTED' if r['adb'] else 'UNREACHABLE'} "
                f"({daemon.CONTAINER_IP} -> {daemon.DEVICE_TARGET})\n"
                f"- Homelab llama-server: {'HEALTHY (200 OK)' if r['llama_server'] else 'UNHEALTHY / OFFLINE'}\n"
                f"- Voice (pocket-tts): {'available' if r['tts_available'] else 'not installed in this runtime'}"
            )
            return {
                "content": [{"type": "text", "text": status_report}]
            }

        else:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}

    except Exception as e:
        return {"isError": True, "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}]}


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Run a tool; camera/ADB tools are serialized and answer "busy" instead of queueing forever."""
    if tool_name not in HARDWARE_TOOLS:
        return _run_tool(tool_name, arguments)
    if not HARDWARE_LOCK.acquire(timeout=config.HARDWARE_LOCK_TIMEOUT_SEC):
        return {"isError": True, "content": [{
            "type": "text",
            "text": "Companion hardware is busy with another request; retry shortly.",
        }]}
    try:
        return _run_tool(tool_name, arguments)
    finally:
        HARDWARE_LOCK.release()


def dispatch(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC request. Returns the response, or None for notifications."""
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
            }
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        result = handle_tool_call(params.get("name"), params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    if method == "notifications/initialized":
        return None
    if req_id is not None:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return None


def run_stdio_server():
    log_debug(f"Starting {SERVER_NAME} v{SERVER_VERSION} stdio JSON-RPC loop")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as err:
            log_debug(f"JSON parse error: {err}")
            continue
        resp = dispatch(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h", "help"):
        print(f"Ambient Companion MCP Server v{SERVER_VERSION} (stdio default; `serve` = HTTP)")
        print("Tools: ambient_escalation_cycle, ambient_triage_scene, ambient_ocr_and_grounding, ambient_speak, ambient_hardware_status")
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import http_service
        http_service.serve()
    else:
        run_stdio_server()
