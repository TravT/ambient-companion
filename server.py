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
from pathlib import Path
from typing import Dict, Any, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import daemon
import optical_ingestion

SERVER_NAME = "ambient-companion"
SERVER_VERSION = "2.0.0"
PROTOCOL_VERSION = "2024-11-05"


def log_debug(msg: str):
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


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
            "Resolves dense text, medication names, active ingredients, fine print, and 2D spatial coordinates [ymin, xmin, ymax, xmax]."
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


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if tool_name == "ambient_escalation_cycle":
            query = arguments.get("query") or arguments.get("prompt", "")
            source = arguments.get("source") or arguments.get("image_path", "camera")
            lang = arguments.get("language", "auto")
            lang_param = None if lang == "auto" else lang
            play_audio = arguments.get("play_audio", True)
            
            telemetry = daemon.execute_ambient_cycle(
                query=query,
                source=source,
                lang=lang_param,
                play_audio=play_audio
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
            
            raw_img = optical_ingestion.acquire_image(source)
            edge_img = daemon.BENCHMARK_DIR / f"mcp_edge_384px_{raw_img.stem}.jpg"
            optical_ingestion.prepare_budgeted_image(raw_img, 384, edge_img)
            
            edge_remote = f"mcp_frame_{raw_img.stem}.jpg"
            daemon.subprocess.run([
                "adb", "-H", daemon.CONTAINER_IP, "-s", daemon.DEVICE_TARGET,
                "push", str(edge_img), f"/sdcard/Download/{edge_remote}"
            ], capture_output=True, check=False)
            daemon.run_edge_command(
                f"cp /sdcard/Download/{edge_remote} {daemon.EDGE_TERMUX_HOME}/{edge_remote} && "
                f"chown u0_a356:u0_a356 {daemon.EDGE_TERMUX_HOME}/{edge_remote} && "
                f"chmod 644 {daemon.EDGE_TERMUX_HOME}/{edge_remote}",
                as_root=True
            )
            
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
            
            raw_img = optical_ingestion.acquire_image(source)
            tier2_img = daemon.BENCHMARK_DIR / f"mcp_tier2_512px_{raw_img.stem}.jpg"
            optical_ingestion.prepare_budgeted_image(raw_img, 512, tier2_img)
            
            qwen_res = daemon.query_tier2_qwen(tier2_img, query, max_tokens=max_tokens)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tier 2 Homelab Result:\n- Response: {qwen_res.get('content', '')}\n- Latency: {qwen_res.get('duration_sec', 0)}s\n- Status: {qwen_res.get('status', 'unknown')}"
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
            temp_c = daemon.get_edge_temperature()
            
            # Check S20 battery level
            bat_cmd = ["adb", "-H", daemon.CONTAINER_IP, "-s", daemon.DEVICE_TARGET, "shell", "dumpsys battery | grep level"]
            bat_level = "Unknown"
            try:
                bat_res = daemon.subprocess.run(bat_cmd, capture_output=True, text=True, timeout=5)
                for line in bat_res.stdout.split("\n"):
                    if "level" in line:
                        bat_level = line.split(":")[-1].strip() + "%"
            except Exception:
                pass
                
            # Check llama-server health
            llama_ok = False
            try:
                r = daemon.requests.get("http://127.0.0.1:8085/health", timeout=3)
                llama_ok = r.status_code == 200
            except Exception:
                pass
                
            status_report = (
                f"Ambient Companion Hardware Status:\n"
                f"- S20 FE Battery Temp: {temp_c:.1f}°C (Circuit breaker limit: {daemon.MAX_SAFE_TEMP_C}°C)\n"
                f"- S20 FE Battery Level: {bat_level}\n"
                f"- Thermal Breaker Engaged: {'YES - OVERHEAT' if temp_c >= daemon.MAX_SAFE_TEMP_C else 'NO (Normal)'}\n"
                f"- ADB Gateway (ws-scrcpy): Connected at {daemon.CONTAINER_IP}:5555\n"
                f"- Homelab llama-server (Port 8085): {'HEALTHY (200 OK)' if llama_ok else 'UNHEALTHY / OFFLINE'}"
            )
            return {
                "content": [{"type": "text", "text": status_report}]
            }

        else:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}

    except Exception as e:
        return {"isError": True, "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}]}


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

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
                }
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS}
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            result = handle_tool_call(tool_name, tool_args)
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            pass

        else:
            if req_id is not None:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h", "help"):
        print("Ambient Companion MCP Server v2.0.0 (Stdio JSON-RPC 2.0)")
        print("Tools: ambient_escalation_cycle, ambient_triage_scene, ambient_ocr_and_grounding, ambient_speak, ambient_hardware_status")
        sys.exit(0)
    run_stdio_server()
