#!/usr/bin/env bash
# ==============================================================================
# Homelab Ambient Companion: master host (Dell Latitude 7390) setup / preflight
#
# Default: read-only preflight for the containerized service. Changes nothing,
# safe to re-run, and prints the exact GitOps deploy command.
#   --dev        also create the repo .venv and install core deps (host-native CLI)
#   --with-tts   with --dev, also install pocket-tts (pulls in torch, multi-GB)
#
# No secrets live here: the API token is injected at deploy time from the vault
# (python3 scripts/get_secret.py) by the Ansible/Nomad deploy.
# ==============================================================================
set -euo pipefail

DEV=0
WITH_TTS=0
for arg in "$@"; do
    case "$arg" in
        --dev) DEV=1 ;;
        --with-tts) WITH_TTS=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"
ADB_HOST="${ADB_GATEWAY_HOST:-127.0.0.1}"
DEVICE="${S20_DEVICE_TARGET:-100.115.165.41:5555}"
LLAMA_HEALTH="${LLAMA_HEALTH_URL:-http://127.0.0.1:8085/health}"
WARNINGS=0

step() { echo -n "$1 "; }
ok()   { echo "OK${1:+ ($1)}"; }
warn() { echo "WARN: $1"; WARNINGS=$((WARNINGS + 1)); }

echo "=========================================================="
echo " Ambient Multimodal Companion: Homelab host preflight"
echo "=========================================================="

step "[1/5] Tooling (docker, nomad, adb)..."
missing=""
for tool in docker nomad adb; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
done
if [ -z "$missing" ]; then ok; else warn "missing:${missing}"; fi

step "[2/5] ADB gateway ${ADB_HOST}:5037 -> ${DEVICE}..."
if [ "$(adb -H "$ADB_HOST" -s "$DEVICE" get-state 2>/dev/null || true)" = "device" ]; then
    ok "device"
else
    warn "S20 FE offline or the custom-ws-scrcpy job is not running (nomad job status custom-ws-scrcpy)"
fi

step "[3/5] Tier 2 llama-server (${LLAMA_HEALTH})..."
if curl -fsS -m 3 "$LLAMA_HEALTH" >/dev/null 2>&1; then
    ok "200"
else
    warn "down. llama-cpp is on-demand: nomad job run -var count=1 nomad_jobs/llama-cpp.nomad (escalations fail until then)"
fi

step "[4/5] Ambient service (http://127.0.0.1:8089/health)..."
if curl -fsS -m 3 http://127.0.0.1:8089/health >/dev/null 2>&1; then
    ok "running"
else
    echo "not running (deploy below)"
fi

step "[5/5] Voice on the S20 (pocket-tts + English profile)..."
TERMUX_HOME=/data/data/com.termux/files/home
if adb -H "$ADB_HOST" -s "$DEVICE" shell "su -c 'test -x /data/data/com.termux/files/usr/bin/pocket-tts && test -s $TERMUX_HOME/voices/voice_profile_user_optionB_full25s.safetensors'" >/dev/null 2>&1; then
    ok "pocket-tts and profile present"
else
    warn "pocket-tts or the English profile is missing on the S20 (run setup_edge_s20.sh in Termux; export the profile with: HF_TOKEN=<vault_hf_token> pocket-tts export-voice <recording.wav> <out.safetensors>, then adb push it to ~/voices/)"
fi

if [ "$DEV" = "1" ]; then
    echo ""
    echo "[dev] Host-native CLI environment in ${REPO_ROOT}/.venv"
    [ -d "${REPO_ROOT}/.venv" ] || python3 -m venv "${REPO_ROOT}/.venv"
    "${REPO_ROOT}/.venv/bin/pip" install -q -r "${PACKAGE_DIR}/requirements.txt"
    if [ "$WITH_TTS" = "1" ]; then
        "${REPO_ROOT}/.venv/bin/pip" install -q -r "${PACKAGE_DIR}/requirements-tts.txt"
    fi
    echo "[dev] Done. Try: ${REPO_ROOT}/.venv/bin/python ${PACKAGE_DIR}/daemon.py --source camera --query \"What is on my desk?\""
fi

echo ""
echo "=========================================================="
echo " Preflight finished with ${WARNINGS} warning(s)."
echo " Deploy (GitOps only, never ad-hoc docker run):"
echo "   ansible-playbook -i ansible/inventory.ini ansible/site.yml --tags docker --vault-password-file ansible/.vault_pass"
echo " Then verify: nomad job status ambient-companion && curl -s 127.0.0.1:8089/ready"
echo "=========================================================="
