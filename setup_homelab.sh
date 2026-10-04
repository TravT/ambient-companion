#!/usr/bin/env bash
# ==============================================================================
# Homelab Ambient Companion: Master Host Environment Setup Script
# Target: Dell Latitude 7390 (Ubuntu 24.04 LTS)
# ==============================================================================
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PACKAGE_DIR}/../.." && pwd)"

echo "=========================================================="
echo " Setting up Ambient Multimodal Companion on Homelab Host"
echo "=========================================================="

# 1. Verify Python & Virtual Environment
echo -n "[1/5] Checking Python virtual environment... "
if [ ! -d "${REPO_ROOT}/.venv" ]; then
    echo "Creating virtual environment at ${REPO_ROOT}/.venv"
    python3 -m venv "${REPO_ROOT}/.venv"
fi
VENV_PY="${REPO_ROOT}/.venv/bin/python3"
VENV_PIP="${REPO_ROOT}/.venv/bin/pip"
echo "OK (${VENV_PY})"

# 2. Install Python Dependencies
echo "[2/5] Ensuring Python requirements..."
"${VENV_PIP}" install -q --upgrade pip
"${VENV_PIP}" install -q pillow requests pocket-tts
echo "      Dependencies verified."

# 3. Check ADB Gateway Connectivity to Galaxy S20 FE
echo -n "[3/5] Verifying ADB Gateway to Galaxy S20 FE (100.115.165.41)... "
if adb -H 172.17.0.2 -s 100.115.165.41:5555 get-state >/dev/null 2>&1; then
    echo "CONNECTED"
else
    echo "WARN: Device offline or ws-scrcpy gateway unreachable."
    echo "      Ensure ws-scrcpy Nomad allocation is healthy on 172.17.0.2:5037."
fi

# 4. Check Nomad llama-server Status
echo -n "[4/5] Checking Homelab llama-server (Port 8085)... "
if curl -s -f http://127.0.0.1:8085/health >/dev/null 2>&1 || curl -s -f http://127.0.0.1:8085/v1/models >/dev/null 2>&1; then
    echo "ACTIVE (200 OK)"
else
    echo "WARN: Port 8085 is not responding. Ensure Nomad job 'llama-cpp' is running:"
    echo "      nomad job run nomad_jobs/llama-cpp.nomad"
fi

# 5. Check Kyutai Pocket-TTS Voice Speaker Latents
echo -n "[5/5] Verifying Voice Speaker Latents... "
VOICE_LATENT="${REPO_ROOT}/data/media/merged/vision/benchmark/voice_profile_user_optionB_full25s.safetensors"
if [ -f "${VOICE_LATENT}" ]; then
    echo "OK ($(ls -lh "${VOICE_LATENT}" | awk '{print $5}'))"
else
    echo "WARN: Personalized speaker embedding missing at ${VOICE_LATENT}."
fi

echo ""
echo "=========================================================="
echo " Setup Complete. To test an ambient cycle:"
echo "   ${VENV_PY} ${PACKAGE_DIR}/daemon.py --source camera --query \"What is on my desk?\""
echo "=========================================================="
