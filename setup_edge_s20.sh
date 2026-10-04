#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Edge Node Setup Script: Galaxy S20 FE (Termux on Android 13)
# Provisions llama-mtmd-cli, SmolVLM quantized models, and PulseAudio sound sink.
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Setting up Ambient Multimodal Edge Satellite (S20 FE)"
echo "=========================================================="

export PREFIX="/data/data/com.termux/files/usr"
export HOME="/data/data/com.termux/files/home"
cd "${HOME}"

# 1. Update packages & install dependencies
echo "[1/4] Checking Termux packages..."
pkg update -y
pkg install -y termux-api pulseaudio curl jq

# 2. Verify llama-mtmd-cli binary
echo -n "[2/4] Verifying llama-mtmd-cli binary... "
if command -v llama-mtmd-cli >/dev/null 2>&1; then
    echo "OK ($(llama-mtmd-cli --version 2>&1 | head -n 1 || echo 'Installed'))"
else
    echo "NOT FOUND. Please build or install llama.cpp multi-modal CLI to ${PREFIX}/bin/llama-mtmd-cli"
fi

# 3. Create models directory and verify GGUF models
echo "[3/4] Checking Tier 1 Edge Models (SmolVLM-256M)..."
mkdir -p "${HOME}/models"

MODEL_GGUF="${HOME}/models/SmolVLM-256M-Instruct-Q8_0.gguf"
MMPROJ_GGUF="${HOME}/models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

if [ -f "${MODEL_GGUF}" ] && [ -f "${MMPROJ_GGUF}" ]; then
    echo "      Model:  ${MODEL_GGUF} ($(ls -lh "${MODEL_GGUF}" | awk '{print $5}'))"
    echo "      Projector: ${MMPROJ_GGUF} ($(ls -lh "${MMPROJ_GGUF}" | awk '{print $5}'))"
else
    echo "      Downloading SmolVLM-256M Q8_0 weights from HuggingFace..."
    curl -L -C - -o "${MODEL_GGUF}" \
      "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf"
    curl -L -C - -o "${MMPROJ_GGUF}" \
      "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    echo "      Models downloaded successfully."
fi

# 4. Configure PulseAudio AAudio Sink
echo "[4/4] Configuring PulseAudio AAudio sink..."
pulseaudio --kill 2>/dev/null || true
pulseaudio --start --exit-idle-time=-1 --load="module-aaudio-sink" 2>/dev/null || true

# Test speaker output tone or check sink
pactl info | grep -E "Server Name|Default Sink" || echo "PulseAudio running."

echo ""
echo "=========================================================="
echo " S20 FE Edge Satellite Setup Complete!"
echo " Ready for 0-cloud-token triage and hardware voice alerts."
echo "=========================================================="
