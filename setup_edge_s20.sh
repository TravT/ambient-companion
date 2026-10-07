#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Edge Node Setup Script: Galaxy S20 FE (Termux on Android 13)
# Provisions llama-mtmd-cli, SmolVLM quantized models, and the PulseAudio sink.
# Idempotent: re-running verifies each model against Hugging Face's published
# SHA-256 and only downloads what is missing or corrupt.
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " Setting up Ambient Multimodal Edge Satellite (S20 FE)"
echo "=========================================================="

export PREFIX="/data/data/com.termux/files/usr"
export HOME="/data/data/com.termux/files/home"
cd "${HOME}"

HF_BASE="https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main"

# 1. Packages (pkg install is a no-op for what is already installed)
echo "[1/5] Checking Termux packages..."
pkg update -y
pkg install -y termux-api pulseaudio curl jq coreutils

# 2. llama-mtmd-cli binary
echo -n "[2/5] Verifying llama-mtmd-cli binary... "
if command -v llama-mtmd-cli >/dev/null 2>&1; then
    echo "OK ($(llama-mtmd-cli --version 2>&1 | head -n 1 || echo 'Installed'))"
else
    echo "NOT FOUND. Please build or install llama.cpp multi-modal CLI to ${PREFIX}/bin/llama-mtmd-cli"
fi

# 3. Models, verified against the upstream SHA-256 (x-linked-etag of the LFS file)
fetch_verified() {
    local name="$1" dest="${HOME}/models/$1" want have
    want="$(curl -sIL "${HF_BASE}/${name}" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-etag"{gsub(/"/,"",$2); print $2}' | tail -n 1)"
    if [ -f "$dest" ] && [ -n "$want" ]; then
        have="$(sha256sum "$dest" | cut -d' ' -f1)"
        if [ "$have" = "$want" ]; then
            echo "      ${name}: OK (sha256 verified, $(du -h "$dest" | cut -f1))"
            return 0
        fi
        echo "      ${name}: checksum mismatch, re-downloading"
    elif [ -f "$dest" ]; then
        echo "      ${name}: present, upstream checksum unavailable (offline?), keeping it"
        return 0
    fi
    curl -L -C - -o "$dest" "${HF_BASE}/${name}"
    if [ -n "$want" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" != "$want" ]; then
        echo "      ${name}: ERROR checksum still wrong after download" >&2
        return 1
    fi
    echo "      ${name}: downloaded"
}

echo "[3/5] Checking Tier 1 Edge Models (SmolVLM-256M)..."
mkdir -p "${HOME}/models"
fetch_verified "SmolVLM-256M-Instruct-Q8_0.gguf"
fetch_verified "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"

# 4. PulseAudio AAudio sink (only restart it when the sink is missing)
echo "[4/5] Configuring PulseAudio AAudio sink..."
if ! pactl info >/dev/null 2>&1; then
    pulseaudio --start --exit-idle-time=-1 --load="module-aaudio-sink" 2>/dev/null || true
fi
pactl info | grep -E "Server Name|Default Sink" || echo "PulseAudio running."

# 5. Voice: pocket-tts runs on the phone (PRJ-12 Phase 10). Idempotent: skipped when the CLI already works.
echo "[5/5] Voice (pocket-tts on the S20)..."
export TMPDIR="${PREFIX}/tmp" CARGO_BUILD_JOBS=2 MAKEFLAGS=-j2 ANDROID_API_LEVEL=24
mkdir -p "${TMPDIR}" "${HOME}/voices"
if ! HF_HUB_DISABLE_XET=1 pocket-tts generate --help >/dev/null 2>&1; then
    echo "      Installing python + torch stack and building pocket-tts (Rust extensions compile: ~20-30 min, keeps the phone warm)..."
    pkg install -y python python-numpy python-scipy python-torch rust maturin libsndfile cmake git patchelf
    pip install sentencepiece
    pip install pocket-tts==3.3.0 soundfile pyyaml
    # Rust extensions built on Android may lack the libpython link (ImportError: PyBaseObject_Type).
    PYLIB="$(basename "$(ls "${PREFIX}"/lib/libpython3*.so | head -n 1)")"
    SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    for so in "${SITE}"/tokenizers/tokenizers*.so "${SITE}"/safetensors/_safetensors_rust*.so "${SITE}"/pydantic_core/_pydantic_core*.so; do
        [ -f "$so" ] || continue
        readelf -d "$so" | grep -q "${PYLIB}" || patchelf --add-needed "${PYLIB}" "$so"
    done
fi
if HF_HUB_DISABLE_XET=1 pocket-tts generate --help >/dev/null 2>&1; then
    echo "      pocket-tts: OK ($(command -v pocket-tts))"
else
    echo "      WARN: pocket-tts still fails to start; check: pocket-tts generate --help"
fi
VOICE="${HOME}/voices/voice_profile_user_optionB_full25s.safetensors"
if [ -f "$VOICE" ]; then
    echo "      English voice profile: OK ($(du -h "$VOICE" | cut -f1))"
else
    echo "      WARN: English voice profile missing at ${VOICE}."
    echo "      Re-create on the Dell: HF_TOKEN=<vault_hf_token> pocket-tts export-voice <recording.wav> <out.safetensors>, then adb push it there."
fi

echo ""
echo "=========================================================="
echo " S20 FE Edge Satellite Setup Complete!"
echo " Ready for 0-cloud-token triage and hardware voice alerts."
echo "=========================================================="
