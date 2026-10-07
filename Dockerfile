# ==============================================================================
# Ambient Multimodal Desktop Companion Container (PRJ-12 / ADR-42)
# Thin orchestrator: HTTP + MCP service that drives the S20 FE through an ADB
# server and escalates to the homelab llama-server. Runs as a non-root user and
# needs no raw_exec. Voice synthesis (pocket-tts) is optional: --build-arg WITH_TTS=1
# ==============================================================================
FROM python:3.12.8-slim-bookworm

ARG WITH_TTS=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# adb: edge control client. ffmpeg: RTSP keyframe extraction.
RUN apt-get update && apt-get install -y --no-install-recommends \
        adb \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin ambient

WORKDIR /app

COPY requirements.txt requirements-tts.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_TTS" = "1" ]; then pip install --no-cache-dir -r requirements-tts.txt; fi

COPY --chown=ambient:ambient . .

# Scratch frames/audio live in the container (purged by the retention thread),
# never on the MergerFS media tier. Only the dropzone is readable as a file source.
ENV PYTHONPATH=/app \
    AMBIENT_DATA_DIR=/data \
    AMBIENT_BENCHMARK_DIR=/tmp/ambient/benchmark \
    AMBIENT_VALIDATION_DIR=/tmp/ambient/validation \
    AMBIENT_ALLOWED_DIRS=/data/dropzone/files \
    AMBIENT_PORT=8089

RUN mkdir -p /tmp/ambient && chown -R ambient:ambient /tmp/ambient

USER ambient

EXPOSE 8089

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python3", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"AMBIENT_PORT\"]}/health', timeout=4)"]

ENTRYPOINT ["python3", "server.py", "serve"]
