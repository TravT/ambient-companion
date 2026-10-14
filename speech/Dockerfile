# ==============================================================================
# ambient-speech (PRJ-12): pocket-tts voices + faster-whisper for the companion, on the Dell CPU.
# Models are NOT in the image: they are mounted read-only at /models and loaded offline.
# ==============================================================================
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin speech

WORKDIR /app

COPY requirements.txt ./
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY --chown=speech:speech speech_service.py ./

# Offline: never reach the Hub from the container. HF_HOME holds the pocket-tts files (hub cache layout).
ENV HF_HOME=/models/hf \
    HF_HUB_OFFLINE=1 \
    SPEECH_WHISPER_DIR=/models/faster-whisper-small \
    SPEECH_HOST=127.0.0.1 \
    SPEECH_PORT=8091

USER speech

EXPOSE 8091

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python3", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"SPEECH_PORT\"]}/health', timeout=4)"]

ENTRYPOINT ["python3", "speech_service.py"]
