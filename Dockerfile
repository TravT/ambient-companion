# ==============================================================================
# Ambient Multimodal Desktop Companion Container
# Multi-arch (amd64/arm64) production image for GHCR release pinning
# ==============================================================================
FROM python:3.12.8-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime tools (curl for healthchecks, adb for S20 FE mobile fleet control)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    adb \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Default environment configurations
ENV ADB_GATEWAY_HOST=172.17.0.2 \
    S20_DEVICE_TARGET=100.115.165.41:5555 \
    LLAMA_SERVER_URL=http://127.0.0.1:8085/v1/chat/completions \
    PYTHONPATH=/app

EXPOSE 8089

ENTRYPOINT ["python3", "daemon.py"]
CMD ["--query", "Homelab ambient companion ready.", "--no-audio"]
