FROM python:3.12-slim

ARG EXTRAS="server,ollama,openai"
ARG WITH_LLAMA_CPP=false

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface

# Install curl for healthcheck, and build tools conditionally if llama-cpp is enabled
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && if [ "$WITH_LLAMA_CPP" = "true" ] || [ "$WITH_LLAMA_CPP" = "1" ] || echo "$EXTRAS" | grep -q "llama-cpp"; then \
        apt-get install -y --no-install-recommends build-essential cmake; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and application directories
RUN groupadd -g 1000 omnillm && \
    useradd -u 1000 -g omnillm -m -s /bin/bash omnillm && \
    mkdir -p /config /data/huggingface && \
    chown -R omnillm:omnillm /config /data

WORKDIR /app

# Copy project source and install with requested extras
COPY . /app

RUN if [ "$WITH_LLAMA_CPP" = "true" ] || [ "$WITH_LLAMA_CPP" = "1" ]; then \
        case "$EXTRAS" in \
            *llama-cpp*) ;; \
            *) EXTRAS="${EXTRAS},llama-cpp" ;; \
        esac; \
    fi && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[${EXTRAS}]"

WORKDIR /config

USER omnillm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["omnillm", "serve", "--host", "0.0.0.0", "--port", "8000"]
