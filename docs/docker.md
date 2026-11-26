# Docker Deployment

Omni-Local-LLM provides a Docker image and Docker Compose configuration for running the OpenAI-compatible API server in a containerized environment.

## Quick Start with Docker Compose

The included `docker-compose.yml` sets up both an Ollama service and OmniLLM:

```bash
# Start both Ollama and OmniLLM
docker compose up -d

# Pull a model in Ollama
docker compose exec ollama ollama pull llama3.2:3b

# Query OmniLLM's OpenAI-compatible endpoint
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3.2:3b",
    "messages": [{"role": "user", "content": "Explain relativity in one sentence."}]
  }'
```

In this setup:
- The `ollama` service persists downloaded models in the `ollama_data` volume.
- OmniLLM connects to Ollama via `OLLAMA_HOST=http://ollama:11434`.
- The API is bound exclusively to `127.0.0.1:8000` on the host.

## Building the Docker Image

### Default Build

The root `Dockerfile` is based on `python:3.12-slim` and installs the `server`, `ollama`, and `openai` extras by default:

```bash
docker build -t omnillm .
```

### Optional `llama-cpp` Backend

To include the in-process `llama.cpp` backend and Hugging Face model downloads, enable `llama-cpp` via the `WITH_LLAMA_CPP` build argument (or specify `EXTRAS`):

```bash
# Build with llama.cpp support
docker build --build-arg WITH_LLAMA_CPP=true -t omnillm:llama-cpp .

# Alternatively, pass the EXTRAS list explicitly
docker build --build-arg EXTRAS="server,ollama,openai,llama-cpp" -t omnillm:llama-cpp .
```

When `WITH_LLAMA_CPP` is enabled, build tools (`build-essential`, `cmake`) are automatically installed during the build phase.

## Running the Container

### Standalone `docker run`

Run the container in the background, publishing port 8000 to the host loopback:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  omnillm
```

The container runs by default as the non-root user `omnillm` (UID 1000) and launches `omnillm serve --host 0.0.0.0 --port 8000`.

### API Key and Model Downloads

Pass `OMNILLM_API_KEY` to require a bearer token on every endpoint except `/health` (so the container health check keeps working):

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  -e OMNILLM_API_KEY="$(openssl rand -hex 32)" \
  omnillm
```

API requests never download models by default: pull them into Ollama first (as in the quick start), pre-download GGUF files into the `/data/huggingface` volume, or set `OMNILLM_ALLOW_DOWNLOADS=1`. See [Access control](server.md#access-control).

### Health Check

The server exposes `GET /health`, which returns `{"status":"ok"}` immediately without querying any upstream model backends.

The `Dockerfile` defines a container health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

You can inspect the health status of a running container with:

```bash
docker inspect --format='{{json .State.Health}}' omnillm
# Or with docker compose:
docker compose ps
```

## Configuration Mounting

The container defines `WORKDIR /config` at runtime and does **not** set the `OMNILLM_CONFIG` environment variable. This allows OmniLLM's standard configuration discovery to automatically pick up `./omnillm.toml` (which points to `/config/omnillm.toml`).

### Mounting a Single `omnillm.toml`

Mount your configuration file directly into `/config/omnillm.toml`:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  -v $(pwd)/omnillm.toml:/config/omnillm.toml:ro \
  omnillm
```

### Mounting a Config Directory

Alternatively, mount an entire directory containing `omnillm.toml`:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  -v $(pwd)/my-config:/config:ro \
  omnillm
```

In `docker-compose.yml`, uncomment the volume mount under `omnillm`:

```yaml
    volumes:
      - ./omnillm.toml:/config/omnillm.toml:ro
```

### Model Cache Persistence

The image sets `HF_HOME=/data/huggingface` so Hugging Face downloads (such as GGUF models downloaded with `omnillm fit --pull`) are stored in `/data/huggingface`. To persist downloaded models across container restarts:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  -v omnillm_hf_cache:/data/huggingface \
  omnillm
```

## Connecting to Host Ollama

If you already have Ollama installed and running on your host machine, you can point containerized OmniLLM to it:

### Linux

On Linux, use `--add-host` to resolve the host machine:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  omnillm
```

### macOS and Windows

On Docker Desktop for macOS and Windows, `host.docker.internal` is already configured:

```bash
docker run -d \
  --name omnillm \
  -p 127.0.0.1:8000:8000 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  omnillm
```

> [!NOTE]
> Ensure that Ollama on the host is listening on all network interfaces (`0.0.0.0:11434`) so that requests from Docker containers can reach it. Set `OLLAMA_HOST=0.0.0.0:11434` in your host's Ollama service configuration.

> [!WARNING]
> Ollama has no authentication, so listening on `0.0.0.0` also exposes it to your local network. Block port 11434 from other machines with your firewall, or bind Ollama to the Docker bridge address instead (usually `OLLAMA_HOST=172.17.0.1:11434`; check with `ip addr show docker0`).

## Security Considerations

1. **Non-Root Execution**: The container runs under an unprivileged user (`omnillm`, UID 1000) with minimal permissions.
2. **Host Header Validation (DNS-Rebinding Protection)**: OmniLLM strictly validates `Host` headers. By default, only loopback addresses (`localhost`, `127.0.0.1`, `::1`) and the container hostname are accepted.
   - Do **not** use wildcard (`*`) allowed hosts or disable host checks on untrusted networks.
   - Always bind port forwarding to loopback (`127.0.0.1:8000:8000`) rather than `0.0.0.0:8000` to prevent exposure to the local network.
   - If placing the container behind a reverse proxy (e.g., Nginx, Caddy, or Traefik), configure the exact hostnames via `OMNILLM_ALLOWED_HOSTS` or pass `--allowed-host <domain>`.
3. **Safe Media Handling**: OmniLLM rejects local file paths and remote URLs in vision requests to ensure clients cannot force the server to read host files or make arbitrary network requests.
4. **API Key**: set `OMNILLM_API_KEY` before publishing the port beyond `127.0.0.1`. Without it, anyone who can reach the port can use your models. The key isn't encrypted in transit, so for remote access also put the container behind a reverse proxy that terminates TLS (such as Nginx, Caddy, or Traefik), and list its host names in `OMNILLM_ALLOWED_HOSTS` or with `--allowed-host`.
5. **No Downloads from API Requests**: clients can only use models already on the server, so they can't fill the disk or make the server fetch and load arbitrary model files. Only set `OMNILLM_ALLOW_DOWNLOADS=1` if you trust every client.

## Future GPU Support

Currently, containerized inference focuses on CPU execution and remote backends (such as Ollama or OpenAI-compatible servers).

For future GPU support with in-process backends:
- **NVIDIA GPUs**: Pass `--gpus all` using the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **`llama-cpp-python` with CUDA**: Future Dockerfile targets can configure `CMAKE_ARGS="-DGGML_CUDA=on"` during installation to enable GPU-accelerated GGUF inference directly inside the container.
- **Docker Compose GPU Passthrough**: In `docker-compose.yml`, GPU access can be assigned to services using `deploy.resources.reservations.devices`.

---

[← Back to README](../README.md) · [API Server Documentation](server.md)
