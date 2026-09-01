#!/bin/bash
set -e

# ─────────────────────────────────────────────
# rhel-ai-app — Three Container Setup
# ─────────────────────────────────────────────

POD_NAME="rhel-ai-pod"
HF_TOKEN="${HF_TOKEN:-}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=<your_token>"
  exit 1
fi

echo "==> Creating pod: $POD_NAME"
podman pod create \
  --name "$POD_NAME" \
  -p 8080:8080 \
  -p 8000:8000 \
  -p 8001:8001 \
  2>/dev/null || echo "Pod already exists, continuing..."

# ─── 1. ChromaDB ─────────────────────────────
echo "==> Starting ChromaDB..."
mkdir -p ~/vectorstore

podman run -d \
  --pod "$POD_NAME" \
  --name chroma \
  -v ~/vectorstore:/chroma/chroma \
  chromadb/chroma 2>/dev/null || echo "chroma already running"

sleep 3
echo "    ChromaDB: $(curl -s http://localhost:8001/api/v2/heartbeat)"

# ─── 2. Granite / vLLM ───────────────────────
echo "==> Starting Granite vLLM server..."
mkdir -p ~/rhaiis-cache ~/rhaiis-cache/.config

podman run -d \
  --pod "$POD_NAME" \
  --name granite-server \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --shm-size=4g \
  --userns=keep-id:uid=1001 \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --env "HF_HUB_OFFLINE=0" \
  --env "VLLM_NO_USAGE_STATS=1" \
  --mount type=tmpfs,destination=/tmp,tmpfs-mode=1777 \
  -v ~/rhaiis-cache:/opt/app-root/src/.cache:Z \
  -v ~/rhaiis-cache/.config:/opt/app-root/src/.config:Z \
  registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2 \
  --model ibm-granite/granite-3.1-8b-instruct \
  --max-model-len 27680 2>/dev/null || echo "granite-server already running"

echo "    Waiting for Granite to load (this takes ~5 min first run)..."
until curl -s http://localhost:8000/v1/models | grep -q "granite"; do
  sleep 15
  echo "    Still loading..."
done
echo "    Granite: UP"

# ─── 3. rhel-ai-app ──────────────────────────
echo "==> Building rhel-ai-app image..."
podman build -t rhel-ai-app:latest .

echo "==> Starting rhel-ai-app..."
podman run -d \
  --pod "$POD_NAME" \
  --name rhel-ai-app \
  rhel-ai-app:latest 2>/dev/null || echo "rhel-ai-app already running"

sleep 3
echo "    App: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080)"

echo ""
echo "==> All containers running:"
podman pod ps
podman ps --pod

echo ""
echo "==> Open: http://localhost:8080"
echo "==> SSH tunnel from local: ssh -L 9090:localhost:8080 cloud-user@<bastion-host>"
