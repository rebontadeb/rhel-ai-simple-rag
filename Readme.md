# rhel-ai-app — Student Lab Guide

> Build a fully local RAG chatbot on RHEL AI using Granite LLM, ChromaDB, and FastAPI.
> No cloud. No OpenAI. No HuggingFace token. Everything runs on-box.

**GitHub:** https://github.com/rebontadeb/rhel-ai-simple-rag

---

## What You Will Build

A web application where you:
1. Upload a PDF document
2. Ask questions about it in a chat UI
3. Granite 3.1 8B answers using only your document

**Stack:**

| Layer | Tool |
|---|---|
| LLM | IBM Granite 3.1 8B — pre-installed in RHEL AI |
| Inference Server | ilab model serve (vLLM backend) |
| Embedding | BAAI/bge-small-en-v1.5 via fastembed (ONNX, CPU, ~33MB) |
| Vector Store | ChromaDB 1.0.0 |
| PDF Extraction | pdfminer.six |
| API | FastAPI + uvicorn |
| UI | HTMX + SSE streaming |
| Containers | Podman |

---

## Step 1 — Verify Environment

```bash
cat /etc/redhat-release
lspci | grep -E -i "vga|3d|display|nvidia|amd|radeon"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ilab --version
podman --version
python3 --version
ls ~/.cache/instructlab/models/
```

Expected:
```
Red Hat Enterprise Linux release 9.4 (Plow)
NVIDIA L4, 23034 MiB
ilab, version 0.26.1
podman version 4.9.4-rhel
Python 3.9.18
granite-3.1-8b-lab-v2    granite-3.1-8b-starter-v2 ...
```

---

## Step 2 — Login to Red Hat Registry

```bash
podman login registry.redhat.io
```

Expected:
```
Login Succeeded!
```

---

## Step 3 — Init ilab

```bash
ilab config init
```

When prompted:
- Vendor → `1` (NVIDIA)
- Profile → `0` (NO SYSTEM PROFILE) — single L4 not in list

---

## Step 4 — Fix ilab Config for Single L4 GPU

Default config sets `tensor-parallel-size 4` — fails on 1 GPU. Default `max_model_len 131072` exceeds L4 KV cache. Run this patch:

```bash
python3 -c "
import yaml
with open('/var/home/cloud-user/.config/instructlab/config.yaml') as f:
    config = yaml.safe_load(f)
config['serve']['model_path'] = '/var/home/cloud-user/.cache/instructlab/models/granite-3.1-8b-starter-v2'
config['serve']['vllm']['gpus'] = 1
config['serve']['vllm']['vllm_args'] = [
    '--max-model-len', '11680',
    '--dtype', 'bfloat16'
]
with open('/var/home/cloud-user/.config/instructlab/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Done')
"
```

Verify:

```bash
grep -A5 "vllm:" ~/.config/instructlab/config.yaml | grep -E "gpus|max-model"
```

---

## Step 5 — Serve Granite (keep terminal open)

```bash
ilab model serve --gpus 1
```

Wait for:
```
INFO: Application startup complete.
```

Takes ~2 min (model cached), ~12 min cold start.

In another terminal verify:

```bash
curl http://localhost:8000/v1/models
```

> ⚠ Do not close this terminal — keep ilab running. Use `nohup` for background:
> ```bash
> nohup ilab model serve --gpus 1 > ~/ilab-serve.log 2>&1 &
> echo $! > ~/ilab-serve.pid
> ```

---

## Step 6 — Start ChromaDB

```bash
mkdir -p ~/vectorstore

podman run -d \
  --name chroma \
  -p 8001:8000 \
  -v ~/vectorstore:/chroma/chroma \
  chromadb/chroma

curl http://localhost:8001/api/v2/heartbeat
```

Expected:
```json
{"nanosecond heartbeat": ...}
```

---

## Step 7 — Clone the App

```bash
cd ~
git clone https://github.com/rebontadeb/rhel-ai-simple-rag.git
cd rhel-ai-simple-rag
```

Or create files manually — see **Application Code** section below.

---

## Step 8 — Fix Ownership for Volume Mounts

Container runs as uid 1001. Host dirs must match:

```bash
mkdir -p ~/rhel-ai-cache/fastembed ~/rhel-ai-data/docs
sudo chmod -R 777 ~/rhel-ai-cache/ ~/rhel-ai-data/
```

---

## Step 9 — Build App Container

```bash
cd ~/rhel-ai-simple-rag
podman build -t rhel-ai-app:latest .
```

Build takes 3-5 min first time (downloads pip packages).

---

## Step 10 — Run App Container

```bash
podman run -d \
  --name rhel-ai-app \
  --network=host \
  -v ~/rhel-ai-cache/fastembed:/opt/app-root/src/.cache:Z \
  -v ~/rhel-ai-data/docs:/opt/app-root/src/data/docs:Z \
  rhel-ai-app:latest
```

Verify:

```bash
podman logs -f rhel-ai-app
```

Expected:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8080
```

> ⚠ `--network=host` required — app reaches ilab on `localhost:8000` and ChromaDB on `localhost:8001`
> ⚠ First run downloads fastembed model (~33MB) to mounted cache dir

---

## Step 11 — Access via SSH Tunnel

Only SSH port is open externally. From your **local machine**:

```bash
ssh -L 9090:localhost:8080 cloud-user@<bastion-hostname>
```

Open browser: `http://localhost:9090`

---

## Step 12 — Use the App

1. Click folder icon in sidebar → select PDF
2. Click **Upload** — wait for `✓ N chunks ingested`
3. Type question in chat → Enter
4. Granite answers using only your document

---

## Step 13 — Stop Everything

```bash
podman stop rhel-ai-app chroma
podman rm rhel-ai-app chroma

# stop ilab
kill $(cat ~/ilab-serve.pid)
```

---

## Application Code

### `config.py`

```python
VLLM_URL = "http://localhost:8000/v1"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 33MB ONNX — no torch, no GPU
LLM_MODEL = "granite-3.1-8b-starter-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 4
DOCS_DIR = "data/docs"
CHROMA_COLLECTION = "rhel-ai-docs"
```

---

### `ingest.py`

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
import re
import uuid
import chromadb
from fastembed import TextEmbedding
from pdfminer.high_level import extract_text as pdfminer_extract
from config import (
    CHROMA_HOST, CHROMA_PORT, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION
)

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(
            model_name=EMBED_MODEL,
            cache_dir="/opt/app-root/src/.cache"
        )
    return _embed_model

def get_chroma_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(CHROMA_COLLECTION)

def clean_text(text: str) -> str:
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
    text = re.sub(r'-\n([a-z])', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = pdfminer_extract(file_path)
        return clean_text(text or "")
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]

def ingest_documents(file_path: str) -> dict:
    text = extract_text(file_path)
    if not text.strip():
        return {"status": "error", "message": "No text extracted"}
    chunks = chunk_text(text)
    model = get_embed_model()
    embeddings = [e.tolist() for e in model.embed(chunks)]
    collection = get_chroma_collection()
    filename = os.path.basename(file_path)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"file_name": filename} for _ in chunks]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return {"status": "ok", "chunks": len(chunks), "docs": 1}

def list_ingested_docs() -> list:
    try:
        collection = get_chroma_collection()
        results = collection.get(include=["metadatas"])
        sources = list({m.get("file_name", "unknown") for m in results["metadatas"] if m})
        return sorted(sources)
    except Exception:
        return []
```

---

### `rag.py`

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import httpx
import json
import chromadb
from fastembed import TextEmbedding
from config import (
    VLLM_URL, LLM_MODEL, TOP_K,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBED_MODEL
)

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(
            model_name=EMBED_MODEL,
            cache_dir="/opt/app-root/src/.cache"
        )
    return _embed_model

def retrieve_context(query: str):
    model = get_embed_model()
    embedding = list(model.embed([query]))[0].tolist()
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        include=["documents", "distances"]
    )
    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    return [doc for doc, dist in zip(docs, distances) if dist < 1.5]

def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(chunks)
    return (
        f"You are a helpful assistant. Answer ONLY using the context below. "
        f"Do NOT use any outside knowledge. "
        f"If the context does not contain the answer, respond exactly with: "
        f"'This information is not available in the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

async def stream_rag_response(query: str):
    chunks = retrieve_context(query)
    if not chunks:
        yield "data: This information is not available in the provided documents.\n\n"
        yield "data: [DONE]\n\n"
        return
    prompt = build_prompt(query, chunks)
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{VLLM_URL}/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    data = json.loads(raw)
                    token = data["choices"][0]["delta"].get("content", "")
                    if token:
                        yield f"data: {token}\n\n"
                except Exception:
                    continue
```

---

### `main.py`

```python
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ingest import ingest_documents, list_ingested_docs
from rag import stream_rag_response
from config import DOCS_DIR

app = FastAPI(title="rhel-ai-app")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class AskRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    docs = list_ingested_docs()
    return templates.TemplateResponse("index.html", {"request": request, "docs": docs})

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_documents(file_path=dest)
    return JSONResponse(result)

@app.get("/docs-list")
async def docs_list():
    return JSONResponse({"docs": list_ingested_docs()})

@app.post("/ask")
async def ask(body: AskRequest):
    return StreamingResponse(
        stream_rag_response(body.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

### `cleanup.py`

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb

client = chromadb.HttpClient(host="localhost", port=8001)
client.delete_collection("rhel-ai-docs")
print("Cleared")
```

---

### `requirements.txt`

```
fastapi==0.115.9
uvicorn[standard]==0.30.6
python-multipart==0.0.9
jinja2==3.1.4
httpx==0.27.2
pydantic==2.9.2
chromadb==1.0.0
fastembed==0.3.1
pdfminer.six
pysqlite3-binary
pypdf==4.3.1
```

---

### `Containerfile`

```dockerfile
FROM registry.access.redhat.com/ubi9/python-39:latest

USER root

RUN dnf install -y gcc make && dnf clean all && \
    mkdir -p /opt/app-root/src/data/docs \
             /opt/app-root/src/vectorstore \
             /opt/app-root/src/static \
             /opt/app-root/src/templates \
             /opt/app-root/src/.tmp \
             /opt/app-root/src/.cache && \
    chown -R 1001:0 /opt/app-root/src && \
    rm -rf /var/cache/dnf

USER 1001

WORKDIR /opt/app-root/src

ENV TMPDIR=/opt/app-root/src/.tmp
ENV PIP_NO_CACHE_DIR=1
ENV FASTEMBED_CACHE_PATH=/opt/app-root/src/.cache

COPY --chown=1001:0 requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /opt/app-root/src/.tmp/*

COPY --chown=1001:0 config.py .
COPY --chown=1001:0 main.py .
COPY --chown=1001:0 ingest.py .
COPY --chown=1001:0 rag.py .
COPY --chown=1001:0 cleanup.py .
COPY --chown=1001:0 templates/ templates/
COPY --chown=1001:0 static/ static/

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Run Commands Reference

```bash
# build
podman build -t rhel-ai-app:latest .

# fix volume ownership (uid 1001 = container user)
mkdir -p ~/rhel-ai-cache/fastembed ~/rhel-ai-data/docs
sudo chown -R 1001:0 ~/rhel-ai-cache/ ~/rhel-ai-data/
sudo chmod -R g+rwX ~/rhel-ai-cache/ ~/rhel-ai-data/

# run
podman run -d \
  --name rhel-ai-app \
  --network=host \
  -v ~/rhel-ai-cache/fastembed:/opt/app-root/src/.cache:Z \
  -v ~/rhel-ai-data/docs:/opt/app-root/src/data/docs:Z \
  rhel-ai-app:latest

# logs
podman logs -f rhel-ai-app

# rebuild after code change
podman rm -f rhel-ai-app
podman build -t rhel-ai-app:latest .
podman run -d --name rhel-ai-app --network=host \
  -v ~/rhel-ai-cache/fastembed:/opt/app-root/src/.cache:Z \
  -v ~/rhel-ai-data/docs:/opt/app-root/src/data/docs:Z \
  rhel-ai-app:latest

# clear vector store
podman exec rhel-ai-app python3 cleanup.py

# SSH tunnel to browser
ssh -L 9090:localhost:8080 cloud-user@<bastion-hostname>
# open: http://localhost:9090
```

---

## API Reference

| Route | Method | Description |
|---|---|---|
| `/` | GET | Chat UI |
| `/ingest` | POST | Upload + ingest PDF/TXT |
| `/ask` | POST | RAG query, SSE streamed |
| `/docs-list` | GET | List ingested documents |

---

## How RAG Works

```
INGEST:
  PDF → pdfminer extracts text → split into 512-char chunks
      → fastembed ONNX embeds each chunk (384-dim vector)
      → store (chunk + vector) in ChromaDB

QUERY:
  Question → embed → similarity search in ChromaDB (top 4)
           → build prompt: "Answer ONLY using this context"
           → send to Granite via ilab/vLLM (stream=True)
           → tokens stream via SSE → browser renders in real-time
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `podman login` fails | Use Red Hat Customer Portal credentials |
| ilab tensor-parallel error | Patch config: `gpus=1`, remove `tensor-parallel-size 4` |
| KV cache error | Set `--max-model-len 11680` in ilab config vllm_args |
| `No usable temporary directory` | `TMPDIR=/opt/app-root/src/.tmp` in Containerfile |
| `Permission denied: .cache` | `sudo chown -R 1001:0 ~/rhel-ai-cache/` |
| chromadb sqlite3 error | `pysqlite3-binary` in requirements.txt |
| Port not accessible externally | SSH tunnel: `ssh -L 9090:localhost:8080 ...` |
| fastembed download fails | Check volume ownership and network access |
| Answer uses general knowledge | Relevance filter `dist < 1.5` in `rag.py` |

---

## Why No HuggingFace Token?

RHEL AI ships with Granite pre-installed at:
```
~/.cache/instructlab/models/granite-3.1-8b-starter-v2/
```

`ilab model serve` loads it directly — no external download needed.

---

## Why fastembed Instead of sentence-transformers?

`sentence-transformers` pulls PyTorch + NVIDIA CUDA libs — 6GB+ unnecessary packages.
`fastembed` uses ONNX runtime — 200MB total, CPU only, same embedding quality.

| | sentence-transformers | fastembed |
|---|---|---|
| Size | ~7GB | ~200MB |
| GPU needed | Yes (pulls CUDA) | No |
| Quality | Good | Same |