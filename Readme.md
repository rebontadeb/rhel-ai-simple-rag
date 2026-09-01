# rhel-ai-app — Student Lab Guide

> Build a fully local RAG chatbot on RHEL AI using Granite LLM, ChromaDB, and FastAPI.
> No cloud. No OpenAI. Everything runs on-box.

---

## What You Will Build

A web application where you:
1. Upload a PDF document
2. Ask questions about it in a chat UI
3. Granite 3.1 8B answers using only your document

**Stack:**

| Layer | Tool |
|---|---|
| LLM | IBM Granite 3.1 8B via Red Hat AI Inference Server (vLLM) |
| Embedding | all-MiniLM-L6-v2 (sentence-transformers, CPU) |
| Vector Store | ChromaDB 1.0.0 |
| PDF Extraction | pdfminer.six |
| API | FastAPI + uvicorn |
| UI | HTMX + SSE streaming |
| Containers | Podman |

---

## Prerequisites

- RHEL AI installed on an EC2 instance (or equivalent)
- NVIDIA GPU (L4 or better recommended)
- Podman installed
- Python 3.9+
- Red Hat Customer Portal account
- HuggingFace account + API token

---

## Project Structure

```
rhel-ai-app/
├── config.py              ← all settings
├── ingest.py              ← PDF → chunks → embeddings → ChromaDB
├── rag.py                 ← query → retrieve → Granite → SSE stream
├── main.py                ← FastAPI routes
├── cleanup.py             ← clear vector store
├── requirements.txt       ← Python dependencies
├── Containerfile          ← builds app container image
├── pod-start.sh           ← starts all 3 containers
├── pod-stop.sh            ← stops everything
├── static/                ← (empty, reserved for static files)
├── data/docs/             ← uploaded PDFs land here
├── vectorstore/           ← ChromaDB persistence
└── templates/
    └── index.html         ← HTMX chat UI
```

---

## Step 1 — Login to Red Hat Registry

```bash
podman login registry.redhat.io
# enter your Red Hat Customer Portal username and password
```

Expected output:
```
Login Succeeded!
```

---

## Step 2 — Get a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → Read access
3. Copy the token

```bash
echo "export HF_TOKEN=hf_your_token_here" > ~/private.env
source ~/private.env
```

---

## Step 3 — Create Project Directory

```bash
mkdir -p ~/rhel-ai-app/{static,data/docs,vectorstore,templates}
cd ~/rhel-ai-app
```

---

## Step 4 — Create Application Files

Create each file below inside `~/rhel-ai-app/`.

---

### `config.py`

```python
VLLM_URL = "http://localhost:8000/v1"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8001
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "ibm-granite/granite-3.1-8b-instruct"
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
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdfminer_extract
from config import (
    CHROMA_HOST, CHROMA_PORT, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION
)

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
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
    embeddings = model.encode(chunks).tolist()
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
from sentence_transformers import SentenceTransformer
from config import (
    VLLM_URL, LLM_MODEL, TOP_K,
    CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION, EMBED_MODEL
)

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def retrieve_context(query: str):
    model = get_embed_model()
    embedding = model.encode([query]).tolist()[0]
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
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9
jinja2==3.1.4
httpx==0.27.2
pydantic==2.9.2
sentence-transformers==2.7.0
pypdf==4.3.1
```

---

### `templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>rhel-ai-app</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
      --accent: #cc0000; --accent-dim: #8b0000;
      --text: #e8e8ec; --text-dim: #7a7d8e; --success: #3ecf8e;
      --mono: 'JetBrains Mono', monospace; --sans: 'Inter', system-ui, sans-serif;
    }
    body { background: var(--bg); color: var(--text); font-family: var(--sans);
      height: 100vh; display: grid;
      grid-template-columns: 260px 1fr; grid-template-rows: 56px 1fr; }
    header { grid-column: 1 / -1; background: var(--surface);
      border-bottom: 1px solid var(--border); display: flex;
      align-items: center; padding: 0 24px; gap: 12px; }
    header .logo { width: 28px; height: 28px; background: var(--accent);
      border-radius: 6px; display: flex; align-items: center;
      justify-content: center; font-weight: 800; font-size: 14px; color: #fff; }
    header h1 { font-size: 15px; font-weight: 600; color: var(--text); }
    header span { font-size: 11px; color: var(--text-dim); background: var(--border);
      padding: 2px 8px; border-radius: 100px; margin-left: auto; font-family: var(--mono); }
    aside { background: var(--surface); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; padding: 20px 16px; gap: 16px; overflow-y: auto; }
    aside h2 { font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--text-dim); }
    .upload-zone { border: 1px dashed var(--border); border-radius: 8px;
      padding: 16px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
    .upload-zone:hover { border-color: var(--accent); }
    .upload-zone p { font-size: 12px; color: var(--text-dim); margin-top: 6px; }
    .upload-zone input { display: none; }
    #upload-btn { width: 100%; padding: 8px; background: var(--accent); color: #fff;
      border: none; border-radius: 6px; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: background 0.2s; }
    #upload-btn:hover { background: var(--accent-dim); }
    #upload-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    #upload-status { font-size: 11px; color: var(--success); min-height: 16px; font-family: var(--mono); }
    .doc-list { display: flex; flex-direction: column; gap: 6px; }
    .doc-item { font-size: 12px; color: var(--text-dim); padding: 6px 8px;
      background: var(--bg); border-radius: 6px; border: 1px solid var(--border);
      font-family: var(--mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .doc-item::before { content: "📄 "; }
    main { display: flex; flex-direction: column; overflow: hidden; }
    #chat-history { flex: 1; overflow-y: auto; padding: 24px;
      display: flex; flex-direction: column; gap: 16px; }
    .msg { display: flex; flex-direction: column; gap: 4px; max-width: 780px; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.assistant { align-self: flex-start; }
    .msg-label { font-size: 10px; font-weight: 600; letter-spacing: 0.06em;
      text-transform: uppercase; color: var(--text-dim); }
    .msg-bubble { padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.65; }
    .msg.user .msg-bubble { background: var(--accent); color: #fff; border-bottom-right-radius: 3px; }
    .msg.assistant .msg-bubble { background: var(--surface); border: 1px solid var(--border);
      border-bottom-left-radius: 3px; color: var(--text); font-family: var(--sans);
      white-space: pre-wrap; word-break: break-word; word-spacing: normal; letter-spacing: normal; }
    .cursor { display: inline-block; width: 2px; height: 14px; background: var(--accent);
      margin-left: 2px; animation: blink 1s step-end infinite; vertical-align: middle; }
    @keyframes blink { 50% { opacity: 0; } }
    .input-bar { padding: 16px 24px; border-top: 1px solid var(--border);
      background: var(--surface); display: flex; gap: 10px; align-items: flex-end; }
    #query-input { flex: 1; background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 10px 14px; color: var(--text); font-size: 14px;
      font-family: var(--sans); resize: none; outline: none; max-height: 120px;
      transition: border-color 0.2s; }
    #query-input:focus { border-color: var(--accent); }
    #query-input::placeholder { color: var(--text-dim); }
    #send-btn { padding: 10px 20px; background: var(--accent); color: #fff; border: none;
      border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer;
      transition: background 0.2s; white-space: nowrap; }
    #send-btn:hover { background: var(--accent-dim); }
    #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .empty-state { flex: 1; display: flex; flex-direction: column; align-items: center;
      justify-content: center; gap: 12px; color: var(--text-dim); }
    .empty-state .icon { font-size: 40px; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  </style>
</head>
<body>
<header>
  <div class="logo">R</div>
  <h1>rhel-ai-app</h1>
  <span>Granite · RAG · vLLM</span>
</header>
<aside>
  <h2>Documents</h2>
  <div class="upload-zone" onclick="document.getElementById('file-input').click()">
    <div style="font-size:24px">📂</div>
    <p>Click to upload PDF or TXT</p>
    <input type="file" id="file-input" accept=".pdf,.txt,.md" onchange="handleUpload(this)"/>
  </div>
  <button id="upload-btn" disabled>Upload</button>
  <div id="upload-status"></div>
  <h2>Ingested</h2>
  <div class="doc-list" id="doc-list">
    {% for doc in docs %}
    <div class="doc-item" title="{{ doc }}">{{ doc }}</div>
    {% else %}
    <div style="font-size:12px; color: var(--text-dim)">No docs yet.</div>
    {% endfor %}
  </div>
</aside>
<main>
  <div id="chat-history">
    <div class="empty-state" id="empty-state">
      <div class="icon">🔍</div>
      <p>Upload a document, then ask anything.</p>
    </div>
  </div>
  <div class="input-bar">
    <textarea id="query-input" rows="1" placeholder="Ask about your documents..."
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="send-btn" onclick="sendQuery()">Ask</button>
  </div>
</main>
<script>
  let selectedFile = null;
  function handleUpload(input) {
    selectedFile = input.files[0];
    if (selectedFile) {
      document.getElementById('upload-btn').disabled = false;
      document.getElementById('upload-status').textContent = selectedFile.name;
    }
  }
  document.getElementById('upload-btn').addEventListener('click', async () => {
    if (!selectedFile) return;
    const btn = document.getElementById('upload-btn');
    const status = document.getElementById('upload-status');
    btn.disabled = true; btn.textContent = 'Ingesting...'; status.textContent = '';
    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const res = await fetch('/ingest', { method: 'POST', body: form });
      const data = await res.json();
      if (data.status === 'ok') { status.textContent = `✓ ${data.chunks} chunks ingested`; refreshDocList(); }
      else { status.textContent = `✗ ${data.message}`; }
    } catch (e) { status.textContent = '✗ Upload failed'; }
    finally { btn.textContent = 'Upload'; selectedFile = null; document.getElementById('file-input').value = ''; }
  });
  async function refreshDocList() {
    const res = await fetch('/docs-list');
    const data = await res.json();
    const list = document.getElementById('doc-list');
    list.innerHTML = data.docs.length
      ? data.docs.map(d => `<div class="doc-item" title="${d}">${d}</div>`).join('')
      : '<div style="font-size:12px; color: var(--text-dim)">No docs yet.</div>';
  }
  function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); } }
  function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }
  function appendMessage(role, text) {
    const history = document.getElementById('chat-history');
    document.getElementById('empty-state')?.remove();
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;
    wrap.innerHTML = `<div class="msg-label">${role === 'user' ? 'You' : 'Granite'}</div><div class="msg-bubble">${text}</div>`;
    history.appendChild(wrap);
    history.scrollTop = history.scrollHeight;
    return wrap.querySelector('.msg-bubble');
  }
  async function sendQuery() {
    const input = document.getElementById('query-input');
    const btn = document.getElementById('send-btn');
    const query = input.value.trim();
    if (!query) return;
    input.value = ''; input.style.height = 'auto'; btn.disabled = true;
    appendMessage('user', query);
    const bubble = appendMessage('assistant', '');
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);
    try {
      const res = await fetch('/ask', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '', fullText = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5);
          if (raw.trim() === '[DONE]') break;
          fullText += raw;
          cursor.previousSibling?.remove();
          bubble.insertBefore(document.createTextNode(fullText), cursor);
          document.getElementById('chat-history').scrollTop = 99999;
        }
      }
    } catch (e) {
      bubble.insertBefore(document.createTextNode('Error connecting to server.'), cursor);
    } finally { cursor.remove(); btn.disabled = false; }
  }
</script>
</body>
</html>
```

---

### `Containerfile`

```dockerfile
FROM registry.access.redhat.com/ubi9/python-39:latest

USER root
RUN dnf install -y gcc make && dnf clean all
USER 1001

WORKDIR /opt/app-root/src

COPY --chown=1001:0 requirements.txt .
COPY --chown=1001:0 config.py .
COPY --chown=1001:0 main.py .
COPY --chown=1001:0 ingest.py .
COPY --chown=1001:0 rag.py .
COPY --chown=1001:0 cleanup.py .
COPY --chown=1001:0 templates/ templates/
COPY --chown=1001:0 static/ static/

RUN mkdir -p data/docs vectorstore

RUN pip install --upgrade pip && \
    pip install pysqlite3-binary pdfminer.six && \
    pip install -r requirements.txt && \
    pip install "chromadb==1.0.0"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### `pod-start.sh`

```bash
#!/bin/bash
set -e

POD_NAME="rhel-ai-pod"
HF_TOKEN="${HF_TOKEN:-}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set. Run: source ~/private.env"
  exit 1
fi

echo "==> Creating pod..."
podman pod create \
  --name "$POD_NAME" \
  -p 8080:8080 \
  -p 8000:8000 \
  -p 8001:8001 \
  2>/dev/null || echo "Pod already exists, continuing..."

# 1. ChromaDB
echo "==> Starting ChromaDB..."
mkdir -p ~/vectorstore
podman run -d \
  --pod "$POD_NAME" \
  --name chroma \
  -v ~/vectorstore:/chroma/chroma \
  chromadb/chroma 2>/dev/null || echo "chroma already running"

sleep 3
echo "    ChromaDB: $(curl -s http://localhost:8001/api/v2/heartbeat)"

# 2. Granite / vLLM
echo "==> Starting Granite (first run ~5 min)..."
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

until curl -s http://localhost:8000/v1/models | grep -q "granite"; do
  sleep 15; echo "    Still loading..."
done
echo "    Granite: UP"

# 3. rhel-ai-app
echo "==> Building and starting rhel-ai-app..."
podman build -t rhel-ai-app:latest .
podman run -d \
  --pod "$POD_NAME" \
  --name rhel-ai-app \
  rhel-ai-app:latest 2>/dev/null || echo "rhel-ai-app already running"

sleep 3
echo ""
echo "==> All done!"
podman ps --pod
echo ""
echo "==> App URL: http://localhost:8080"
echo "==> SSH tunnel: ssh -L 9090:localhost:8080 cloud-user@<your-bastion>"
```

---

### `pod-stop.sh`

```bash
#!/bin/bash
echo "==> Stopping pod..."
podman pod stop rhel-ai-pod
podman pod rm rhel-ai-pod
echo "==> Done. Data in ~/vectorstore preserved."
```

---

## Step 5 — Run Everything

```bash
cd ~/rhel-ai-app
source ~/private.env
chmod +x pod-start.sh pod-stop.sh
./pod-start.sh
```

---

## Step 6 — Access the App

Only SSH port is open externally. Use SSH tunnel from your local machine:

```bash
ssh -L 9090:localhost:8080 cloud-user@<your-bastion-hostname>
```

Open browser: `http://localhost:9090`

---

## Step 7 — Use the App

1. Click the folder icon in the sidebar
2. Select a PDF from your computer
3. Click **Upload** — wait for "✓ N chunks ingested"
4. Type a question in the chat bar
5. Press Enter or click **Ask**
6. Granite answers using only your document

---

## Step 8 — Stop Everything

```bash
./pod-stop.sh
```

---

## Useful Commands

```bash
# check all containers
podman ps --pod

# watch Granite startup logs
podman logs -f granite-server

# watch app logs
podman logs -f rhel-ai-app

# clear vector store and re-ingest
python3 cleanup.py

# rebuild app after code change
podman rm -f rhel-ai-app
podman build -t rhel-ai-app:latest .
podman run -d --pod rhel-ai-pod --name rhel-ai-app rhel-ai-app:latest

# test via curl
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
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

## How RAG Works (Concept)

```
INGEST:
  PDF → extract text → split into chunks → embed each chunk
      → store (chunk + embedding) in ChromaDB

QUERY:
  Question → embed question → find similar chunks in ChromaDB
           → send chunks + question to Granite
           → Granite answers using only those chunks
           → stream tokens back to browser
```

The key insight: Granite never sees the full document. It only sees the most relevant chunks retrieved by similarity search. This keeps responses accurate and grounded.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `podman login` fails | Use Red Hat Customer Portal credentials, not email |
| Granite exits immediately | Check `podman logs granite-server` for error |
| `No usable temporary directory` | Add `--mount type=tmpfs,destination=/tmp,tmpfs-mode=1777` |
| `KV cache not enough memory` | Add `--max-model-len 27680` |
| `chromadb sqlite3 error` | `pip install pysqlite3-binary` — already in Containerfile |
| `resolution-too-deep` pip error | Pin all versions in requirements.txt |
| Port 8080 not accessible | Use SSH tunnel: `ssh -L 9090:localhost:8080 ...` |
| `[DONE]` visible in response | UI bug — trim check on SSE `[DONE]` token |
| Answer uses general knowledge | Check relevance filter `dist < 1.5` in `rag.py` |
