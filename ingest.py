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
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)
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
    embeddings = list(model.embed(chunks))
    embeddings = [e.tolist() for e in embeddings]
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
