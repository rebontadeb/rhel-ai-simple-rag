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
        _embed_model = TextEmbedding(model_name=EMBED_MODEL, cache_dir="/opt/app-root/src/.cache")
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