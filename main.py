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
