"""FastAPI backend for SustainChat."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_retrieval import RetrievalSession, generate_sustainability_report
from rag_retrieval.generation import _load_local_env
from rag_retrieval.report_metrics import public_metrics

_load_local_env()

app = FastAPI(title="SustainChat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global session state ──────────────────────────────────────────────────────
_session: RetrievalSession = RetrievalSession()
_uploaded_files: list[dict] = []
_report = None
_report_markdown: str | None = None
_chat_history: list[dict] = []
_temp_dir = tempfile.mkdtemp(prefix="sustainchat_")


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    language: str = "English"


class ReportRequest(BaseModel):
    title: str = "Istanbul Metropolitan Municipality Sustainability Report"
    language: str = "English"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/status")
async def status():
    return {
        "files": _uploaded_files,
        "has_report": _report is not None,
        "message_count": len(_chat_history) // 2,
    }


@app.get("/emission-factors")
async def emission_factors():
    """Return emission factors found in uploaded documents vs reference defaults."""
    from rag_retrieval.data_engine import _load_reference_data

    reference = _load_reference_data().get("emission_factors", {})
    found: list[dict] = []
    for doc_id, factors in _session.document_emission_factors.items():
        doc = _session.documents.get(doc_id)
        found.append({
            "filename": doc.filename if doc else doc_id,
            "factors": factors,
        })

    # Pick a live engine to show what is actually being applied
    applied: dict = {}
    for engine in _session.data_engines.values():
        if engine is not None:
            applied = {
                k: {"value": v, "source": engine.emission_factors_source.get(k, "reference")}
                for k, v in engine.emission_factors.items()
            }
            break

    custom = None
    if _session.custom_formula:
        f = _session.custom_formula
        custom = {
            "expression": f.expression,
            "constants": f.constants,
            "variable_hints": f.variable_hints,
            "confidence": f.confidence,
            "source_text": f.source_text,
        }

    return {
        "reference_defaults": reference,
        "extracted_from_documents": found,
        "applied_in_calculations": applied,
        "custom_formula": custom,
        "formula_extraction_method": _session.formula_extraction_method,
    }


@app.post("/upload")
async def upload_files(files: list[UploadFile]):
    global _session, _uploaded_files

    saved_paths = []
    new_files = []

    for file in files:
        suffix = Path(file.filename or "file.bin").suffix.lower()
        dest = Path(_temp_dir) / f"{uuid.uuid4()}{suffix}"
        dest.write_bytes(await file.read())
        saved_paths.append(dest)
        new_files.append({
            "name": file.filename,
            "type": "PDF" if suffix == ".pdf" else "XLS",
        })

    stats = await asyncio.to_thread(_session.build_index, saved_paths)
    _uploaded_files.extend(new_files)

    return {
        "files": new_files,
        "stats": {
            "document_count": stats.document_count,
            "chunk_count": stats.chunk_count,
            "elapsed_seconds": round(stats.elapsed_seconds, 2),
        },
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if not _uploaded_files:
        raise HTTPException(status_code=400, detail="Upload at least one file first.")

    async def event_stream():
        from rag_retrieval.chatbot import stream_chat_response

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        full_response: list[str] = []

        def producer():
            try:
                for chunk in stream_chat_response(
                    request.message,
                    _session,
                    _chat_history,
                    _report_markdown,
                    request.language,
                ):
                    full_response.append(chunk)
                    asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put(("error", str(exc))), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                yield f"data: {json.dumps({'text': value})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'error': value, 'done': True})}\n\n"
                return
            elif kind == "done":
                _chat_history.append({"role": "user", "content": request.message})
                _chat_history.append({"role": "assistant", "content": "".join(full_response)})
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate-report")
async def generate_report_endpoint(request: ReportRequest):
    global _report, _report_markdown

    if not _uploaded_files:
        raise HTTPException(status_code=400, detail="Upload at least one file first.")

    output_dir = Path(_temp_dir) / "reports"
    try:
        report = await asyncio.to_thread(
            generate_sustainability_report,
            _session,
            title=request.title,
            language=request.language,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _report = report
    _report_markdown = report.ai_content_markdown

    charts = [
        {"title": c["title"], "filename": Path(c["path"]).name}
        for c in report.charts
    ]
    metrics = public_metrics(report.report_input.structured_results)

    return {
        "title": report.title,
        "language": report.language,
        "generated_at": report.generated_at,
        "warnings": report.warnings,
        "markdown": report.ai_content_markdown,
        "has_pdf": bool(report.pdf_path and report.pdf_path.exists()),
        "charts": charts,
        "metrics": metrics,
    }


@app.get("/report/html")
async def download_html():
    if not _report or not _report.html_path or not _report.html_path.exists():
        raise HTTPException(status_code=404, detail="No report available.")
    return FileResponse(
        _report.html_path,
        media_type="text/html",
        filename="sustainability_report.html",
    )


@app.get("/report/pdf")
async def download_pdf():
    if not _report or not _report.pdf_path or not _report.pdf_path.exists():
        raise HTTPException(status_code=404, detail="No PDF available.")
    return FileResponse(
        _report.pdf_path,
        media_type="application/pdf",
        filename="sustainability_report.pdf",
    )


@app.get("/charts/{filename}")
async def get_chart(filename: str):
    path = Path(_temp_dir) / "reports" / "charts" / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path, media_type="image/png")


@app.delete("/session")
async def clear_session():
    global _session, _uploaded_files, _report, _report_markdown, _chat_history
    _session = RetrievalSession()
    _uploaded_files.clear()
    _report = None
    _report_markdown = None
    _chat_history.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
