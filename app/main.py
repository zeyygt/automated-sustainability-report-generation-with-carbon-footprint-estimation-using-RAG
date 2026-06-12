"""FastAPI backend for SustainChat."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_retrieval import RetrievalSession, generate_sustainability_report
from rag_retrieval.generation import _load_local_env
from rag_retrieval.report_metrics import public_metrics
from app.session_state import AppSessionState, AppSessionStore

_load_local_env()

app = FastAPI(title="SustainChat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_ID_HEADER = "X-Session-ID"
_app_sessions = AppSessionStore()


# ── Request models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    language: str = "English"


class ReportRequest(BaseModel):
    title: str = "Istanbul Metropolitan Municipality Sustainability Report"
    language: str = "English"


class FormulaVariableInput(BaseModel):
    type: str
    value: float | None = None
    column: str | None = None


class FormulaInputRequest(BaseModel):
    variables: dict[str, FormulaVariableInput]


class MethodologyResolutionRequest(BaseModel):
    formula_doc_id: str | None = None
    factor_doc_ids: dict[str, str] = {}


def _session_id_from_request(request: Request) -> str | None:
    return request.headers.get(SESSION_ID_HEADER) or request.query_params.get("session_id")


def _state_for_request(request: Request) -> AppSessionState:
    session_id = _session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    try:
        return _app_sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found.") from exc


def _emission_factors_payload(state: AppSessionState) -> dict:
    session = state.retrieval_session
    from rag_retrieval.data_engine import _load_reference_data

    reference = _load_reference_data().get("emission_factors", {})
    found: list[dict] = []
    for doc_id, factors in session.document_emission_factors.items():
        doc = session.documents.get(doc_id)
        found.append(
            {
                "doc_id": doc_id,
                "filename": doc.filename if doc else doc_id,
                "factors": factors,
            }
        )

    applied: dict = {}
    for engine in session.data_engines.values():
        if engine is not None:
            applied = {
                k: {"value": v, "source": engine.emission_factors_source.get(k, "reference")}
                for k, v in engine.emission_factors.items()
            }
            break

    custom = None
    if session.custom_formula:
        f = session.custom_formula
        custom = {
            "expression": f.expression,
            "constants": f.constants,
            "variable_hints": f.variable_hints,
            "confidence": f.confidence,
            "source_text": f.source_text,
        }

    return {
        "session_id": session.session_id,
        "reference_defaults": reference,
        "extracted_from_documents": found,
        "applied_in_calculations": applied,
        "factor_override_keys": list(getattr(session, "factor_override_keys", []) or []),
        "custom_formula": custom,
        "formula_extraction_method": session.formula_extraction_method,
        "custom_formula_status": getattr(session, "custom_formula_status", "default"),
        "custom_formula_missing_variables": list(getattr(session, "custom_formula_missing_variables", []) or []),
        "custom_formula_validation": list(getattr(session, "custom_formula_validation_by_document", []) or []),
        "custom_formula_user_inputs": dict(getattr(session, "custom_formula_user_inputs", {}) or {}),
        "formula_input_columns": list(getattr(session, "formula_input_columns", []) or []),
        "has_structured_data": bool(getattr(session, "has_structured_data", False)),
        "structured_document_count": int(getattr(session, "structured_document_count", 0) or 0),
        "structured_district_count": int(getattr(session, "structured_district_count", 0) or 0),
        "methodology_status": getattr(session, "methodology_status", "clear"),
        "methodology_warnings": list(getattr(session, "methodology_warnings", []) or []),
        "methodology_resolution": dict(getattr(session, "methodology_resolution", {}) or {}),
        "factor_conflicts": list(getattr(session, "factor_conflicts", []) or []),
        "formula_conflicts": list(getattr(session, "formula_conflicts", []) or []),
        "report_generation_status": getattr(session, "report_generation_status", "ready"),
        "report_generation_warnings": list(getattr(session, "report_generation_warnings", []) or []),
        "calculation_audit": dict(getattr(session, "calculation_audit", {}) or {}),
    }


def _session_upload_path(temp_dir: Path, filename: str | None) -> Path:
    candidate_name = Path(filename or "upload.bin").name or "upload.bin"
    candidate = temp_dir / candidate_name
    if not candidate.exists():
        return candidate

    stem = Path(candidate_name).stem or "upload"
    suffix = Path(candidate_name).suffix
    counter = 2
    while True:
        candidate = temp_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/session")
async def create_session():
    state = _app_sessions.create()
    return {
        "session_id": state.session_id,
        "files": state.uploaded_files,
        "has_report": False,
        "message_count": 0,
    }


@app.get("/status")
async def status(request: Request):
    state = _state_for_request(request)
    return {
        "session_id": state.session_id,
        "files": state.uploaded_files,
        "has_report": state.report is not None,
        "message_count": len(state.chat_history) // 2,
    }


@app.get("/emission-factors")
async def emission_factors(request: Request):
    state = _state_for_request(request)
    return _emission_factors_payload(state)


@app.post("/formula-inputs")
async def formula_inputs(request: Request, payload: FormulaInputRequest):
    state = _state_for_request(request)
    if state.retrieval_session.custom_formula is None:
        raise HTTPException(status_code=400, detail="No custom formula is available for resolution.")

    try:
        serialized = {
            name: value.model_dump(exclude_none=True)
            for name, value in payload.variables.items()
        }
        state.retrieval_session.update_custom_formula_inputs(serialized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _emission_factors_payload(state)


@app.post("/methodology-resolution")
async def methodology_resolution(request: Request, payload: MethodologyResolutionRequest):
    state = _state_for_request(request)
    try:
        state.retrieval_session.update_methodology_resolution(
            formula_doc_id=payload.formula_doc_id,
            factor_doc_ids=payload.factor_doc_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _emission_factors_payload(state)


@app.post("/upload")
async def upload_files(request: Request, files: list[UploadFile]):
    state = _state_for_request(request)

    saved_paths = []
    new_files = []

    for file in files:
        suffix = Path(file.filename or "file.bin").suffix.lower()
        dest = _session_upload_path(state.temp_dir, file.filename)
        dest.write_bytes(await file.read())
        saved_paths.append(dest)
        new_files.append({
            "name": file.filename,
            "type": "PDF" if suffix == ".pdf" else "XLS",
        })

    stats = await asyncio.to_thread(state.retrieval_session.build_index, saved_paths)
    state.uploaded_files.extend(new_files)

    return {
        "session_id": state.session_id,
        "files": new_files,
        "stats": {
            "document_count": stats.document_count,
            "chunk_count": stats.chunk_count,
            "elapsed_seconds": round(stats.elapsed_seconds, 2),
        },
    }


@app.post("/chat")
async def chat(request: Request, payload: ChatRequest):
    state = _state_for_request(request)
    if not state.uploaded_files:
        raise HTTPException(status_code=400, detail="Upload at least one file first.")

    async def event_stream():
        from rag_retrieval.chatbot import stream_chat_response

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        full_response: list[str] = []

        def producer():
            try:
                for chunk in stream_chat_response(
                    payload.message,
                    state.retrieval_session,
                    state.chat_history,
                    state.report_markdown,
                    payload.language,
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
                state.chat_history.append({"role": "user", "content": payload.message})
                state.chat_history.append({"role": "assistant", "content": "".join(full_response)})
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/generate-report")
async def generate_report_endpoint(request: Request, payload: ReportRequest):
    state = _state_for_request(request)
    if not state.uploaded_files:
        raise HTTPException(status_code=400, detail="Upload at least one file first.")

    output_dir = state.temp_dir / "reports"
    try:
        report = await asyncio.to_thread(
            generate_sustainability_report,
            state.retrieval_session,
            title=payload.title,
            language=payload.language,
            output_dir=str(output_dir),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    state.report = report
    state.report_markdown = report.ai_content_markdown

    charts = [
        {"title": c["title"], "filename": Path(c["path"]).name}
        for c in report.charts
    ]
    metrics = public_metrics(report.report_input.structured_results)

    return {
        "session_id": state.session_id,
        "title": report.title,
        "language": report.language,
        "generated_at": report.generated_at,
        "warnings": report.warnings,
        "markdown": report.ai_content_markdown,
        "has_pdf": bool(report.pdf_path and report.pdf_path.exists()),
        "charts": charts,
        "metrics": metrics,
        "audit": dict(getattr(state.retrieval_session, "calculation_audit", {}) or {}),
    }


@app.get("/report/html")
async def download_html(request: Request):
    state = _state_for_request(request)
    report = state.report
    if not report or not report.html_path or not report.html_path.exists():
        raise HTTPException(status_code=404, detail="No report available.")
    return FileResponse(
        report.html_path,
        media_type="text/html",
        filename="sustainability_report.html",
    )


@app.get("/report/pdf")
async def download_pdf(request: Request):
    state = _state_for_request(request)
    report = state.report
    if not report or not report.pdf_path or not report.pdf_path.exists():
        raise HTTPException(status_code=404, detail="No PDF available.")
    return FileResponse(
        report.pdf_path,
        media_type="application/pdf",
        filename="sustainability_report.pdf",
    )


@app.get("/charts/{filename}")
async def get_chart(filename: str, request: Request):
    state = _state_for_request(request)
    path = state.temp_dir / "reports" / "charts" / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chart not found.")
    return FileResponse(path, media_type="image/png")


@app.delete("/session")
async def clear_session(request: Request):
    session_id = _session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    _app_sessions.close(session_id)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
