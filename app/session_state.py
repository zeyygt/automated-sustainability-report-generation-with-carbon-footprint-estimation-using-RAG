"""Per-browser session state for the SustainChat web app."""

from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_retrieval import RetrievalSession


@dataclass(slots=True)
class AppSessionState:
    retrieval_session: RetrievalSession
    uploaded_files: list[dict[str, str]] = field(default_factory=list)
    report: Any | None = None
    report_markdown: str | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="sustainchat_")))

    @property
    def session_id(self) -> str:
        return self.retrieval_session.session_id

    def clear(self) -> None:
        self.retrieval_session.clear()
        self.uploaded_files.clear()
        self.report = None
        self.report_markdown = None
        self.chat_history.clear()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class AppSessionStore:
    """Thread-safe in-memory app-session store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, AppSessionState] = {}

    def create(self) -> AppSessionState:
        state = AppSessionState(retrieval_session=RetrievalSession())
        with self._lock:
            self._sessions[state.session_id] = state
        return state

    def get(self, session_id: str) -> AppSessionState:
        with self._lock:
            return self._sessions[session_id]

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def close(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            state.clear()
