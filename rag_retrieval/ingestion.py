"""Runtime document ingestion for session-scoped indexes."""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Iterable

from .models import DocumentInput


SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class DocumentIngestor:
    """Validates runtime uploads and converts them into pipeline inputs."""

    def ingest_paths(self, paths: Iterable[str | Path]) -> list[DocumentInput]:
        documents: list[DocumentInput] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            if not path.is_file():
                raise ValueError(f"Expected a file path, got directory: {path}")

            extension = path.suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported document type for {path.name}")

            mime_type = mimetypes.guess_type(path.name)[0] or SUPPORTED_EXTENSIONS[extension]
            documents.append(
                DocumentInput(
                    doc_id=str(uuid.uuid4()),
                    path=path,
                    filename=path.name,
                    mime_type=mime_type,
                    metadata={"extension": extension, "size_bytes": path.stat().st_size},
                )
            )
        return documents
