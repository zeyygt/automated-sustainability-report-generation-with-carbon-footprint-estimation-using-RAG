"""Generate a formatted sustainability report from local test uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_retrieval import RetrievalSession, generate_sustainability_report
from rag_retrieval.report_builder import DEFAULT_REPORT_QUERIES


DEFAULT_FILE_CANDIDATES = ("test.pdf", "test1.pdf")


def _default_paths() -> list[Path]:
    for filename in DEFAULT_FILE_CANDIDATES:
        path = Path(filename)
        if path.exists():
            return [path]
    return [Path(DEFAULT_FILE_CANDIDATES[0])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sustainability report from uploaded session documents.")
    parser.add_argument("files", nargs="*", help="Input PDF/Excel files. Defaults to the local test PDF.")
    parser.add_argument("--title", default="Istanbul Metropolitan Municipality Sustainability Report")
    parser.add_argument("--language", default="English")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--query", action="append", dest="queries", help="Report query/scope. Can be passed multiple times.")
    parser.add_argument("--model", default=None, help="OpenAI model override. Defaults to OPENAI_MODEL or gpt-4o.")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort override. Defaults to high.")
    args = parser.parse_args()

    paths = [Path(value) for value in args.files] if args.files else _default_paths()
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise SystemExit("No input files found.")

    session = RetrievalSession()
    stats = session.build_index(paths)
    report = generate_sustainability_report(
        session,
        queries=args.queries or DEFAULT_REPORT_QUERIES,
        title=args.title,
        language=args.language,
        output_dir=args.output_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )

    summary = {
        "build": {
            "elapsed_seconds": round(stats.elapsed_seconds, 4),
            "document_count": stats.document_count,
            "element_count": stats.element_count,
            "chunk_count": stats.chunk_count,
            "embedding_count": stats.embedding_count,
        },
        "html_path": str(report.html_path),
        "pdf_path": str(report.pdf_path),
        "chart_count": len(report.charts),
        "warnings": report.warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
