"""Run RAGAS-based retrieval evaluation on local sample documents."""

from __future__ import annotations

import json
import sys

from rag_retrieval.ragas_evaluation import run_ragas_retrieval_evaluation


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    report = run_ragas_retrieval_evaluation(file_paths=argv or None)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
