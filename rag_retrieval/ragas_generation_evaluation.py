"""RAGAS-based grounded answer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable
import warnings

from .generation import _load_local_env
from .qa_generation import generate_short_answer
from .session import RetrievalSession


@dataclass(frozen=True, slots=True)
class GenerationEvalExample:
    query: str


DEFAULT_FILES = ("test1.pdf", "rawtext.pdf")

DEFAULT_EXAMPLES = (
    GenerationEvalExample("What was Bakirkoy natural gas consumption in 2024?"),
    GenerationEvalExample("What was Kadikoy electricity consumption in 2024?"),
    GenerationEvalExample("What was Adalar carbon footprint in 2024?"),
    GenerationEvalExample("What was Besiktas carbon footprint in 2024?"),
    GenerationEvalExample("What was Uskudar electricity consumption in 2022?"),
    GenerationEvalExample("What was Bayrampasa natural gas consumption in 2024?"),
)


def run_ragas_generation_evaluation(
    file_paths: Iterable[str | Path] | None = None,
    language: str = "English",
    model: str | None = None,
) -> dict:
    try:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness, ResponseGroundedness
    except ImportError as exc:  # pragma: no cover - dependency gated
        raise ImportError(
            "RAGAS generation evaluation requires optional dependencies. Install with: pip install '.[evaluation]'"
        ) from exc

    _load_local_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for RAGAS generation evaluation.")

    session = RetrievalSession()
    paths = _resolve_paths(file_paths)
    stats = session.build_index(paths)

    rows = []
    answers = []
    for example in DEFAULT_EXAMPLES:
        answer = generate_short_answer(example.query, session, language=language, model=model)
        answers.append(answer)
        rows.append(
            {
                "user_input": example.query,
                "retrieved_contexts": answer["contexts"],
                "response": answer["response"],
            }
        )

    judge_model = model or os.getenv("RAGAS_JUDGE_MODEL") or "gpt-4o-mini"
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=float(os.getenv("OPENAI_TIMEOUT", 45)),
    )
    llm = llm_factory(
        judge_model,
        client=client,
        temperature=0.0,
        max_tokens=250,
    )
    embeddings = OpenAIEmbeddings(
        client=client,
        model=os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
    )

    faithfulness_metric = Faithfulness(llm=llm)
    relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    groundedness_metric = ResponseGroundedness(llm=llm)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        warnings.filterwarnings("ignore", category=ResourceWarning, module="ragas")
        faithfulness_scores = []
        relevancy_scores = []
        groundedness_scores = []
        for row in rows:
            faithfulness_scores.append(
                faithfulness_metric.score(
                    user_input=row["user_input"],
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
            )
            relevancy_scores.append(
                relevancy_metric.score(
                    user_input=row["user_input"],
                    response=row["response"],
                )
            )
            groundedness_scores.append(
                groundedness_metric.score(
                    response=row["response"],
                    retrieved_contexts=row["retrieved_contexts"],
                )
            )

    per_query = []
    for row, answer, faithfulness, relevancy, groundedness in zip(
        rows,
        answers,
        faithfulness_scores,
        relevancy_scores,
        groundedness_scores,
        strict=True,
    ):
        per_query.append(
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
                "faithfulness": _metric_value(faithfulness),
                "answer_relevancy": _metric_value(relevancy),
                "response_groundedness": _metric_value(groundedness),
                "faithfulness_reason": getattr(faithfulness, "reason", None),
                "answer_relevancy_reason": getattr(relevancy, "reason", None),
                "response_groundedness_reason": getattr(groundedness, "reason", None),
                "query_result": answer["query_result"],
            }
        )

    output = {
        "files": [str(path) for path in paths],
        "build": {
            "elapsed_seconds": round(stats.elapsed_seconds, 4),
            "document_count": stats.document_count,
            "element_count": stats.element_count,
            "chunk_count": stats.chunk_count,
            "embedding_count": stats.embedding_count,
        },
        "generation_model": judge_model,
        "metrics": {
            "faithfulness": _mean_metric(faithfulness_scores),
            "answer_relevancy": _mean_metric(relevancy_scores),
            "response_groundedness": _mean_metric(groundedness_scores),
        },
        "answers": answers,
        "per_query": per_query,
    }

    output_dir = Path("outputs/evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ragas_generation_evaluation.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    output["answer_artifacts"] = write_generation_answer_artifacts(output, output_dir)
    output["output_path"] = str(output_path)
    return output


def render_generation_answers_markdown(output: dict) -> str:
    metrics = output.get("metrics") or {}
    lines = [
        "# Generated Answers Evaluation",
        "",
        f"- Generation model: {output.get('generation_model') or 'unknown'}",
        f"- Faithfulness: {_format_metric(metrics.get('faithfulness'))}",
        f"- Answer relevancy: {_format_metric(metrics.get('answer_relevancy'))}",
        f"- Response groundedness: {_format_metric(metrics.get('response_groundedness'))}",
        "",
    ]

    for index, item in enumerate(output.get("per_query") or [], start=1):
        lines.extend(
            [
                f"## Query {index}",
                "",
                f"**Question:** {item.get('user_input') or ''}",
                "",
                f"**Generated answer:** {item.get('response') or ''}",
                "",
                f"- Faithfulness: {_format_metric(item.get('faithfulness'))}",
                f"- Answer relevancy: {_format_metric(item.get('answer_relevancy'))}",
                f"- Response groundedness: {_format_metric(item.get('response_groundedness'))}",
                "",
            ]
        )

        contexts = item.get("retrieved_contexts") or []
        if contexts:
            lines.append("**Retrieved context snippets:**")
            lines.append("")
            for context in contexts:
                lines.append(f"- {context}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_generation_answer_artifacts(output: dict, output_dir: Path) -> dict[str, str]:
    answers_payload = []
    for item in output.get("per_query") or []:
        answers_payload.append(
            {
                "query": item.get("user_input"),
                "response": item.get("response"),
                "faithfulness": item.get("faithfulness"),
                "answer_relevancy": item.get("answer_relevancy"),
                "response_groundedness": item.get("response_groundedness"),
                "retrieved_contexts": item.get("retrieved_contexts") or [],
            }
        )

    answers_json_path = output_dir / "generated_answers.json"
    answers_json_path.write_text(json.dumps(answers_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    answers_md_path = output_dir / "generated_answers.md"
    answers_md_path.write_text(render_generation_answers_markdown(output), encoding="utf-8")

    return {
        "json": str(answers_json_path),
        "markdown": str(answers_md_path),
    }


def _resolve_paths(file_paths: Iterable[str | Path] | None) -> list[Path]:
    candidates = list(file_paths) if file_paths is not None else [Path(name) for name in DEFAULT_FILES]
    paths = [Path(value) for value in candidates]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No evaluation files found for RAGAS generation evaluation.")
    return existing


def _metric_value(result) -> float | None:
    value = getattr(result, "value", None)
    return float(value) if value is not None else None


def _mean_metric(results: list) -> float:
    values = [_metric_value(result) for result in results]
    valid = [value for value in values if value is not None]
    return float(sum(valid) / len(valid)) if valid else 0.0


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
