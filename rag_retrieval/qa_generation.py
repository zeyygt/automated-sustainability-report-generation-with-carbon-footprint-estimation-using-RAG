"""Short-form grounded answer generation for evaluation and chat use."""

from __future__ import annotations

import json
import os

from .generation import _load_local_env
from .pipeline import format_retrieval_hits
from .pipeline import handle_query
from .session import RetrievalSession


def build_answer_contexts(query_result: dict, max_retrieval: int = 5, max_structured: int = 5) -> list[str]:
    contexts: list[str] = []

    for item in (query_result.get("retrieval_context") or [])[:max_retrieval]:
        text = (item.get("text") or "").strip()
        if text:
            contexts.append(" ".join(text.split()))

    for item in (query_result.get("structured_results") or [])[:max_structured]:
        data = item.get("data") or {}
        district = data.get("district")
        if not district:
            continue

        parts = [f"District: {district}"]
        if data.get("total_emission") is not None:
            parts.append(f"total_emission={data['total_emission']:,.2f}")
        if data.get("gas_emission") is not None:
            parts.append(f"natural_gas_emission={data['gas_emission']:,.2f}")
        if data.get("electricity_emission") is not None:
            parts.append(f"electricity_emission={data['electricity_emission']:,.2f}")
        if data.get("direct_emissions") is not None:
            parts.append(f"direct_emissions={data['direct_emissions']:,.2f}")
        if data.get("per_capita") is not None:
            parts.append(f"per_capita={data['per_capita']:,.2f}")
        if data.get("per_household") is not None:
            parts.append(f"per_household={data['per_household']:,.2f}")
        if data.get("growth") is not None:
            parts.append(f"growth={float(data['growth']) * 100:.2f}%")
        contexts.append(", ".join(parts))

    return contexts


def generate_short_answer(
    query_text: str,
    session: RetrievalSession,
    language: str = "English",
    model: str | None = None,
) -> dict:
    _load_local_env()
    query_result = handle_query(query_text, session)
    if not query_result.get("retrieval_context"):
        query_result = {
            **query_result,
            "retrieval_context": format_retrieval_hits(session.search(query_text)),
        }
    contexts = build_answer_contexts(query_result)
    response = _generate_response(query_text, contexts, language=language, model=model)
    return {
        "query": query_text,
        "response": response,
        "contexts": contexts,
        "query_result": query_result,
    }


def _generate_response(
    query_text: str,
    contexts: list[str],
    language: str = "English",
    model: str | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _deterministic_fallback(query_text, contexts)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=float(os.getenv("OPENAI_TIMEOUT", 45)))
    system = (
        "You answer sustainability data questions using only the provided context. "
        f"Respond in {language}. "
        "Be concise and precise. "
        "Preserve numbers and units exactly when they are present in the context. "
        "If the question asks for consumption, answer with consumption; do not convert it into emissions. "
        "If the question asks for emissions, answer with emissions. "
        "Do not infer values that are not explicitly supported by the context. "
        "If the exact answer is unavailable, say that it is not available in the provided context."
    )
    user_payload = {
        "question": query_text,
        "contexts": contexts,
    }
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL") or "gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        max_tokens=min(int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", 800)), 300),
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


def _deterministic_fallback(query_text: str, contexts: list[str]) -> str:
    if not contexts:
        return "The requested information is not available in the provided context."
    return f"Based on the provided context, the answer should be derived from: {contexts[0][:240]}"
