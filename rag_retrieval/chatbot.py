"""Conversational chat response streaming around the RAG pipeline."""

from __future__ import annotations

import os
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import RetrievalSession


def stream_chat_response(
    message: str,
    session: "RetrievalSession",
    chat_history: list[dict],
    report_markdown: str | None = None,
    language: str = "English",
    model: str | None = None,
) -> Generator[str, None, None]:
    """Yield response tokens for a conversational question grounded in RAG context."""
    from .pipeline import handle_query
    from .generation import _load_local_env

    _load_local_env()
    api_key = os.getenv("OPENAI_API_KEY")

    result = handle_query(message, session)
    context_parts: list[str] = []

    for item in (result.get("retrieval_context") or [])[:6]:
        text = (item.get("text") or "").strip()
        if text:
            context_parts.append(text)

    for item in (result.get("structured_results") or [])[:5]:
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
        if data.get("growth") is not None:
            parts.append(f"growth={float(data['growth']) * 100:.1f}%")
        context_parts.append(", ".join(parts))

    context = "\n\n".join(context_parts)
    is_english = language.strip().lower().startswith("en")
    lang_note = (
        "Respond in English."
        if is_english
        else f"IMPORTANT: You MUST respond entirely in {language}. Every word of your response must be in {language}, regardless of the language of the question or the context."
    )

    system = (
        "You are a sustainability data assistant for Istanbul Metropolitan Municipality (IBB). "
        "Answer questions using the provided document context and district data. "
        "Be precise with numbers. Use markdown tables and bullet lists when helpful. "
        "Do not mention RAG, chunks, embeddings, parsers, or technical pipeline details. "
        f"{lang_note}"
    )
    if context:
        system += f"\n\nRelevant context from uploaded documents:\n{context}"
    if report_markdown:
        system += f"\n\nPreviously generated sustainability report:\n{report_markdown[:3000]}"

    if not api_key:
        yield (
            "⚠️ No OpenAI API key found. Please add `OPENAI_API_KEY` to your `.env` file "
            "and restart the server."
        )
        return

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=float(os.getenv("OPENAI_TIMEOUT", 45)))
    recent = (chat_history or [])[-10:]

    stream = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL") or "gpt-4o",
        messages=[
            {"role": "system", "content": system},
            *recent,
            {"role": "user", "content": message},
        ],
        stream=True,
        max_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", 800)),
        temperature=0.3,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
