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

    # Expose the formula extraction outcome so the assistant can reference it
    custom_formula = getattr(session, "custom_formula", None)
    formula_method = getattr(session, "formula_extraction_method", "default")
    formula_status = getattr(session, "custom_formula_status", "default")
    formula_missing = list(getattr(session, "custom_formula_missing_variables", []) or [])
    if custom_formula:
        context_parts.append(
            f"Custom emission formula extracted from documents "
            f"(method: {formula_method}):\n"
            f"  expression: {custom_formula.expression}\n"
            f"  constants: {custom_formula.constants}\n"
            f"  source: \"{custom_formula.source_text}\""
        )
        if formula_status in {"incomplete", "partial"} and formula_missing:
            context_parts.append(
                "Important: the uploaded custom formula cannot be safely applied yet. "
                f"Missing variables: {', '.join(formula_missing)}. "
                "Ask the user to define each missing variable as either a spreadsheet column "
                "or a fixed constant before relying on total emissions."
            )
    elif formula_method == "default":
        context_parts.append(
            "Note: No custom emission formula was found in the uploaded documents. "
            "Calculations use the default formula: "
            "total_CO2 = electricity_consumption * electricity_factor + natural_gas_consumption * natural_gas_factor."
        )

    # Collect emission factors found across all documents (session-wide)
    doc_factors = getattr(session, "document_emission_factors", {})
    if doc_factors:
        factor_lines = []
        for doc_id, factors in doc_factors.items():
            doc = (getattr(session, "documents", {}) or {}).get(doc_id)
            fname = doc.filename if doc else doc_id
            for key, val in factors.items():
                factor_lines.append(f"  {key}: {val} (from {fname})")
        context_parts.append("Emission factors extracted from uploaded documents:\n" + "\n".join(factor_lines))

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
        if data.get("water_consumption") is not None and float(data.get("water_consumption") or 0.0) > 0.0:
            parts.append(f"water_consumption={float(data['water_consumption']):,.2f} m3")
        if data.get("water_per_capita") is not None:
            parts.append(f"water_per_capita={float(data['water_per_capita']):,.2f}")
        if data.get("water_growth") is not None:
            parts.append(f"water_growth={float(data['water_growth']) * 100:.1f}%")
        if data.get("growth") is not None:
            parts.append(f"growth={float(data['growth']) * 100:.1f}%")
        extra_metrics = []
        for metric_key, summary in sorted((data.get("metrics") or {}).items()):
            if metric_key in {"electricity", "natural_gas", "water"}:
                continue
            value = float(summary.get("value") or 0.0)
            if value <= 0.0 or not summary.get("sustainability_related", True):
                continue
            unit = str(summary.get("unit") or "").strip()
            extra_metrics.append(f"{metric_key}={value:,.2f}{(' ' + unit) if unit else ''}")
        if extra_metrics:
            parts.append("additional_metrics=" + "; ".join(extra_metrics[:4]))
        factors_used = data.get("emission_factors_used") or {}
        factors_src = data.get("emission_factors_source") or {}
        if factors_used:
            for key, val in factors_used.items():
                src = factors_src.get(key, "reference")
                parts.append(f"{key}_factor={val} (source: {src})")
        formula_item_status = data.get("formula_status")
        if formula_item_status:
            parts.append(f"formula_status={formula_item_status}")
        missing_variables = data.get("formula_missing_variables") or []
        if missing_variables:
            parts.append(f"formula_missing_variables={','.join(missing_variables)}")
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
