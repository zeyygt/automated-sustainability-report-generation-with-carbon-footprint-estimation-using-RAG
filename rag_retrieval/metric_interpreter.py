"""Optional LLM-assisted interpretation for discovered sustainability metrics."""

from __future__ import annotations

import json
import os
from typing import Iterable

from .metric_discovery import normalize_metric_override


def suggest_metric_overrides(
    metrics: Iterable[dict[str, object]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 20.0,
) -> dict[str, dict[str, object]]:
    candidates = [
        metric for metric in metrics
        if not metric.get("is_known_metric")
        and metric.get("classification_source") != "user"
    ]
    if not candidates:
        return {}

    key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if not key:
        return {}

    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=timeout)
    payload = [
        {
            "metric_key": metric.get("metric_key"),
            "display_name": metric.get("display_name"),
            "unit": metric.get("unit"),
            "source_kind": metric.get("source_kind"),
            "district_dimension": metric.get("district_dimension"),
            "time_dimension": metric.get("time_dimension"),
        }
        for metric in candidates
    ]
    system = (
        "You classify municipal sustainability metrics. "
        "Return strict JSON with an object that has a single key 'metrics'. "
        "Each item must contain: metric_key, sustainability_related, category, role, report_section. "
        "Allowed roles: emission_input, resource_kpi, offset_or_sink, context_indicator. "
        "Allowed report sections: Emissions Overview, Water Overview, Resource Overview, District Context and Sustainability Signals. "
        "Be conservative: if a metric is not clearly sustainability-related, mark sustainability_related false."
    )
    response = client.chat.completions.create(
        model=model or os.getenv("OPENAI_MODEL") or "gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"metrics": payload}, ensure_ascii=False)},
        ],
        max_tokens=700,
        temperature=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    parsed = _load_json_object(content)
    suggestions = parsed.get("metrics", []) if isinstance(parsed, dict) else []
    normalized: dict[str, dict[str, object]] = {}
    for suggestion in suggestions:
        metric_key = str(suggestion.get("metric_key") or "").strip()
        if not metric_key:
            continue
        override = normalize_metric_override(metric_key, suggestion)
        override["classification_source"] = "llm"
        normalized[metric_key] = override
    return normalized


def _load_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}
