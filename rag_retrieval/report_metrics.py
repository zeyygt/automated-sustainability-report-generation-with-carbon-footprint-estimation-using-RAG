"""Public-facing metric preparation for sustainability reports."""

from __future__ import annotations

from .metric_registry import metric_registry
from .text import normalize_for_search


def public_metrics(structured_results: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}

    for item in structured_results:
        data = item.get("data", {})
        district = data.get("district")
        if not district:
            continue

        key = normalize_for_search(str(district))
        current = grouped.setdefault(
            key,
            {
                "district": str(district),
                "total_emission": 0.0,
                "gas_emission": 0.0,
                "electricity_emission": 0.0,
                "direct_emissions": 0.0,
                "emission_unit": "kgCO2e",
                "population": None,
                "per_capita": None,
                "per_household": None,
                "growth": None,
                "warnings": [],
                "metric_summaries": {},
            },
        )

        current["district"] = str(district)
        _merge_top_level_numeric(current, data, "total_emission")
        _merge_top_level_numeric(current, data, "gas_emission")
        _merge_top_level_numeric(current, data, "electricity_emission")
        _merge_top_level_numeric(current, data, "direct_emissions")
        _merge_optional_numeric(current, data, "population")
        _merge_optional_numeric(current, data, "per_capita")
        _merge_optional_numeric(current, data, "per_household")
        _merge_optional_numeric(current, data, "growth")
        if data.get("emission_unit"):
            current["emission_unit"] = str(data.get("emission_unit"))
        current["warnings"] = sorted(set([*current.get("warnings", []), *(data.get("warnings", []) or [])]))

        all_metrics = dict(data.get("metrics") or {})
        for metric_key, definition in metric_registry().items():
            if metric_key not in all_metrics:
                legacy_metric = _legacy_metric_summary(metric_key, definition, data)
                if legacy_metric is not None:
                    all_metrics[metric_key] = legacy_metric

        for metric_key, metric_data in all_metrics.items():
            if metric_data is None:
                continue
            current_summary = current["metric_summaries"].get(metric_key)
            current["metric_summaries"][metric_key] = _merge_metric_summary(current_summary, metric_data)

    values = []
    for item in grouped.values():
        water = item["metric_summaries"].get("water", {})
        item["water_consumption"] = float(water.get("value") or 0.0)
        item["water_per_capita"] = water.get("per_capita")
        item["water_growth"] = water.get("growth")
        item["available_metric_keys"] = sorted(
            metric_key
            for metric_key, summary in item["metric_summaries"].items()
            if float(summary.get("value") or 0.0) > 0.0 or summary.get("growth") is not None
        )
        item["growth_percent"] = _growth_percent(item.get("growth"))
        item["growth_display"] = _growth_display(item.get("growth"))
        values.append(item)

    return sorted(values, key=_sort_key, reverse=True)


def _merge_top_level_numeric(target: dict, source: dict, key: str) -> None:
    current = float(target.get(key) or 0.0)
    candidate = float(source.get(key) or 0.0)
    if abs(candidate) > abs(current):
        target[key] = candidate


def _merge_optional_numeric(target: dict, source: dict, key: str) -> None:
    candidate = source.get(key)
    current = target.get(key)
    if candidate is None:
        return
    if current is None:
        target[key] = candidate
        return
    try:
        if abs(float(candidate)) > abs(float(current)):
            target[key] = candidate
    except (TypeError, ValueError):
        target[key] = candidate


def _legacy_metric_summary(metric_key: str, definition, data: dict) -> dict | None:
    if metric_key == "water" and data.get("water_consumption") is not None:
        return {
            "metric_key": metric_key,
            "label": "Water",
            "category": definition.category,
            "unit": definition.unit,
            "role": definition.role,
            "report_section": definition.report_section,
            "value": float(data.get("water_consumption") or 0.0),
            "per_capita": data.get("water_per_capita"),
            "growth": data.get("water_growth"),
        }
    return None


def _merge_metric_summary(current: dict | None, candidate: dict) -> dict:
    merged = dict(current or {})
    for key in ("metric_key", "label", "category", "unit", "role", "report_section", "source_column", "sustainability_related", "is_known_metric", "source_kind"):
        if key in candidate and candidate.get(key) not in (None, ""):
            merged[key] = candidate.get(key)

    current_value = float(merged.get("value") or 0.0)
    candidate_value = float(candidate.get("value") or 0.0)
    if abs(candidate_value) > abs(current_value):
        merged["value"] = candidate_value
        merged["per_capita"] = candidate.get("per_capita")
        merged["growth"] = candidate.get("growth")
    else:
        if merged.get("per_capita") is None and candidate.get("per_capita") is not None:
            merged["per_capita"] = candidate.get("per_capita")
        if merged.get("growth") is None and candidate.get("growth") is not None:
            merged["growth"] = candidate.get("growth")
        merged.setdefault("value", current_value)

    return merged


def _sort_key(item: dict) -> tuple[float, ...]:
    total_emission = float(item.get("total_emission") or 0.0)
    water = float(item.get("water_consumption") or 0.0)
    gas = float(item.get("gas_emission") or 0.0)
    electricity = float(item.get("electricity_emission") or 0.0)
    custom_values = [
        float(summary.get("value") or 0.0)
        for metric_key, summary in (item.get("metric_summaries") or {}).items()
        if metric_key not in {"electricity", "natural_gas", "water"}
    ]
    max_custom = max(custom_values, default=0.0)
    return (
        1 if total_emission > 0.0 else 0,
        total_emission,
        water,
        gas + electricity,
        max_custom,
    )


def _growth_percent(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 100
    except (TypeError, ValueError):
        return None


def _growth_display(value) -> str:
    percent = _growth_percent(value)
    return "" if percent is None else f"{percent:.2f}%"
