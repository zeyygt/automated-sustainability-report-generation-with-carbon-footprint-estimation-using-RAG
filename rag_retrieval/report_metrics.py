"""Public-facing metric preparation for reports."""

from __future__ import annotations

from .text import normalize_for_search


def public_metrics(structured_results: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in structured_results:
        data = item.get("data", {})
        district = data.get("district")
        if not district:
            continue
        key = normalize_for_search(str(district))
        metric = {
            "district": str(district),
            "total_emission": float(data.get("total_emission") or 0.0),
            "gas_emission": float(data.get("gas_emission") or 0.0),
            "electricity_emission": float(data.get("electricity_emission") or 0.0),
            "direct_emissions": float(data.get("direct_emissions") or 0.0),
            "per_capita": data.get("per_capita"),
            "per_household": data.get("per_household"),
            "growth": data.get("growth"),
            "growth_percent": _growth_percent(data.get("growth")),
            "growth_display": _growth_display(data.get("growth")),
            "warnings": list(data.get("warnings", []) or ()),
        }
        current = grouped.get(key)
        if current is None or metric["total_emission"] > current["total_emission"]:
            grouped[key] = metric
        else:
            current["warnings"] = sorted(set([*current.get("warnings", []), *metric.get("warnings", [])]))

    return sorted(grouped.values(), key=lambda value: value["total_emission"], reverse=True)


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
