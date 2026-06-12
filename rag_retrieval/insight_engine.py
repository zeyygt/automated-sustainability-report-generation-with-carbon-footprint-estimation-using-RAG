"""Deterministic municipal insight extraction for sustainability reporting."""

from __future__ import annotations

from statistics import median
from typing import Any


def build_report_insights(
    metrics: list[dict[str, Any]],
    *,
    warnings: list[str] | None = None,
    detected_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    detected_metrics = list(detected_metrics or [])
    emission_metrics = [item for item in metrics if float(item.get("total_emission") or 0.0) > 0.0]
    water_metrics = [item for item in metrics if float(item.get("water_consumption") or 0.0) > 0.0]
    growth_metrics = [item for item in metrics if item.get("growth") is not None]

    highest_emission = _top_n(emission_metrics, "total_emission", 5)
    highest_water = _top_n(water_metrics, "water_consumption", 5)
    fastest_growth = _top_n(growth_metrics, "growth", 5)
    improving = _bottom_n(growth_metrics, "growth", 5)
    context_highlights = _context_highlights(metrics, limit=6)
    pressure_districts = _pressure_districts(metrics)
    low_coverage = _coverage_flags(metrics, warnings)
    outliers = _outliers(metrics)

    municipality = {
        "district_count": len(metrics),
        "emission_district_count": len(emission_metrics),
        "water_district_count": len(water_metrics),
        "total_emissions": _sum_metric(emission_metrics, "total_emission"),
        "total_water_consumption": _sum_metric(water_metrics, "water_consumption"),
        "median_growth": _median_metric(growth_metrics, "growth"),
        "highest_emission_district": highest_emission[0] if highest_emission else None,
        "highest_water_district": highest_water[0] if highest_water else None,
        "coverage_level": _coverage_level(metrics, warnings, detected_metrics),
    }

    headlines = _headline_statements(
        municipality,
        highest_emission=highest_emission,
        highest_water=highest_water,
        fastest_growth=fastest_growth,
        pressure_districts=pressure_districts,
        context_highlights=context_highlights,
        warnings=warnings,
    )

    return {
        "municipality": municipality,
        "headlines": headlines,
        "priority_districts": pressure_districts,
        "highest_emission_districts": highest_emission,
        "highest_water_districts": highest_water,
        "fastest_growth_districts": fastest_growth,
        "improving_districts": improving,
        "context_highlights": context_highlights,
        "outliers": outliers,
        "coverage": {
            "level": municipality["coverage_level"],
            "flags": low_coverage,
            "warning_count": len(warnings),
            "detected_metric_count": len(detected_metrics),
        },
    }


def _headline_statements(
    municipality: dict[str, Any],
    *,
    highest_emission: list[dict[str, Any]],
    highest_water: list[dict[str, Any]],
    fastest_growth: list[dict[str, Any]],
    pressure_districts: list[dict[str, Any]],
    context_highlights: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    lines: list[str] = []
    district_count = municipality.get("district_count") or 0
    if district_count:
        lines.append(
            f"The assessment covers {district_count} districts with a combined emissions estimate of {float(municipality.get('total_emissions') or 0.0):,.2f}."
        )
    if highest_emission:
        top = highest_emission[0]
        lines.append(
            f"{top['district']} currently has the highest total emissions in the uploaded dataset at {float(top['value']):,.2f}."
        )
    if highest_water:
        top = highest_water[0]
        lines.append(
            f"{top['district']} also leads water demand at {float(top['value']):,.2f} m3, which should be considered alongside emissions planning."
        )
    if fastest_growth:
        top = fastest_growth[0]
        lines.append(
            f"{top['district']} shows the fastest reported growth trend at {float(top['value']) * 100:.2f}%."
        )
    if pressure_districts:
        districts = ", ".join(item["district"] for item in pressure_districts[:3])
        lines.append(f"Priority districts for closer operational review include {districts}.")
    if context_highlights:
        top = context_highlights[0]
        lines.append(
            f"Contextual sustainability signals are also present; for example, {top['district']} stands out on {top['metric_label']}."
        )
    if warnings:
        lines.append("Some findings remain directional because parts of the uploaded dataset are incomplete or unevenly covered.")
    return lines


def _pressure_districts(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    emission_values = [float(item.get("total_emission") or 0.0) for item in metrics if float(item.get("total_emission") or 0.0) > 0.0]
    water_values = [float(item.get("water_consumption") or 0.0) for item in metrics if float(item.get("water_consumption") or 0.0) > 0.0]
    emission_threshold = median(emission_values) if emission_values else 0.0
    water_threshold = median(water_values) if water_values else 0.0
    for item in metrics:
        score = 0.0
        total = float(item.get("total_emission") or 0.0)
        water = float(item.get("water_consumption") or 0.0)
        growth = float(item.get("growth") or 0.0) if item.get("growth") is not None else 0.0
        if total >= emission_threshold and total > 0.0:
            score += 2.0
        if water >= water_threshold and water > 0.0:
            score += 1.5
        if growth > 0:
            score += min(growth * 5.0, 1.5)
        if item.get("warnings"):
            score += 0.2
        if score <= 0.0:
            continue
        scored.append(
            {
                "district": item.get("district"),
                "score": round(score, 3),
                "total_emission": total,
                "water_consumption": water,
                "growth": item.get("growth"),
            }
        )
    return sorted(scored, key=lambda item: (item["score"], item["total_emission"], item["water_consumption"]), reverse=True)[:8]


def _context_highlights(metrics: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for item in metrics:
        for metric_key, summary in sorted((item.get("metric_summaries") or {}).items()):
            if metric_key in {"electricity", "natural_gas", "water"}:
                continue
            if summary.get("report_section") != "District Context and Sustainability Signals":
                continue
            value = float(summary.get("value") or 0.0)
            if value <= 0.0:
                continue
            highlights.append(
                {
                    "district": item.get("district"),
                    "metric_key": metric_key,
                    "metric_label": summary.get("label") or metric_key.replace("_", " ").title(),
                    "value": value,
                    "unit": summary.get("unit") or "",
                    "role": summary.get("role"),
                }
            )
    return sorted(highlights, key=lambda item: item["value"], reverse=True)[:limit]


def _coverage_flags(metrics: list[dict[str, Any]], warnings: list[str]) -> list[str]:
    flags = []
    if not metrics:
        flags.append("no_structured_metrics")
    if any("electricity_consumption_not_found" in warning for warning in warnings):
        flags.append("partial_electricity_coverage")
    if any("natural_gas_consumption_not_found" in warning for warning in warnings):
        flags.append("partial_natural_gas_coverage")
    if any("water_consumption_not_found" in warning for warning in warnings):
        flags.append("partial_water_coverage")
    if any("custom_formula_missing" in warning for warning in warnings):
        flags.append("custom_formula_incomplete")
    if any("population_reference_not_found" in warning for warning in warnings):
        flags.append("population_reference_gap")
    return flags


def _coverage_level(metrics: list[dict[str, Any]], warnings: list[str], detected_metrics: list[dict[str, Any]]) -> str:
    if not metrics:
        return "low"
    serious = _coverage_flags(metrics, warnings)
    sustainability_metric_count = sum(1 for metric in detected_metrics if metric.get("sustainability_related"))
    if not serious and sustainability_metric_count >= 3:
        return "high"
    if len(serious) <= 2:
        return "moderate"
    return "low"


def _outliers(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for field, label in (
        ("total_emission", "total emissions"),
        ("water_consumption", "water consumption"),
        ("growth", "growth"),
    ):
        valid = [
            (item.get("district"), float(item.get(field) or 0.0))
            for item in metrics
            if item.get(field) is not None and float(item.get(field) or 0.0) > 0.0
        ]
        if len(valid) < 3:
            continue
        values = [value for _, value in valid]
        threshold = median(values) * 1.5
        for district, value in valid:
            if value > threshold:
                results.append(
                    {
                        "district": district,
                        "metric": field,
                        "label": label,
                        "value": value,
                    }
                )
    return results[:12]


def _top_n(items: list[dict[str, Any]], field: str, count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "district": item.get("district"),
                "value": float(item.get(field) or 0.0),
                "growth": item.get("growth"),
            }
            for item in items
        ),
        key=lambda item: item["value"],
        reverse=True,
    )
    return [item for item in ranked if item["value"] > 0.0][:count]


def _bottom_n(items: list[dict[str, Any]], field: str, count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {
                "district": item.get("district"),
                "value": float(item.get(field) or 0.0),
                "growth": item.get("growth"),
            }
            for item in items
        ),
        key=lambda item: item["value"],
    )
    return ranked[:count]


def _sum_metric(items: list[dict[str, Any]], field: str) -> float:
    return float(sum(float(item.get(field) or 0.0) for item in items))


def _median_metric(items: list[dict[str, Any]], field: str) -> float | None:
    values = [float(item.get(field) or 0.0) for item in items if item.get(field) is not None]
    if not values:
        return None
    return float(median(values))
