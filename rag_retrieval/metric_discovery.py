"""Generic sustainability metric discovery and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .metric_registry import MetricDefinition
from .text import normalize_for_search, search_tokens


REPORT_SECTIONS = (
    "Emissions Overview",
    "Water Overview",
    "Resource Overview",
    "District Context and Sustainability Signals",
)

METRIC_ROLES = (
    "emission_input",
    "resource_kpi",
    "offset_or_sink",
    "context_indicator",
)


@dataclass(frozen=True, slots=True)
class DiscoveredMetric:
    metric_key: str
    display_name: str
    category: str
    unit: str
    role: str
    report_section: str
    sustainability_related: bool
    default_formula_support: bool
    default_factor_support: bool
    is_known_metric: bool
    source_kind: str
    source_column: str | None
    source_label: str | None
    metric_terms: tuple[str, ...]
    detection_method: str
    numeric_availability: float
    district_dimension: bool
    time_dimension: bool
    source_document_count: int = 1
    classification_source: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_key": self.metric_key,
            "display_name": self.display_name,
            "category": self.category,
            "unit": self.unit,
            "role": self.role,
            "report_section": self.report_section,
            "sustainability_related": self.sustainability_related,
            "default_formula_support": self.default_formula_support,
            "default_factor_support": self.default_factor_support,
            "is_known_metric": self.is_known_metric,
            "source_kind": self.source_kind,
            "source_column": self.source_column,
            "source_label": self.source_label,
            "metric_terms": list(self.metric_terms),
            "detection_method": self.detection_method,
            "numeric_availability": self.numeric_availability,
            "district_dimension": self.district_dimension,
            "time_dimension": self.time_dimension,
            "source_document_count": self.source_document_count,
            "classification_source": self.classification_source,
        }


_IGNORE_COLUMN_KEYS = {
    "value",
    "values",
    "page",
    "confidence",
    "source_document",
    "source",
    "row",
    "record",
    "month",
    "ay",
    "month_no",
    "ay_no",
    "district_code",
    "code",
    "id",
    "index",
}

_UNIT_TOKENS = {
    "kwh",
    "mwh",
    "m3",
    "m2",
    "m",
    "m²",
    "l",
    "lt",
    "liter",
    "litre",
    "liters",
    "litres",
    "ton",
    "tons",
    "tonne",
    "tonnes",
    "percent",
    "pct",
    "percentage",
    "tco2e",
    "co2e",
}

_GENERIC_TOKENS = {
    "metric",
    "indicator",
    "value",
    "values",
    "total",
    "annual",
    "monthly",
    "daily",
    "year",
    "years",
    "district",
    "ilce",
}

_TOKEN_CATEGORY_RULES: tuple[tuple[set[str], dict[str, object]], ...] = (
    (
        {"electricity", "natural", "gas", "diesel", "gasoline", "fuel", "energy", "heating", "carbon", "emission", "co2", "ghg"},
        {"category": "climate", "role": "emission_input", "report_section": "Emissions Overview", "sustainability_related": True},
    ),
    (
        {"water", "wastewater", "reservoir", "dam", "leakage", "reuse", "potable"},
        {"category": "water", "role": "resource_kpi", "report_section": "Water Overview", "sustainability_related": True},
    ),
    (
        {"waste", "recycling", "recycled", "compost", "landfill"},
        {"category": "waste", "role": "resource_kpi", "report_section": "Resource Overview", "sustainability_related": True},
    ),
    (
        {"mobility", "transport", "bus", "metro", "tram", "ridership", "fleet"},
        {"category": "mobility", "role": "context_indicator", "report_section": "District Context and Sustainability Signals", "sustainability_related": True},
    ),
    (
        {"tree", "forest", "green", "canopy", "park", "biodiversity", "soil"},
        {"category": "ecology", "role": "context_indicator", "report_section": "District Context and Sustainability Signals", "sustainability_related": True},
    ),
    (
        {"renewable", "solar", "wind", "pv"},
        {"category": "energy", "role": "context_indicator", "report_section": "District Context and Sustainability Signals", "sustainability_related": True},
    ),
    (
        {"air", "quality", "noise", "heat", "flood", "resilience", "risk", "occupancy", "availability"},
        {"category": "resilience", "role": "context_indicator", "report_section": "District Context and Sustainability Signals", "sustainability_related": True},
    ),
)


def discover_metrics(
    dataframe,
    *,
    registry: dict[str, MetricDefinition],
    district_column: str | None,
    time_column: str | None,
    metric_column: str | None,
    unit_column: str | None,
    value_column: str | None,
    metric_columns: dict[str, str | None],
    metric_aliases_for_key,
    metric_overrides: dict[str, dict[str, object]] | None = None,
) -> dict[str, DiscoveredMetric]:
    metrics: dict[str, DiscoveredMetric] = {}
    overrides = dict(metric_overrides or {})
    known_columns = {column for column in metric_columns.values() if column}

    for metric_key, definition in registry.items():
        source_column = metric_columns.get(metric_key)
        source_kind = "column" if source_column else "metric_row"
        detection_method = "registry_column" if source_column else "registry_alias"
        availability = _numeric_availability(dataframe, source_column) if source_column else _metric_row_availability(
            dataframe,
            metric_column,
            value_column,
            metric_aliases_for_key(metric_key, include_units=False),
        )
        metrics[metric_key] = _apply_overrides(
            DiscoveredMetric(
                metric_key=metric_key,
                display_name=metric_key.replace("_", " ").title(),
                category=definition.category,
                unit=definition.unit,
                role=definition.role,
                report_section=definition.report_section,
                sustainability_related=True,
                default_formula_support=definition.default_formula_support,
                default_factor_support=definition.default_factor_support,
                is_known_metric=True,
                source_kind=source_kind,
                source_column=source_column,
                source_label=metric_key.replace("_", " ").title(),
                metric_terms=metric_aliases_for_key(metric_key, include_units=False),
                detection_method=detection_method,
                numeric_availability=availability,
                district_dimension=bool(district_column),
                time_dimension=bool(time_column),
                classification_source="registry",
            ),
            overrides.get(metric_key),
        )

    for column in getattr(dataframe, "columns", []):
        if _skip_candidate_column(column, district_column, time_column, metric_column, unit_column, known_columns):
            continue
        availability = _numeric_availability(dataframe, column)
        if availability <= 0.0:
            continue
        metric_key = _canonical_metric_key(column)
        profile = _heuristic_profile(column, unit=_unit_from_name(column))
        metrics[metric_key] = _merge_metric(
            metrics.get(metric_key),
            _apply_overrides(
                DiscoveredMetric(
                    metric_key=metric_key,
                    display_name=_display_name(column),
                    category=profile["category"],
                    unit=profile["unit"],
                    role=profile["role"],
                    report_section=profile["report_section"],
                    sustainability_related=bool(profile["sustainability_related"]),
                    default_formula_support=False,
                    default_factor_support=False,
                    is_known_metric=metric_key in registry,
                    source_kind="column",
                    source_column=str(column),
                    source_label=_display_name(column),
                    metric_terms=(normalize_for_search(column),),
                    detection_method="column_candidate",
                    numeric_availability=availability,
                    district_dimension=bool(district_column),
                    time_dimension=bool(time_column),
                ),
                overrides.get(metric_key),
            ),
        )

    if metric_column and metric_column in getattr(dataframe, "columns", ()) and value_column and value_column in getattr(dataframe, "columns", ()):
        metric_values = dataframe[metric_column].dropna().astype(str)
        for raw_label in sorted({value.strip() for value in metric_values if value and value.strip()}):
            metric_key = _canonical_metric_key(raw_label)
            matching_rows = dataframe[metric_column].astype(str).map(normalize_for_search) == normalize_for_search(raw_label)
            availability = _numeric_availability(dataframe.loc[matching_rows], value_column)
            unit = _common_unit(dataframe.loc[matching_rows], unit_column) or _unit_from_name(raw_label)
            profile = _heuristic_profile(raw_label, unit=unit)
            metrics[metric_key] = _merge_metric(
                metrics.get(metric_key),
                _apply_overrides(
                    DiscoveredMetric(
                        metric_key=metric_key,
                        display_name=_display_name(raw_label),
                        category=profile["category"],
                        unit=profile["unit"],
                        role=profile["role"],
                        report_section=profile["report_section"],
                        sustainability_related=bool(profile["sustainability_related"]),
                        default_formula_support=False,
                        default_factor_support=False,
                        is_known_metric=metric_key in registry,
                        source_kind="metric_row",
                        source_column=None,
                        source_label=raw_label,
                        metric_terms=(normalize_for_search(raw_label),),
                        detection_method="metric_row_candidate",
                        numeric_availability=availability,
                        district_dimension=bool(district_column),
                        time_dimension=bool(time_column),
                    ),
                    overrides.get(metric_key),
                ),
            )

    return dict(sorted(metrics.items(), key=lambda item: (not item[1].is_known_metric, item[1].display_name.casefold())))


def normalize_metric_override(metric_key: str, payload: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {"metric_key": metric_key}
    if "display_name" in payload:
        value = str(payload.get("display_name") or "").strip()
        if value:
            normalized["display_name"] = value
    if "unit" in payload:
        normalized["unit"] = str(payload.get("unit") or "").strip()
    if "category" in payload:
        value = normalize_for_search(payload.get("category")).replace(" ", "_")
        normalized["category"] = value or "other"
    if "role" in payload:
        value = normalize_for_search(payload.get("role")).replace(" ", "_")
        normalized["role"] = value if value in METRIC_ROLES else "context_indicator"
    if "report_section" in payload:
        value = str(payload.get("report_section") or "").strip()
        normalized["report_section"] = value or REPORT_SECTIONS[-1]
    if "sustainability_related" in payload:
        normalized["sustainability_related"] = bool(payload.get("sustainability_related"))
    return normalized


def merge_metric_overrides(*layers: dict[str, dict[str, object]] | None) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for layer in layers:
        for metric_key, payload in (layer or {}).items():
            current = dict(merged.get(metric_key, {}))
            current.update(payload or {})
            merged[metric_key] = current
    return merged


def summarize_metric_for_api(metric: DiscoveredMetric, *, documents: list[str], used_in_calculation: bool = False, used_in_narrative: bool = False) -> dict[str, object]:
    payload = metric.to_dict()
    payload.update(
        {
            "documents": sorted({document for document in documents if document}),
            "used_in_calculation": bool(used_in_calculation),
            "used_in_narrative": bool(used_in_narrative),
        }
    )
    return payload


def _apply_overrides(metric: DiscoveredMetric, override: dict[str, object] | None) -> DiscoveredMetric:
    if not override:
        return metric
    updates = {}
    for key in ("display_name", "category", "unit", "role", "report_section", "sustainability_related"):
        if key in override and override.get(key) not in (None, ""):
            updates[key] = override.get(key)
    if updates:
        updates["classification_source"] = str(override.get("classification_source") or "user")
        return replace(metric, **updates)
    return metric


def _merge_metric(current: DiscoveredMetric | None, candidate: DiscoveredMetric) -> DiscoveredMetric:
    if current is None:
        return candidate
    if current.is_known_metric and not candidate.is_known_metric:
        return current
    if candidate.is_known_metric and not current.is_known_metric:
        return candidate
    if candidate.numeric_availability > current.numeric_availability:
        merged = candidate
    else:
        merged = current
    return replace(
        merged,
        sustainability_related=current.sustainability_related or candidate.sustainability_related,
        source_document_count=max(current.source_document_count, candidate.source_document_count),
    )


def _skip_candidate_column(
    column: str,
    district_column: str | None,
    time_column: str | None,
    metric_column: str | None,
    unit_column: str | None,
    known_columns: set[str],
) -> bool:
    if column in {district_column, time_column, metric_column, unit_column}:
        return True
    if column in known_columns:
        return True
    normalized = normalize_for_search(column).replace(" ", "_")
    if normalized in _IGNORE_COLUMN_KEYS:
        return True
    if normalized.startswith("source_") or normalized.endswith("_id"):
        return True
    return False


def _heuristic_profile(name: str, unit: str | None = None) -> dict[str, object]:
    normalized = normalize_for_search(name)
    tokens = set(search_tokens(normalized))
    token_set = {token for token in tokens if token not in _GENERIC_TOKENS and token not in _UNIT_TOKENS}
    profile = {
        "category": "other",
        "unit": _clean_unit(unit) or _unit_from_name(name),
        "role": "context_indicator",
        "report_section": "District Context and Sustainability Signals",
        "sustainability_related": False,
    }

    for trigger_tokens, updates in _TOKEN_CATEGORY_RULES:
        if trigger_tokens & token_set:
            profile.update(updates)
            break

    if not profile["sustainability_related"]:
        if {"count", "rate", "share", "ratio", "area", "occupancy", "index", "score"} & token_set and token_set:
            profile["sustainability_related"] = bool(
                token_set
                & {
                    "tree", "green", "recycling", "renewable", "dam", "reservoir", "water", "wastewater",
                    "waste", "air", "quality", "mobility", "transport", "fleet", "heat", "flood",
                }
            )
        elif {"tree", "green", "recycling", "renewable", "dam", "reservoir", "water", "waste", "wastewater"} & token_set:
            profile["sustainability_related"] = True

    if profile["category"] == "water" and profile["role"] == "resource_kpi":
        if {"occupancy", "availability", "risk"} & token_set:
            profile["role"] = "context_indicator"
    if {"tree", "forest", "green"} & token_set and {"co2", "carbon", "sink", "offset"} & token_set:
        profile["role"] = "offset_or_sink"
        profile["report_section"] = "Emissions Overview"

    return profile


def _numeric_availability(dataframe, column: str | None) -> float:
    if dataframe is None or column is None or column not in getattr(dataframe, "columns", ()):
        return 0.0
    try:
        import pandas as pd  # noqa: F401
    except ImportError:  # pragma: no cover
        return 0.0
    series = dataframe[column]
    if series.empty:
        return 0.0
    numeric = series.astype(str).str.replace(",", "", regex=False)
    values = series.to_frame().assign(_clean=numeric)
    parsed = values["_clean"]
    converted = parsed.astype(str)
    numbers = converted.map(_parse_floatable)
    return float(numbers.notna().sum()) / max(len(series), 1)


def _metric_row_availability(dataframe, metric_column: str | None, value_column: str | None, aliases: tuple[str, ...]) -> float:
    if (
        dataframe is None
        or not metric_column
        or not value_column
        or metric_column not in getattr(dataframe, "columns", ())
        or value_column not in getattr(dataframe, "columns", ())
    ):
        return 0.0
    normalized_metric = dataframe[metric_column].astype(str).map(normalize_for_search)
    alias_set = {normalize_for_search(alias) for alias in aliases if alias}
    mask = normalized_metric.apply(lambda value: _matches_alias_set(value, alias_set))
    if not mask.any():
        return 0.0
    return _numeric_availability(dataframe.loc[mask], value_column)


def _matches_alias_set(metric_value: str, aliases: set[str]) -> bool:
    normalized = normalize_for_search(metric_value)
    tokens = set(search_tokens(normalized))
    for alias in aliases:
        alias_tokens = set(search_tokens(alias))
        if alias and alias in normalized:
            return True
        if alias_tokens and alias_tokens <= tokens:
            return True
    return False


def _parse_floatable(value: object):
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


def _common_unit(dataframe, unit_column: str | None) -> str:
    if dataframe is None or not unit_column or unit_column not in getattr(dataframe, "columns", ()):
        return ""
    units = [str(value).strip() for value in dataframe[unit_column].dropna().astype(str) if str(value).strip()]
    if not units:
        return ""
    return max(set(units), key=units.count)


def _display_name(raw_value: str) -> str:
    parts = [token for token in re.split(r"[_\s]+", str(raw_value).strip()) if token]
    if not parts:
        return "Metric"
    rendered = []
    for token in parts:
        if token.lower() in {"kwh", "mwh", "m3", "co2", "co2e", "tco2e"}:
            rendered.append(token.upper())
        else:
            rendered.append(token.capitalize())
    return " ".join(rendered)


def _canonical_metric_key(raw_value: str) -> str:
    normalized = normalize_for_search(raw_value)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    normalized = re.sub(r"_(?:kwh|mwh|m3|m2|l|lt|percent|pct|percentage|ton|tonne|tonnes|tco2e|co2e)$", "", normalized)
    normalized = re.sub(r"^(?:total|annual|monthly)_", "", normalized)
    return normalized or "metric"


def _unit_from_name(raw_value: str) -> str:
    normalized = normalize_for_search(raw_value)
    if "kwh" in normalized:
        return "kWh"
    if "mwh" in normalized:
        return "MWh"
    if "m3" in normalized:
        return "m3"
    if "m2" in normalized or "m²" in raw_value:
        return "m2"
    if "tco2e" in normalized:
        return "tCO2e"
    if "percent" in normalized or "pct" in normalized or "occupancy" in normalized or "rate" in normalized or "share" in normalized:
        return "%"
    if any(token in normalized for token in ("liter", "litre", "liters", "litres", "_l", " l")):
        return "L"
    if any(token in normalized for token in ("ton", "tonne", "tonnes")):
        return "t"
    return ""


def _clean_unit(unit: str | None) -> str:
    if not unit:
        return ""
    text = str(unit).strip()
    lowered = normalize_for_search(text).replace(" ", "")
    if lowered in {"kwh", "kw/h"}:
        return "kWh"
    if lowered == "mwh":
        return "MWh"
    if lowered in {"m3", "m^3"}:
        return "m3"
    if lowered in {"m2", "m^2"}:
        return "m2"
    if lowered in {"l", "lt", "liter", "litre", "liters", "litres"}:
        return "L"
    if lowered in {"t", "ton", "tonne", "tonnes"}:
        return "t"
    if lowered in {"percent", "pct", "%"}:
        return "%"
    if lowered in {"tco2e", "co2e"}:
        return "tCO2e"
    return text
