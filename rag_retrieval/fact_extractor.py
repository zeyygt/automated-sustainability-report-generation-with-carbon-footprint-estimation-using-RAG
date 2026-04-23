"""Regex-based fact extraction from parsed document text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import ElementType, ParsedDocument
from .text import normalize_for_search


TEXT_ELEMENT_TYPES = {ElementType.PARAGRAPH, ElementType.LIST_ITEM}
VALUE_UNIT_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)*(?:\.\d+)?)\s*"
    r"(?P<scale>million|milyon|thousand|bin)?\s*"
    r"(?P<unit>m3|m³|kwh|mwh|tco2e|ton|tons|tonnes)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


class FactExtractor:
    """Extract deterministic metric facts from parsed PDF/text paragraphs."""

    def __init__(self, districts: list[str] | None = None):
        self.districts = districts or _reference_districts()

    def extract(self, parsed_document: ParsedDocument) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        for element in parsed_document.elements:
            if element.element_type not in TEXT_ELEMENT_TYPES:
                continue
            for sentence in _split_sentences(element.text):
                facts.extend(self._extract_from_sentence(sentence, parsed_document, element.page))
        return facts

    def to_dataframe(self, parsed_document: ParsedDocument):
        import pandas as pd

        return pd.DataFrame(self.extract(parsed_document))

    def _extract_from_sentence(self, sentence: str, parsed_document: ParsedDocument, page: int) -> list[dict[str, Any]]:
        normalized = normalize_for_search(sentence)
        district = self._find_district(normalized)
        metric = _detect_metric(normalized)
        years = YEAR_RE.findall(normalized)
        year = int(years[-1]) if years else None
        if not district or not metric:
            return []

        facts: list[dict[str, Any]] = []
        for match in VALUE_UNIT_RE.finditer(sentence):
            value = _parse_value(match.group("value"), match.group("scale"))
            unit = _normalize_unit(match.group("unit"))
            if value is None:
                continue
            fact: dict[str, Any] = {
                "district": district,
                "year": year,
                "metric": metric,
                "value": value,
                "unit": unit,
                "source_document": parsed_document.filename,
                "page": page,
                "confidence": "pattern",
            }
            if metric == "natural_gas_consumption":
                fact["natural_gas_consumption_m3"] = value
            elif metric == "electricity_consumption":
                fact["electricity_consumption_kwh"] = value * 1000 if unit == "mwh" else value
            elif metric == "emissions":
                fact["emissions"] = value
            facts.append(fact)
        return facts

    def _find_district(self, normalized_sentence: str) -> str | None:
        for district in self.districts:
            if normalize_for_search(district) in normalized_sentence:
                return district
        return None


def _detect_metric(normalized_text: str) -> str | None:
    if any(term in normalized_text for term in ("electricity", "elektrik", "kwh", "mwh")):
        return "electricity_consumption"
    if any(term in normalized_text for term in ("natural gas", "dogalgaz", "gas", "m3", "m 3")):
        return "natural_gas_consumption"
    if any(term in normalized_text for term in ("emission", "emissions", "tco2e", "carbon", "co2")):
        return "emissions"
    return None


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _parse_value(value: str, scale: str | None) -> float | None:
    normalized = _normalize_number_text(value)
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if scale and normalize_for_search(scale) in {"million", "milyon"}:
        parsed *= 1_000_000
    elif scale and normalize_for_search(scale) in {"thousand", "bin"}:
        parsed *= 1_000
    return parsed


def _normalize_number_text(value: str) -> str:
    text = value.strip()
    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")
    if "," in text:
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            return text.replace(",", ".")
        return text.replace(",", "")
    if text.count(".") > 1:
        return text.replace(".", "")
    return text


def _normalize_unit(unit: str) -> str:
    normalized = normalize_for_search(unit).replace(" ", "")
    if normalized == "m":
        return "m3"
    return normalized


def _reference_districts() -> list[str]:
    path = Path(__file__).resolve().parent / "reference_Data.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("districts", {}).keys())
