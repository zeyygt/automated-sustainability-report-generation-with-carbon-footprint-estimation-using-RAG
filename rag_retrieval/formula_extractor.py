"""Extract emission factors from parsed documents and spreadsheet DataFrames."""

from __future__ import annotations

import re
from typing import Any

from .models import ParsedDocument
from .text import normalize_for_search


# --- keyword helpers ---

_ELEC_KW = ("electricity", "elektrik", "electric", "kwh", "mwh")
_GAS_KW = ("natural gas", "natural_gas", "dogalgaz", "doğalgaz", "gas", "m3", "m³")
_FACTOR_KW = ("emission factor", "emisyon fakt", "co2 factor", "carbon factor", "ef ")


def _has_elec(text: str) -> bool:
    return any(kw in text for kw in _ELEC_KW)


def _has_gas(text: str) -> bool:
    return any(kw in text for kw in _GAS_KW)


def _has_factor(text: str) -> bool:
    return any(kw in text for kw in _FACTOR_KW)


# --- text regex patterns ---

# Forward: "electricity ... emission factor ... 0.43"
_ELEC_FACTOR_RE = re.compile(
    r"(?:electricity|elektrik|electric)[^.\n]{0,80}"
    r"(?:emission\s*factor|emisyon\s*fakt[öo]r[üu]?|co2\s*factor|factor|fakt[öo]r[üu]?)[^.\n]{0,60}"
    r"(?P<value>\d+(?:[.,]\d+)+|\d+\.\d+)",
    re.IGNORECASE,
)

# Forward: "natural gas ... emission factor ... 2.1"
_GAS_FACTOR_RE = re.compile(
    r"(?:natural\s*gas|do[gğ]algaz|gaz\b)[^.\n]{0,80}"
    r"(?:emission\s*factor|emisyon\s*fakt[öo]r[üu]?|co2\s*factor|factor|fakt[öo]r[üu]?)[^.\n]{0,60}"
    r"(?P<value>\d+(?:[.,]\d+)+|\d+\.\d+)",
    re.IGNORECASE,
)

# Reverse: "emission factor ... electricity ... 0.43"
_FACTOR_ELEC_RE = re.compile(
    r"(?:emission\s*factor|emisyon\s*fakt[öo]r[üu]?|co2\s*factor|factor)[^.\n]{0,60}"
    r"(?:electricity|elektrik|electric)[^.\n]{0,60}"
    r"(?P<value>\d+(?:[.,]\d+)+|\d+\.\d+)",
    re.IGNORECASE,
)

# Reverse: "emission factor ... natural gas ... 2.1"
_FACTOR_GAS_RE = re.compile(
    r"(?:emission\s*factor|emisyon\s*fakt[öo]r[üu]?|co2\s*factor|factor)[^.\n]{0,60}"
    r"(?:natural\s*gas|do[gğ]algaz|gaz\b)[^.\n]{0,60}"
    r"(?P<value>\d+(?:[.,]\d+)+|\d+\.\d+)",
    re.IGNORECASE,
)

_TEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_ELEC_FACTOR_RE, "electricity"),
    (_GAS_FACTOR_RE, "natural_gas"),
    (_FACTOR_ELEC_RE, "electricity"),
    (_FACTOR_GAS_RE, "natural_gas"),
]

_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?P<value>\d[\d.,]*)(?![A-Za-z])")
_FACTOR_CONTEXT_TERMS = (
    "factor",
    "emission factor",
    "emisyon faktoru",
    "emisyon faktor",
    "approved factor",
    "ef",
)
_FACTOR_UNIT_TERMS = (
    "kgco2",
    "kgco2e",
    "tco2",
    "tco2e",
    "co2",
    "kwh",
    "mwh",
    "m3",
)
_NON_FACTOR_HINT_TERMS = (
    "version",
    "report year",
    "district count",
)

# Plausible range for emission factors (kgCO2/kWh or kgCO2/m³)
_FACTOR_MIN = 0.001
_FACTOR_MAX = 50.0


def _parse_float(text: str) -> float | None:
    normalized = text.strip().replace(",", ".")
    parts = normalized.split(".")
    if len(parts) > 2:
        normalized = "".join(parts[:-1]) + "." + parts[-1]
    try:
        value = float(normalized)
        return value if _FACTOR_MIN <= value <= _FACTOR_MAX else None
    except ValueError:
        return None


def _extract_numeric_candidates(text: str) -> list[float]:
    candidates: list[float] = []
    for match in _NUMBER_RE.finditer(str(text)):
        value = _parse_float(match.group("value"))
        if value is not None:
            candidates.append(value)
    return candidates


def _extract_factor_value(text: str) -> float | None:
    raw_text = str(text)
    best_value: float | None = None
    best_score: float | None = None

    for match in _NUMBER_RE.finditer(raw_text):
        raw_value = match.group("value")
        value = _parse_float(raw_value)
        if value is None:
            continue
        score = _score_factor_candidate(raw_text, match.start("value"), match.end("value"), raw_value, value)
        if best_score is None or score > best_score:
            best_value = value
            best_score = score

    return best_value


def _score_factor_candidate(text: str, start: int, end: int, raw_value: str, value: float) -> float:
    before = normalize_for_search(text[max(0, start - 40) : start])
    after = normalize_for_search(text[end : min(len(text), end + 40)])
    window = normalize_for_search(text[max(0, start - 40) : min(len(text), end + 40)])
    score = 0.0

    if "." in raw_value or "," in raw_value:
        score += 1.0
    if value < 10.0:
        score += 0.5

    if any(term in before for term in _FACTOR_CONTEXT_TERMS):
        score += 3.0
    elif any(term in window for term in _FACTOR_CONTEXT_TERMS):
        score += 1.5

    if "=" in text[max(0, start - 6) : start] or ":" in text[max(0, start - 6) : start]:
        score += 1.5

    if any(term in after for term in _FACTOR_UNIT_TERMS):
        score += 4.0
    elif any(term in window for term in _FACTOR_UNIT_TERMS):
        score += 1.0

    if raw_value.isdigit() and 1900 <= value <= 2100:
        score -= 5.0

    if any(term in window for term in _NON_FACTOR_HINT_TERMS):
        score -= 3.0

    return score


class FormulaExtractor:
    """Extract emission factors from parsed documents and DataFrames.

    Returns a dict with keys "electricity" and/or "natural_gas" mapping to
    float emission factors (kg CO₂ per unit). Document-provided values
    supplement reference_Data.json defaults — only keys found in the document
    are returned; missing keys fall through to the JSON defaults in DataEngine.
    """

    def extract_from_document(self, parsed_document: ParsedDocument) -> dict[str, float]:
        """Scan text/PDF elements for emission factor declarations."""
        factors: dict[str, float] = {}
        for element in parsed_document.elements:
            text = element.text
            if not text:
                continue
            _scan_text(text, factors)
            if len(factors) == 2:
                break
        return factors

    def extract_from_dataframe(self, dataframe: Any) -> dict[str, float]:
        """Scan a spreadsheet DataFrame for emission factor rows or columns."""
        if dataframe is None:
            return {}
        try:
            return _scan_dataframe(dataframe)
        except Exception:
            return {}

    def extract(self, parsed_document: ParsedDocument, dataframe: Any = None) -> dict[str, float]:
        """Extract from text then spreadsheet; spreadsheet values take precedence."""
        factors = self.extract_from_document(parsed_document)
        if dataframe is not None:
            df_factors = self.extract_from_dataframe(dataframe)
            factors.update(df_factors)
        return factors


def _scan_text(text: str, factors: dict[str, float]) -> None:
    for pattern, key in _TEXT_PATTERNS:
        if key in factors:
            continue
        match = pattern.search(text)
        if match:
            value = _parse_float(match.group("value"))
            if value is not None:
                factors[key] = value


def _scan_dataframe(df: Any) -> dict[str, float]:
    """Look for rows or columns that signal emission factors."""
    factors: dict[str, float] = {}

    # --- scan rows ---
    for _, row in df.iterrows():
        str_cells = [normalize_for_search(str(v).strip()) for v in row.values]
        combined = " ".join(str_cells)

        if not _has_factor(combined):
            continue

        has_elec = _has_elec(combined)
        has_gas = _has_gas(combined)
        if not has_elec and not has_gas:
            continue

        for raw in row.values:
            value = _extract_factor_value(str(raw))
            if value is None:
                continue
            if has_elec and "electricity" not in factors:
                factors["electricity"] = value
            elif has_gas and "natural_gas" not in factors:
                factors["natural_gas"] = value

    # --- scan column headers that signal emission factor ---
    for col in df.columns:
        col_str = normalize_for_search(str(col))
        if not _has_factor(col_str):
            continue
        for idx in df.index:
            idx_str = normalize_for_search(str(idx))
            try:
                value = _extract_factor_value(str(df.at[idx, col]))
                if value is None:
                    continue
                if _has_elec(idx_str) and "electricity" not in factors:
                    factors["electricity"] = value
                elif _has_gas(idx_str) and "natural_gas" not in factors:
                    factors["natural_gas"] = value
            except (KeyError, TypeError):
                continue

    return factors
