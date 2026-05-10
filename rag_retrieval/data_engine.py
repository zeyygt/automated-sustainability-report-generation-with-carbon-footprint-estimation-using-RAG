"""Deterministic structured analysis over document-derived DataFrames."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .text import normalize_for_search


class DataEngine:
    """Compute deterministic insights from Excel, PDF table, or extracted fact DataFrames."""

    def __init__(self, dataframe, emission_factors: dict | None = None, custom_formula=None):
        self.pd = _import_pandas()
        self.reference_data = _load_reference_data()
        base_factors: dict = dict(self.reference_data.get("emission_factors", {}))
        doc_factors: dict = dict(emission_factors or {})
        self.emission_factors = {**base_factors, **doc_factors}
        self.emission_factors_source: dict[str, str] = {
            key: ("document" if key in doc_factors else "reference")
            for key in self.emission_factors
        }
        self.custom_formula = custom_formula  # ExtractedFormula | None
        self.district_population = self.reference_data.get("districts", {})
        self.avg_household_size = self.reference_data.get("avg_household_size")
        self.baseline_year = self.reference_data.get("baseline_year")
        self.dataframe = self._normalize_wide_table(self._normalize_dataframe(dataframe))
        self.district_column = self._detect_district_column()
        self.value_column = self._detect_value_column()
        self.electricity_column = self._detect_electricity_column()
        self.gas_column = self._detect_gas_column()
        self.emissions_column = self._detect_emissions_column()
        self.metric_column = self._detect_metric_column()
        self.unit_column = self._detect_unit_column()
        self.time_column = self._detect_time_column()
        self.growth_rate_column = self._detect_growth_rate_column()

    def analyze_district(self, district: str) -> dict:
        if self.dataframe.empty or not self.district_column:
            return {}

        district_rows = self._filter_district_rows(district)
        if district_rows.empty:
            return {}

        electricity = self._sum_column(district_rows, self.electricity_column)
        gas = self._sum_column(district_rows, self.gas_column)
        electricity += self._sum_metric_value(
            district_rows,
            ("electricity", "electricity_consumption", "electric", "elektrik", "kwh", "mwh"),
            self.electricity_column,
        )
        gas += self._sum_metric_value(
            district_rows,
            ("natural_gas", "natural_gas_consumption", "dogalgaz", "gas", "m3"),
            self.gas_column,
        )
        direct_emissions = self._sum_column(district_rows, self.emissions_column)
        direct_emissions += self._sum_metric_value(
            district_rows,
            ("emission", "emissions", "carbon", "co2", "tco2e"),
            self.emissions_column,
        )
        if (
            gas == 0.0
            and self.gas_column is None
            and self.electricity_column is None
            and self.emissions_column is None
            and self.metric_column is None
        ):
            gas = self._sum_column(district_rows, self.value_column)

        electricity_emission = electricity * self._emission_factor("electricity")
        gas_emission = gas * self._emission_factor("natural_gas")

        if self.custom_formula:
            total_emission = self._eval_custom_formula(electricity, gas, direct_emissions)
        else:
            total_emission = electricity_emission + gas_emission
            if total_emission == 0.0 and direct_emissions:
                total_emission = direct_emissions
        population = self._population_for_district(district)
        household_count = self._safe_divide(population, self.avg_household_size) if population is not None else None
        growth = self._growth_for_rows(district_rows)
        if growth is None and self.growth_rate_column:
            col = self.growth_rate_column
            if col in district_rows.columns:
                values = self._numeric_series(district_rows[col]).dropna()
                if not values.empty:
                    raw = float(values.iloc[0])
                    # normalize: if stored as whole-number percent (e.g. 5.0 = 5%), divide by 100
                    growth = raw / 100.0 if abs(raw) > 1.5 else raw
        warnings = self._analysis_warnings(
            electricity=electricity,
            gas=gas,
            direct_emissions=direct_emissions,
            total_emission=total_emission,
            growth=growth,
            population=population,
            household_count=household_count,
        )

        return {
            "district": str(district_rows[self.district_column].iloc[0]),
            "electricity_emission": float(electricity_emission),
            "gas_emission": float(gas_emission),
            "direct_emissions": float(direct_emissions),
            "total_emission": float(total_emission),
            "per_capita": self._safe_divide(total_emission, population),
            "per_household": self._safe_divide(total_emission, household_count),
            "growth": growth,
            "warnings": warnings,
            "emission_factors_used": dict(self.emission_factors),
            "emission_factors_source": dict(self.emission_factors_source),
            "formula_expression": self.custom_formula.expression if self.custom_formula else None,
            "formula_source": self.custom_formula.source_text if self.custom_formula else None,
        }

    def _eval_custom_formula(self, electricity: float, gas: float, direct_emissions: float) -> float:
        """Evaluate the LLM-extracted formula. Falls back to default sum on any error."""
        from .llm_formula_extractor import safe_eval

        variables: dict[str, float] = {
            # consumption aliases
            "electricity": electricity,
            "electricity_consumption": electricity,
            "gas": gas,
            "natural_gas": gas,
            "natural_gas_consumption": gas,
            "direct_emissions": direct_emissions,
            # emission factor aliases
            "electricity_factor": self._emission_factor("electricity"),
            "electricity_emission_factor": self._emission_factor("electricity"),
            "gas_factor": self._emission_factor("natural_gas"),
            "natural_gas_factor": self._emission_factor("natural_gas"),
            "gas_emission_factor": self._emission_factor("natural_gas"),
            # constants declared in the formula doc
            **self.custom_formula.constants,
        }
        # Default any remaining variables named in the formula to 0.0 so the
        # formula evaluates even when the data has no matching column (e.g. renewable).
        for var in self.custom_formula.variable_hints:
            if var not in variables:
                variables[var] = 0.0

        try:
            result = safe_eval(self.custom_formula.expression, variables)
            return float(result)
        except Exception:
            # fall back to the default formula
            default = electricity * self._emission_factor("electricity") + gas * self._emission_factor("natural_gas")
            return default if default != 0.0 else direct_emissions

    def districts(self) -> list[str]:
        if self.dataframe.empty or not self.district_column:
            return []
        values = self.dataframe[self.district_column].dropna().astype(str)
        return sorted({value.strip() for value in values if value.strip()})

    def compare_districts(self) -> dict:
        if self.dataframe.empty or not self.district_column or not self.value_column:
            return {}

        values = self._numeric_series(self.dataframe[self.value_column])
        rows = self.dataframe.assign(_data_engine_value=values).dropna(subset=["_data_engine_value"])
        if rows.empty:
            return {}

        grouped = rows.groupby(self.district_column, dropna=True)["_data_engine_value"].sum()
        if grouped.empty:
            return {}

        top_district = grouped.idxmax()
        lowest_district = grouped.idxmin()
        return {
            "top_district": str(top_district),
            "top_value": float(grouped.loc[top_district]),
            "lowest_district": str(lowest_district),
            "lowest_value": float(grouped.loc[lowest_district]),
        }

    def _detect_district_column(self) -> str | None:
        preferred_terms = ("district", "ilce", "region", "city", "location", "site", "facility", "municipality")
        for column in self.dataframe.columns:
            if any(term in column for term in preferred_terms) and self._is_mostly_text(column):
                return column

        candidates = [
            column
            for column in self.dataframe.columns
            if self._is_mostly_text(column) and not self._is_time_column_name(column) and not self._is_value_column_name(column)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda column: self.dataframe[column].nunique(dropna=True))

    def _detect_value_column(self) -> str | None:
        best_column: str | None = None
        best_score = -1.0
        for column in self.dataframe.columns:
            if self._is_time_column_name(column):
                continue
            numeric_ratio = self._numeric_ratio(column)
            if numeric_ratio <= 0:
                continue

            score = numeric_ratio
            if self._is_value_column_name(column):
                score += 2.0
            if best_column is None or score > best_score:
                best_column = column
                best_score = score
        return best_column

    def _detect_electricity_column(self) -> str | None:
        return self._detect_consumption_column(("electric", "electricity", "kwh"))

    def _detect_gas_column(self) -> str | None:
        return self._detect_consumption_column(("gas", "natural_gas", "dogalgaz", "m3"))

    def _detect_emissions_column(self) -> str | None:
        return self._detect_consumption_column(("emission", "emissions", "carbon", "co2", "tco2e"))

    def _detect_growth_rate_column(self) -> str | None:
        return self._detect_consumption_column(("growth", "büyüme", "degisim", "change_rate", "degisim_orani"))

    def _detect_consumption_column(self, terms: tuple[str, ...]) -> str | None:
        best_column: str | None = None
        best_score = -1.0
        for column in self.dataframe.columns:
            if self._is_time_column_name(column):
                continue
            if not any(term in column for term in terms):
                continue
            numeric_ratio = self._numeric_ratio(column)
            if numeric_ratio <= 0:
                continue
            score = numeric_ratio
            if self._is_value_column_name(column):
                score += 1.0
            if best_column is None or score > best_score:
                best_column = column
                best_score = score
        return best_column

    def _detect_metric_column(self) -> str | None:
        metric_terms = ("metric", "indicator", "category", "type", "kalem", "gosterge")
        for column in self.dataframe.columns:
            if any(term in column for term in metric_terms) and self._is_mostly_text(column):
                return column
        return None

    def _detect_unit_column(self) -> str | None:
        unit_terms = ("unit", "birim")
        for column in self.dataframe.columns:
            if any(term in column for term in unit_terms):
                return column
        return None

    def _detect_time_column(self) -> str | None:
        for column in self.dataframe.columns:
            tokens = set(column.split("_"))
            if "year" in tokens or "yil" in tokens:
                return column
        for column in self.dataframe.columns:
            tokens = set(column.split("_"))
            if "month" in tokens or "ay" in tokens:
                return column
        return None

    def _normalize_dataframe(self, dataframe):
        if dataframe is None:
            return self.pd.DataFrame()
        normalized = dataframe.copy()
        normalized.columns = _unique_names(_normalize_column_name(column) for column in normalized.columns)
        return normalized

    def _normalize_wide_table(self, dataframe):
        if dataframe.empty or any(self._is_time_column_name(column) for column in dataframe.columns):
            return dataframe

        year_columns = [
            column
            for column in dataframe.columns
            if _year_from_column(column) is not None and _numeric_ratio_for_series(self.pd, dataframe[column]) > 0
        ]
        if not year_columns:
            return dataframe

        id_columns = [column for column in dataframe.columns if column not in year_columns]
        if not id_columns:
            return dataframe

        records = []
        for _, row in dataframe.iterrows():
            base = {column: row[column] for column in id_columns}
            for year_column in year_columns:
                value = row[year_column]
                if self.pd.isna(value) or str(value).strip() == "":
                    continue
                record = dict(base)
                record["year"] = _year_from_column(year_column)
                record[_metric_column_from_year_column(year_column)] = value
                records.append(record)

        if not records:
            return dataframe
        return self.pd.DataFrame(records)

    def _filter_district_rows(self, district: str):
        target = normalize_for_search(district)
        if not target:
            return self.dataframe.iloc[0:0]
        normalized_values = self.dataframe[self.district_column].astype(str).map(normalize_for_search)
        exact_matches = self.dataframe[normalized_values == target]
        if not exact_matches.empty:
            return exact_matches
        return self.dataframe[normalized_values.str.contains(re.escape(target), na=False)]

    def _sort_for_trend(self, rows):
        if not self.time_column or self.time_column not in rows.columns:
            return rows

        sort_key = self._numeric_series(rows[self.time_column])
        if sort_key.notna().any():
            return rows.assign(_data_engine_time=sort_key).sort_values("_data_engine_time", kind="mergesort")
        return rows.sort_values(self.time_column, kind="mergesort")

    def _sum_column(self, rows, column: str | None) -> float:
        if not column or column not in rows.columns:
            return 0.0
        values = self._numeric_series(rows[column])
        return float(values.dropna().sum())

    def _sum_metric_value(
        self,
        rows,
        metric_terms: tuple[str, ...],
        specific_column: str | None = None,
    ) -> float:
        if not self.metric_column or self.metric_column not in rows.columns or not self.value_column:
            return 0.0
        if self.value_column not in rows.columns or self.value_column == specific_column:
            return 0.0

        metric_values = rows[self.metric_column].astype(str).map(normalize_for_search)
        mask = metric_values.apply(lambda value: any(term in value for term in metric_terms))
        if specific_column and specific_column in rows.columns:
            specific_values = self._numeric_series(rows[specific_column])
            mask &= specific_values.isna()
        if not mask.any():
            return 0.0

        values = self._numeric_series(rows.loc[mask, self.value_column])
        if self.unit_column and self.unit_column in rows.columns:
            units = rows.loc[mask, self.unit_column].astype(str).map(lambda value: normalize_for_search(value).replace(" ", ""))
            values = values.where(units != "mwh", values * 1000)
        return float(values.dropna().sum())

    def _growth_for_rows(self, rows) -> float | None:
        column = self._growth_value_column(rows)
        if not column or column not in rows.columns or not self.time_column or self.time_column not in rows.columns:
            return None

        values = self._numeric_series(rows[column])
        times = self._numeric_series(rows[self.time_column])
        valid_rows = rows.assign(_data_engine_value=values, _data_engine_time=times).dropna(
            subset=["_data_engine_value", "_data_engine_time"]
        )
        if valid_rows.empty:
            return None

        grouped = valid_rows.groupby("_data_engine_time")["_data_engine_value"].sum().sort_index()
        if grouped.empty or len(grouped) < 2:
            return None

        baseline_key = float(self.baseline_year) if self.baseline_year is not None else grouped.index[0]
        if baseline_key not in grouped.index:
            baseline_key = grouped.index[0]
        baseline_value = float(grouped.loc[baseline_key])
        current_value = float(grouped.iloc[-1])
        return self._safe_growth(baseline_value, current_value)

    def _growth_value_column(self, rows) -> str | None:
        candidates = (self.gas_column, self.electricity_column, self.emissions_column, self.value_column)
        for column in candidates:
            if not column or column not in rows.columns:
                continue
            if self._numeric_series(rows[column]).notna().any():
                return column
        return None

    def _analysis_warnings(
        self,
        *,
        electricity: float,
        gas: float,
        direct_emissions: float,
        total_emission: float,
        growth: float | None,
        population: float | None,
        household_count: float | None,
    ) -> list[str]:
        warnings: list[str] = []
        if electricity == 0.0:
            warnings.append("electricity_consumption_not_found")
        if gas == 0.0:
            warnings.append("natural_gas_consumption_not_found")
        if total_emission == 0.0 and direct_emissions == 0.0:
            warnings.append("no_consumption_or_emissions_detected")
        if direct_emissions > 0 and (electricity > 0 or gas > 0):
            warnings.append("direct_emissions_reported_separately_to_avoid_double_counting")
        elif direct_emissions > 0 and total_emission == direct_emissions:
            warnings.append("total_emission_uses_direct_emissions_only")
        if growth is None:
            warnings.append("growth_unavailable_missing_or_invalid_time_series")
        if population is None:
            warnings.append("population_reference_not_found")
        elif household_count is None:
            warnings.append("household_count_unavailable")
        if electricity > 0 and self._emission_factor("electricity") == 0.0:
            warnings.append("electricity_emission_factor_not_found")
        if gas > 0 and self._emission_factor("natural_gas") == 0.0:
            warnings.append("natural_gas_emission_factor_not_found")
        return warnings

    def _numeric_series(self, series):
        cleaned = series.astype(str).map(_clean_numeric_text)
        return self.pd.to_numeric(cleaned, errors="coerce")

    def _numeric_ratio(self, column: str) -> float:
        series = self.dataframe[column]
        if series.empty:
            return 0.0
        numeric = self._numeric_series(series)
        return float(numeric.notna().sum()) / max(len(series), 1)

    def _is_mostly_text(self, column: str) -> bool:
        series = self.dataframe[column].dropna()
        if series.empty:
            return False
        return self._numeric_ratio(column) < 0.5

    @staticmethod
    def _is_value_column_name(column: str) -> bool:
        value_terms = (
            "amount",
            "consumption",
            "dogalgaz",
            "electricity",
            "emission",
            "emissions",
            "energy",
            "miktar",
            "tuketim",
            "usage",
            "value",
            "waste",
            "water",
        )
        return any(term in column for term in value_terms)

    @staticmethod
    def _is_time_column_name(column: str) -> bool:
        tokens = set(column.split("_"))
        return bool(tokens & {"year", "yil", "month", "ay", "date", "period"})

    def _emission_factor(self, key: str) -> float:
        try:
            return float(self.emission_factors.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _population_for_district(self, district: str) -> float | None:
        target = normalize_for_search(district)
        for district_name, data in self.district_population.items():
            if normalize_for_search(district_name) == target:
                try:
                    return float(data.get("population"))
                except (AttributeError, TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator in (None, 0):
            return None
        value = numerator / denominator
        if math.isfinite(value):
            return float(value)
        return None

    @staticmethod
    def _trend(first_value: float, last_value: float) -> str:
        if last_value > first_value:
            return "increasing"
        if last_value < first_value:
            return "decreasing"
        return "stable"

    @staticmethod
    def _safe_growth(first_value: float, last_value: float) -> float | None:
        if first_value == 0:
            return 0.0 if last_value == 0 else None
        growth = (last_value - first_value) / first_value
        if math.isfinite(growth):
            return float(growth)
        return None


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for DataEngine. Install with: pip install pandas") from exc
    return pd


def _load_reference_data() -> dict:
    directory = Path(__file__).resolve().parent
    for filename in ("reference_data.json", "reference_Data.json"):
        path = directory / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _normalize_column_name(column: Any) -> str:
    normalized = normalize_for_search(str(column).strip())
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized or "column"


def _unique_names(names) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f"{name}_{count}")
    return unique


def _year_from_column(column: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", column)
    return int(match.group(0)) if match else None


def _metric_column_from_year_column(column: str) -> str:
    column_without_year = re.sub(r"(?:19|20)\d{2}", "", column).strip("_")
    if any(term in column_without_year for term in ("electric", "electricity", "kwh")):
        return "electricity_consumption_kwh"
    if any(term in column_without_year for term in ("gas", "natural_gas", "dogalgaz", "m3", "consumption")):
        return "natural_gas_consumption_m3"
    if any(term in column_without_year for term in ("emission", "emissions", "co2", "tco2e")):
        return "emissions"
    return "value"


def _numeric_ratio_for_series(pd, series) -> float:
    if series.empty:
        return 0.0
    cleaned = series.astype(str).map(_clean_numeric_text)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return float(numeric.notna().sum()) / max(len(series), 1)


def _clean_numeric_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text.replace(",", "")
