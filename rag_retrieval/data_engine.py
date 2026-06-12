"""Deterministic structured analysis over document-derived DataFrames."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metric_discovery import DiscoveredMetric, discover_metrics
from .metric_registry import MetricDefinition, metric_registry
from .text import normalize_for_search, search_tokens


@dataclass(frozen=True, slots=True)
class FormulaBinding:
    kind: str
    reference: str | tuple[str, ...] | float
    hint: str = ""


class DataEngine:
    """Compute deterministic insights from Excel, PDF table, or extracted fact DataFrames."""

    def __init__(
        self,
        dataframe,
        emission_factors: dict | None = None,
        custom_formula=None,
        custom_formula_inputs=None,
        metric_overrides: dict[str, dict[str, object]] | None = None,
    ):
        self.pd = _import_pandas()
        self.reference_data = _load_reference_data()
        self.metric_registry = metric_registry()
        base_factors: dict = dict(self.reference_data.get("emission_factors", {}))
        doc_factors: dict = dict(emission_factors or {})
        self.emission_factors = {**base_factors, **doc_factors}
        self.emission_factors_source: dict[str, str] = {
            key: ("document" if key in doc_factors else "reference")
            for key in self.emission_factors
        }
        self.custom_formula = custom_formula  # ExtractedFormula | None
        self.custom_formula_inputs = dict(custom_formula_inputs or {})
        self.metric_overrides = dict(metric_overrides or {})
        self.district_population = self.reference_data.get("districts", {})
        self.avg_household_size = self.reference_data.get("avg_household_size")
        self.baseline_year = self.reference_data.get("baseline_year")
        self.dataframe = self._normalize_wide_table(self._normalize_dataframe(dataframe))
        self.district_column = self._detect_district_column()
        self.value_column = self._detect_value_column()
        self.metric_column = self._detect_metric_column()
        self.unit_column = self._detect_unit_column()
        self.time_column = self._detect_time_column()
        self.metric_columns = {
            metric_key: self._detect_registry_metric_column(definition)
            for metric_key, definition in self.metric_registry.items()
        }
        self.electricity_column = self.metric_columns.get("electricity")
        self.gas_column = self.metric_columns.get("natural_gas")
        self.water_column = self.metric_columns.get("water")
        self.emissions_column = self._detect_emissions_column()
        self.growth_rate_column = self._detect_growth_rate_column()
        self.discovered_metrics: dict[str, DiscoveredMetric] = discover_metrics(
            self.dataframe,
            registry=self.metric_registry,
            district_column=self.district_column,
            time_column=self.time_column,
            metric_column=self.metric_column,
            unit_column=self.unit_column,
            value_column=self.value_column,
            metric_columns=self.metric_columns,
            metric_aliases_for_key=self._metric_aliases,
            metric_overrides=self.metric_overrides,
        )
        self.report_metric_definitions: dict[str, DiscoveredMetric] = {
            metric_key: metric
            for metric_key, metric in self.discovered_metrics.items()
            if metric.is_known_metric or metric.sustainability_related
        }
        self.custom_formula_bindings: dict[str, FormulaBinding] = {}
        self.custom_formula_bound_variables: dict[str, str] = {}
        self.custom_formula_missing_variables: list[str] = []
        self.custom_formula_status: str = "default"
        self._configure_custom_formula()

    def analyze_district(self, district: str) -> dict:
        if self.dataframe.empty or not self.district_column:
            return {}

        district_rows = self._filter_district_rows(district)
        if district_rows.empty:
            return {}

        population = self._population_for_district(district)
        household_count = self._safe_divide(population, self.avg_household_size) if population is not None else None
        metric_summaries = self._summarize_known_metrics(district_rows, population)

        electricity = float(metric_summaries.get("electricity", {}).get("value") or 0.0)
        gas = float(metric_summaries.get("natural_gas", {}).get("value") or 0.0)
        water = float(metric_summaries.get("water", {}).get("value") or 0.0)
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
            and not any(
                metric_key != "natural_gas" and (
                    metric.source_column is not None or metric.source_kind == "metric_row"
                )
                for metric_key, metric in self.report_metric_definitions.items()
            )
        ):
            gas = self._sum_column(district_rows, self.value_column)
            metric_summaries.setdefault("natural_gas", {}).update({"value": float(gas)})

        electricity_emission = electricity * self._emission_factor("electricity")
        gas_emission = gas * self._emission_factor("natural_gas")

        formula_status = "default"
        formula_missing_values: list[str] = []
        formula_warnings: list[str] = []
        if self.custom_formula:
            if self.custom_formula_status == "ready":
                total_emission, formula_status, formula_missing_values, formula_warnings = self._eval_custom_formula(
                    district_rows,
                    metric_summaries,
                    direct_emissions,
                )
            else:
                total_emission = self._default_total_emission(electricity, gas, direct_emissions)
                formula_status = "custom_incomplete"
                formula_warnings.extend(
                    [
                        f"custom_formula_missing_variable_definition:{name}"
                        for name in self.custom_formula_missing_variables
                    ]
                )
        else:
            total_emission = self._default_total_emission(electricity, gas, direct_emissions)
        growth = self._overall_growth_for_rows(district_rows, metric_summaries)
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
            water=water,
            direct_emissions=direct_emissions,
            total_emission=total_emission,
            growth=growth,
            population=population,
            household_count=household_count,
            metric_summaries=metric_summaries,
        )
        warnings.extend(formula_warnings)

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
            "metrics": metric_summaries,
            "available_metric_keys": [key for key, summary in metric_summaries.items() if summary.get("value") or summary.get("growth") is not None],
            "emission_factors_used": dict(self.emission_factors),
            "emission_factors_source": dict(self.emission_factors_source),
            "formula_expression": self.custom_formula.expression if self.custom_formula else None,
            "formula_source": self.custom_formula.source_text if self.custom_formula else None,
            "formula_status": formula_status,
            "formula_missing_variables": list(self.custom_formula_missing_variables),
            "formula_missing_values": formula_missing_values,
            "formula_bound_variables": dict(self.custom_formula_bound_variables),
            "water_consumption": water,
            "water_per_capita": metric_summaries.get("water", {}).get("per_capita"),
            "water_growth": metric_summaries.get("water", {}).get("growth"),
        }

    def _eval_custom_formula(self, rows, metric_summaries: dict[str, dict[str, object]], direct_emissions: float) -> tuple[float, str, list[str], list[str]]:
        """Evaluate the LLM-extracted formula with explicit variable validation."""
        from .llm_formula_extractor import safe_eval

        electricity = float(metric_summaries.get("electricity", {}).get("value") or 0.0)
        gas = float(metric_summaries.get("natural_gas", {}).get("value") or 0.0)
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
        for metric_key, summary in metric_summaries.items():
            if not self.metric_registry.get(metric_key, None):
                continue
            if not self.metric_registry[metric_key].default_formula_support:
                continue
            value = float(summary.get("value") or 0.0)
            variables[metric_key] = value
            variables[f"{metric_key}_consumption"] = value
            variables[f"{metric_key}_value"] = value

        missing_values: list[str] = []
        for var, binding in self.custom_formula_bindings.items():
            value, has_value = self._formula_binding_value(rows, binding)
            variables[var] = value
            if not has_value:
                missing_values.append(var)

        if missing_values:
            return (
                self._default_total_emission(electricity, gas, direct_emissions),
                "custom_missing_values_fallback",
                missing_values,
                [f"custom_formula_missing_variable_value:{name}" for name in missing_values],
            )

        try:
            result = safe_eval(self.custom_formula.expression, variables)
            return float(result), "custom_applied", [], []
        except Exception as exc:
            return (
                self._default_total_emission(electricity, gas, direct_emissions),
                "custom_runtime_fallback",
                [],
                [f"custom_formula_runtime_error:{type(exc).__name__}"],
            )

    def _configure_custom_formula(self) -> None:
        self.custom_formula_bindings = {}
        self.custom_formula_bound_variables = {}
        self.custom_formula_missing_variables = []
        self.custom_formula_status = "default"
        if not self.custom_formula:
            return

        try:
            from .llm_formula_extractor import extract_formula_variables

            referenced = extract_formula_variables(self.custom_formula.expression)
        except ValueError:
            self.custom_formula_status = "invalid"
            self.custom_formula_missing_variables = ["invalid_expression"]
            return

        for variable in sorted(referenced):
            if variable in self.custom_formula_inputs:
                binding = self._binding_from_user_input(variable, self.custom_formula_inputs[variable])
                if binding is not None:
                    self.custom_formula_bindings[variable] = binding
                    self.custom_formula_bound_variables[variable] = self._binding_label(binding)
                    continue
            if variable in self.custom_formula.constants:
                self.custom_formula_bound_variables[variable] = "constant"
                continue
            if variable in self._builtin_formula_variables():
                self.custom_formula_bound_variables[variable] = "builtin"
                continue

            hint = str(self.custom_formula.variable_hints.get(variable, "") or "")
            binding = self._find_formula_binding(variable, hint)
            if binding is None:
                self.custom_formula_missing_variables.append(variable)
                continue
            self.custom_formula_bindings[variable] = binding
            self.custom_formula_bound_variables[variable] = self._binding_label(binding)

        self.custom_formula_status = "ready" if not self.custom_formula_missing_variables else "incomplete"

    def _builtin_formula_variables(self) -> set[str]:
        variables = {
            "gas",
            "gas_factor",
            "gas_emission_factor",
            "direct_emissions",
            "electricity_factor",
            "electricity_emission_factor",
        }
        for metric_key, definition in self.metric_registry.items():
            if not definition.default_formula_support:
                continue
            variables.add(metric_key)
            variables.add(f"{metric_key}_consumption")
            variables.add(f"{metric_key}_value")
        variables.update(
            {
                "natural_gas",
                "natural_gas_consumption",
                "natural_gas_factor",
            }
        )
        return variables

    def _find_formula_binding(self, variable_name: str, hint: str) -> FormulaBinding | None:
        column = self._match_formula_column(variable_name, hint)
        if column:
            return FormulaBinding(kind="column", reference=column, hint=hint)

        search_terms = self._formula_search_terms(variable_name, hint)
        if self._metric_has_formula_match(search_terms):
            return FormulaBinding(kind="metric", reference=tuple(search_terms), hint=hint)
        return None

    def _binding_from_user_input(self, variable_name: str, user_input: dict[str, object]) -> FormulaBinding | None:
        kind = normalize_for_search(user_input.get("type", "")).replace(" ", "_")
        if kind == "constant":
            try:
                value = float(user_input.get("value"))
            except (TypeError, ValueError):
                return None
            return FormulaBinding(kind="constant", reference=value, hint=f"user_input:{variable_name}")

        if kind == "column":
            column_name = str(user_input.get("column", "") or "")
            resolved = self._resolve_formula_column_reference(column_name)
            if resolved:
                return FormulaBinding(kind="column", reference=resolved, hint=f"user_input:{variable_name}")
        return None

    def _match_formula_column(self, variable_name: str, hint: str) -> str | None:
        target = normalize_for_search(variable_name).replace(" ", "_").strip("_")
        target_compact = target.replace("_", "")
        target_tokens = {token for token in target.split("_") if token}
        hint_tokens = {token for token in self._formula_search_terms(variable_name, hint) if token}
        skip_columns = {
            self.district_column,
            self.metric_column,
            self.unit_column,
            self.time_column,
        }
        best_column: str | None = None
        best_score = 0.0

        for column in self.dataframe.columns:
            if column in skip_columns or self._numeric_ratio(column) <= 0:
                continue

            column_norm = normalize_for_search(column).replace(" ", "_")
            column_compact = column_norm.replace("_", "")
            column_tokens = {token for token in column_norm.split("_") if token}
            score = 0.0

            if target and column_norm == target:
                score += 10.0
            if target_compact and column_compact == target_compact:
                score += 9.0
            if target and target in column_norm:
                score += 6.0

            if target_tokens:
                overlap = len(target_tokens & column_tokens)
                if overlap == len(target_tokens):
                    score += 5.0
                elif overlap:
                    score += 2.0 * (overlap / len(target_tokens))

            if hint_tokens:
                hint_overlap = len(hint_tokens & column_tokens)
                if hint_overlap:
                    score += 1.5 * (hint_overlap / len(hint_tokens))

            if score > best_score:
                best_column = column
                best_score = score

        return best_column if best_score >= 5.0 else None

    def _resolve_formula_column_reference(self, column_name: str) -> str | None:
        target = normalize_for_search(column_name).replace(" ", "_").strip("_")
        target_compact = target.replace("_", "")
        if not target:
            return None

        for column in self.dataframe.columns:
            column_norm = normalize_for_search(column).replace(" ", "_").strip("_")
            if column_norm == target or column_norm.replace("_", "") == target_compact:
                return str(column)
        return None

    def _formula_search_terms(self, variable_name: str, hint: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        for raw in [variable_name.replace("_", " "), hint]:
            for token in search_tokens(raw):
                if token in {"in", "the", "of", "and", "or", "per", "unit"}:
                    continue
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
        return terms

    def _metric_has_formula_match(self, search_terms: list[str]) -> bool:
        if not search_terms or not self.metric_column or self.metric_column not in self.dataframe.columns:
            return False
        metric_values = self.dataframe[self.metric_column].dropna().astype(str).map(normalize_for_search)
        return any(self._metric_matches_terms(value, tuple(search_terms)) for value in metric_values)

    def _metric_matches_terms(self, metric_value: str, search_terms: tuple[str, ...]) -> bool:
        metric_tokens = set(search_tokens(metric_value))
        term_set = {term for term in search_terms if term}
        if not term_set:
            return False
        if " ".join(search_terms) in metric_value:
            return True
        overlap = len(term_set & metric_tokens)
        return overlap > 0 and overlap / len(term_set) >= 0.5

    def _formula_binding_value(self, rows, binding: FormulaBinding) -> tuple[float, bool]:
        if binding.kind == "constant":
            return float(binding.reference), True

        if binding.kind == "column":
            values = self._numeric_series(rows[binding.reference]).dropna()
            if values.empty:
                return 0.0, False
            return float(values.sum()), True

        if binding.kind == "metric":
            return self._sum_metric_terms(rows, binding.reference)

        return 0.0, False

    def _sum_metric_terms(self, rows, search_terms: tuple[str, ...]) -> tuple[float, bool]:
        if (
            not self.metric_column
            or self.metric_column not in rows.columns
            or not self.value_column
            or self.value_column not in rows.columns
        ):
            return 0.0, False

        metric_values = rows[self.metric_column].astype(str).map(normalize_for_search)
        mask = metric_values.apply(lambda value: self._metric_matches_terms(value, search_terms))
        if not mask.any():
            return 0.0, False

        values = self._numeric_series(rows.loc[mask, self.value_column]).dropna()
        if values.empty:
            return 0.0, False

        if self.unit_column and self.unit_column in rows.columns:
            units = rows.loc[mask, self.unit_column].astype(str).map(
                lambda value: normalize_for_search(value).replace(" ", "")
            )
            values = values.where(units != "mwh", values * 1000)
        return float(values.sum()), True

    @staticmethod
    def _binding_label(binding: FormulaBinding) -> str:
        if binding.kind == "constant":
            return "user_constant"
        if binding.kind == "column":
            return f"column:{binding.reference}"
        if binding.kind == "metric":
            return f"metric:{','.join(binding.reference)}"
        return binding.kind

    def _default_total_emission(self, electricity: float, gas: float, direct_emissions: float) -> float:
        total = electricity * self._emission_factor("electricity") + gas * self._emission_factor("natural_gas")
        return total if total != 0.0 else direct_emissions

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

    def formula_candidate_columns(self) -> list[str]:
        skip_columns = {
            self.district_column,
            self.metric_column,
            self.unit_column,
            self.time_column,
        }
        candidates = []
        for column in self.dataframe.columns:
            if column in skip_columns:
                continue
            if self._numeric_ratio(column) <= 0:
                continue
            candidates.append(str(column))
        return sorted(set(candidates))

    def _summarize_known_metrics(self, rows, population: float | None) -> dict[str, dict[str, object]]:
        summaries: dict[str, dict[str, object]] = {}
        for metric_key, definition in self.report_metric_definitions.items():
            value = self._metric_total(rows, metric_key)
            growth = self._growth_for_metric(rows, metric_key)
            summaries[metric_key] = {
                "metric_key": metric_key,
                "label": definition.display_name,
                "category": definition.category,
                "unit": definition.unit,
                "role": definition.role,
                "report_section": definition.report_section,
                "value": float(value),
                "per_capita": self._safe_divide(value, population),
                "growth": growth,
                "source_column": definition.source_column,
                "sustainability_related": definition.sustainability_related,
                "is_known_metric": definition.is_known_metric,
                "source_kind": definition.source_kind,
            }
        return summaries

    def _metric_total(self, rows, metric_key: str) -> float:
        definition = self.report_metric_definitions.get(metric_key)
        if definition is None:
            return 0.0
        specific_column = definition.source_column
        value = self._sum_metric_specific_column(rows, metric_key, specific_column)
        if definition.source_kind == "metric_row" or (definition.source_kind != "column" and value == 0.0):
            value += self._sum_metric_value(rows, definition.metric_terms, specific_column)
        return float(value)

    def _sum_metric_specific_column(self, rows, metric_key: str, column: str | None) -> float:
        if not column or column not in rows.columns:
            return 0.0
        values = self._numeric_series(rows[column])
        total = float(values.dropna().sum())
        if total == 0.0:
            return 0.0
        normalized_column = normalize_for_search(column).replace(" ", "_")
        if metric_key == "electricity" and "mwh" in normalized_column:
            return total * 1000.0
        if metric_key in {"water", "wastewater"} and any(token in normalized_column for token in ("liter", "litre", "liters", "litres")):
            return total / 1000.0
        return total

    def _metric_aliases(self, metric_key: str, *, include_units: bool = True) -> tuple[str, ...]:
        definition = self.metric_registry.get(metric_key)
        if definition is None:
            return (metric_key,)
        aliases = [metric_key, metric_key.replace("_", " "), *definition.aliases]
        normalized: list[str] = []
        seen: set[str] = set()
        ignored_unit_aliases = {"kwh", "mwh", "m3", "m³", "l", "ton", "tons", "tonnes"}
        for alias in aliases:
            value = normalize_for_search(alias)
            if not include_units and value in ignored_unit_aliases:
                continue
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return tuple(normalized)

    def _detect_registry_metric_column(self, definition: MetricDefinition) -> str | None:
        best_column: str | None = None
        best_score = 0.0
        aliases = self._metric_aliases(definition.metric_key, include_units=False)
        alias_tokens = [self._alias_tokens(alias) for alias in aliases]
        for column in self.dataframe.columns:
            if self._is_time_column_name(column):
                continue
            numeric_ratio = self._numeric_ratio(column)
            if numeric_ratio <= 0:
                continue

            column_norm = normalize_for_search(column).replace(" ", "_")
            column_tokens = set(search_tokens(column_norm))
            if self.metric_column and column == self.value_column and definition.metric_key not in column_norm:
                continue

            alias_score = 0.0
            if definition.metric_key in column_norm:
                alias_score = max(alias_score, 4.0)
            for alias, tokens in zip(aliases, alias_tokens):
                alias_norm = alias.replace(" ", "_")
                if column_norm == alias_norm:
                    alias_score = max(alias_score, 10.0)
                elif alias_norm in column_norm:
                    alias_score = max(alias_score, 7.0)
                elif tokens:
                    overlap = len(tokens & column_tokens)
                    if overlap == len(tokens):
                        alias_score = max(alias_score, 6.0)
                    elif overlap and overlap / len(tokens) >= 0.6:
                        alias_score = max(alias_score, 4.5)
            if alias_score <= 0.0:
                continue

            value_count = int(self._numeric_series(self.dataframe[column]).notna().sum())
            score = alias_score + numeric_ratio + min(float(value_count), 25.0) / 5.0
            if self._is_value_column_name(column):
                score += 1.0
            if best_column is None or score > best_score:
                best_column = column
                best_score = score

        return best_column if best_score >= 4.0 else None

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

        metric_rows = self._metric_rows(rows, metric_terms)
        if metric_rows.empty:
            return 0.0
        if specific_column and specific_column in rows.columns:
            specific_values = self._numeric_series(rows[specific_column])
            metric_rows = metric_rows.loc[specific_values.loc[metric_rows.index].isna()]
        if metric_rows.empty:
            return 0.0

        values = self._numeric_series(metric_rows[self.value_column])
        if self.unit_column and self.unit_column in metric_rows.columns:
            units = metric_rows[self.unit_column].astype(str).map(lambda value: normalize_for_search(value).replace(" ", ""))
            values = values.where(units != "mwh", values * 1000)
        return float(values.dropna().sum())

    def _metric_rows(self, rows, metric_terms: tuple[str, ...]):
        if not self.metric_column or self.metric_column not in rows.columns:
            return rows.iloc[0:0]
        metric_values = rows[self.metric_column].astype(str).map(normalize_for_search)
        mask = metric_values.apply(lambda value: self._metric_matches_aliases(value, metric_terms))
        return rows.loc[mask]

    def _metric_matches_aliases(self, metric_value: str, aliases: tuple[str, ...]) -> bool:
        normalized_value = normalize_for_search(metric_value)
        tokens = set(search_tokens(normalized_value))
        for alias in aliases:
            alias_value = normalize_for_search(alias)
            if not alias_value:
                continue
            alias_tokens = self._alias_tokens(alias_value)
            if self._alias_substring_match(alias_value, alias_tokens, normalized_value):
                return True
            if alias_tokens and alias_tokens <= tokens:
                return True
        return False

    @staticmethod
    def _alias_tokens(value: str) -> set[str]:
        ignored = {
            "consumption",
            "use",
            "usage",
            "generated",
            "generation",
            "supplied",
            "supply",
            "total",
            "municipal",
            "clean",
            "potable",
            "tuketim",
            "tuketimi",
            "kullanim",
            "kullanimi",
        }
        return {token for token in search_tokens(value) if token not in ignored}

    @staticmethod
    def _alias_substring_match(alias_value: str, alias_tokens: set[str], normalized_value: str) -> bool:
        if not alias_tokens:
            return False
        if len(alias_tokens) == 1:
            token = next(iter(alias_tokens))
            if len(token) <= 2:
                return False
            return token in set(search_tokens(normalized_value))
        pattern = r"\b" + re.escape(alias_value).replace(r"\ ", r"\s+") + r"\b"
        return re.search(pattern, normalized_value) is not None

    def _growth_for_metric(self, rows, metric_key: str) -> float | None:
        if not self.time_column or self.time_column not in rows.columns:
            return None

        definition = self.report_metric_definitions.get(metric_key)
        if definition is None:
            return None
        specific_column = definition.source_column
        if specific_column and specific_column in rows.columns:
            return self._growth_from_series(rows, specific_column)

        metric_rows = self._metric_rows(rows, definition.metric_terms)
        if metric_rows.empty or not self.value_column or self.value_column not in metric_rows.columns:
            return None
        return self._growth_from_series(metric_rows, self.value_column)

    def _overall_growth_for_rows(self, rows, metric_summaries: dict[str, dict[str, object]]) -> float | None:
        growth = self._growth_for_rows(rows)
        if growth is not None:
            return growth

        for summary in metric_summaries.values():
            if float(summary.get("value") or 0.0) <= 0.0:
                continue
            metric_growth = summary.get("growth")
            if metric_growth is not None:
                return float(metric_growth)
        return None

    def _growth_for_rows(self, rows) -> float | None:
        column = self._growth_value_column(rows)
        if not column or column not in rows.columns or not self.time_column or self.time_column not in rows.columns:
            return None
        return self._growth_from_series(rows, column)

    def _growth_from_series(self, rows, value_column: str) -> float | None:
        values = self._numeric_series(rows[value_column])
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
        candidates = (self.gas_column, self.electricity_column, self.water_column, self.emissions_column, self.value_column)
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
        water: float,
        direct_emissions: float,
        total_emission: float,
        growth: float | None,
        population: float | None,
        household_count: float | None,
        metric_summaries: dict[str, dict[str, object]],
    ) -> list[str]:
        warnings: list[str] = []
        has_resource_metrics = any(float(summary.get("value") or 0.0) > 0.0 for summary in metric_summaries.values())
        if electricity == 0.0 and (gas > 0.0 or direct_emissions > 0.0 or not has_resource_metrics):
            warnings.append("electricity_consumption_not_found")
        if gas == 0.0 and (electricity > 0.0 or direct_emissions > 0.0 or not has_resource_metrics):
            warnings.append("natural_gas_consumption_not_found")
        if total_emission == 0.0 and direct_emissions == 0.0 and not has_resource_metrics:
            warnings.append("no_consumption_or_emissions_detected")
        if direct_emissions > 0 and (electricity > 0 or gas > 0):
            warnings.append("direct_emissions_reported_separately_to_avoid_double_counting")
        elif direct_emissions > 0 and total_emission == direct_emissions:
            warnings.append("total_emission_uses_direct_emissions_only")
        if water == 0.0 and metric_summaries.get("water", {}).get("source_column"):
            warnings.append("water_consumption_not_found")
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
            "wastewater",
            "diesel",
            "gasoline",
            "fuel",
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
    tokens = set(column_without_year.split("_"))
    if {"electric", "electricity", "kwh"} & tokens:
        return "electricity_consumption_kwh"
    if {"water", "su", "potable", "clean", "cleanwater"} & tokens:
        return "water_consumption_m3"
    if "wastewater" in tokens or ("waste" in tokens and "water" in tokens) or ("atik" in tokens and "su" in tokens):
        return "wastewater_m3"
    if {"diesel", "motorin"} & tokens:
        return "diesel_liters"
    if {"gasoline", "petrol", "benzin"} & tokens:
        return "gasoline_liters"
    if {"waste", "atik", "atik"} & tokens:
        return "waste_tonnes"
    if {"gas", "natural", "naturalgas", "dogalgaz", "consumption", "m3"} & tokens:
        return "natural_gas_consumption_m3"
    if {"emission", "emissions", "co2", "tco2e"} & tokens:
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
