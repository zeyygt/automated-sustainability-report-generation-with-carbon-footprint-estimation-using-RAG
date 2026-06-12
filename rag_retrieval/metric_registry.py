"""Central registry for built-in sustainability metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_key: str
    aliases: tuple[str, ...]
    category: str
    unit: str
    role: str
    default_formula_support: bool
    default_factor_support: bool
    report_section: str


_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        metric_key="electricity",
        aliases=(
            "electricity",
            "electricity consumption",
            "electric consumption",
            "electric use",
            "power consumption",
            "elektrik",
            "elektrik tuketimi",
            "kwh",
            "mwh",
        ),
        category="energy",
        unit="kWh",
        role="emission_input",
        default_formula_support=True,
        default_factor_support=True,
        report_section="Emissions Overview",
    ),
    MetricDefinition(
        metric_key="natural_gas",
        aliases=(
            "natural gas",
            "natural gas consumption",
            "gas consumption",
            "gas use",
            "dogalgaz",
            "doğalgaz",
            "dogalgaz tuketimi",
            "natural_gas",
        ),
        category="energy",
        unit="m3",
        role="emission_input",
        default_formula_support=True,
        default_factor_support=True,
        report_section="Emissions Overview",
    ),
    MetricDefinition(
        metric_key="water",
        aliases=(
            "water",
            "water consumption",
            "water use",
            "water usage",
            "clean water",
            "potable water",
            "su tuketimi",
            "water supplied",
        ),
        category="water",
        unit="m3",
        role="resource_kpi",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Water Overview",
    ),
    MetricDefinition(
        metric_key="fuel",
        aliases=("fuel", "fuel consumption", "fuel use", "yakit", "yakıt", "akaryakit", "akaryakıt"),
        category="mobility",
        unit="L",
        role="emission_input",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Resource Overview",
    ),
    MetricDefinition(
        metric_key="diesel",
        aliases=("diesel", "diesel consumption", "diesel use", "motorin"),
        category="mobility",
        unit="L",
        role="emission_input",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Resource Overview",
    ),
    MetricDefinition(
        metric_key="gasoline",
        aliases=("gasoline", "petrol", "benzin", "gasoline consumption", "petrol consumption"),
        category="mobility",
        unit="L",
        role="emission_input",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Resource Overview",
    ),
    MetricDefinition(
        metric_key="waste",
        aliases=("waste", "solid waste", "municipal waste", "atik", "atık", "waste generated"),
        category="waste",
        unit="t",
        role="resource_kpi",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Resource Overview",
    ),
    MetricDefinition(
        metric_key="wastewater",
        aliases=("wastewater", "waste water", "wastewater treated", "atik su", "atık su", "wastewater flow"),
        category="water",
        unit="m3",
        role="resource_kpi",
        default_formula_support=True,
        default_factor_support=False,
        report_section="Resource Overview",
    ),
)


def metric_registry() -> dict[str, MetricDefinition]:
    return {metric.metric_key: metric for metric in _METRICS}


def metric_definition(metric_key: str) -> MetricDefinition | None:
    return metric_registry().get(metric_key)


def report_sections() -> tuple[str, ...]:
    return ("Emissions Overview", "Water Overview", "Resource Overview")
