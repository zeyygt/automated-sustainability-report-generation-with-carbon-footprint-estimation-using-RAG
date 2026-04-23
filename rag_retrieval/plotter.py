"""Generate report charts from structured DataEngine outputs."""

from __future__ import annotations

from pathlib import Path

from .report_metrics import public_metrics
from .report_models import ReportInput


def generate_report_charts(report_input: ReportInput, output_dir: str | Path) -> list[dict[str, str]]:
    metrics = _metrics(report_input)
    if not metrics:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    charts: list[dict[str, str]] = []
    charts.extend(_bar_chart_total_emissions(plt, metrics, output_path))
    charts.extend(_bar_chart_growth(plt, metrics, output_path))
    charts.extend(_stacked_components_chart(plt, metrics, output_path))
    return charts


def _metrics(report_input: ReportInput) -> list[dict]:
    values = []
    for data in public_metrics(report_input.structured_results):
        values.append(
            {
                "label": data.get("district", "Unknown"),
                "district": data.get("district", "Unknown"),
                "total_emission": float(data.get("total_emission") or 0.0),
                "gas_emission": float(data.get("gas_emission") or 0.0),
                "electricity_emission": float(data.get("electricity_emission") or 0.0),
                "direct_emissions": float(data.get("direct_emissions") or 0.0),
                "growth": data.get("growth"),
            }
        )
    return values


def _bar_chart_total_emissions(plt, metrics: list[dict], output_dir: Path) -> list[dict[str, str]]:
    selected = sorted(metrics, key=lambda item: item["total_emission"], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh([item["label"] for item in selected], [item["total_emission"] for item in selected], color="#22577a")
    ax.invert_yaxis()
    ax.set_title("Total Emissions by District")
    ax.set_xlabel("Total emissions")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    path = output_dir / "total_emissions_by_district.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"title": "Total Emissions by District", "path": str(path)}]


def _bar_chart_growth(plt, metrics: list[dict], output_dir: Path) -> list[dict[str, str]]:
    selected = [item for item in metrics if item.get("growth") is not None]
    if not selected:
        return []
    selected = sorted(selected, key=lambda item: float(item["growth"]))[:6] + sorted(
        selected, key=lambda item: float(item["growth"]), reverse=True
    )[:6]
    seen = set()
    deduped = []
    for item in selected:
        if item["label"] in seen:
            continue
        seen.add(item["label"])
        deduped.append(item)

    colors = ["#c1121f" if float(item["growth"]) < 0 else "#2a9d8f" for item in deduped]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh([item["label"] for item in deduped], [float(item["growth"]) * 100 for item in deduped], color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_title("Emissions Growth Rate by District")
    ax.set_xlabel("Growth (%)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    path = output_dir / "growth_by_district.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"title": "Emissions Growth Rate by District", "path": str(path)}]


def _stacked_components_chart(plt, metrics: list[dict], output_dir: Path) -> list[dict[str, str]]:
    selected = sorted(metrics, key=lambda item: item["total_emission"], reverse=True)[:10]
    labels = [item["label"] for item in selected]
    gas = [item["gas_emission"] for item in selected]
    electricity = [item["electricity_emission"] for item in selected]
    direct = [item["direct_emissions"] for item in selected]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, gas, label="Natural gas", color="#264653")
    ax.barh(labels, electricity, left=gas, label="Electricity", color="#e9c46a")
    left = [g + e for g, e in zip(gas, electricity)]
    ax.barh(labels, direct, left=left, label="Direct emissions", color="#e76f51")
    ax.invert_yaxis()
    ax.set_title("Emissions Components")
    ax.set_xlabel("Emissions")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    path = output_dir / "emission_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [{"title": "Emissions Components", "path": str(path)}]
