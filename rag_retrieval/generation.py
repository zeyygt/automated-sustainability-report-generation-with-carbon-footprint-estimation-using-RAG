"""Source-grounded AI report content generation."""

from __future__ import annotations

import json
import os
from typing import Any

from .insight_engine import build_report_insights
from .recommendation_engine import build_report_recommendations
from .report_metrics import public_metrics
from .report_models import GeneratedReport, ReportInput


DEFAULT_MODEL = "gpt-4o"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_OUTPUT_TOKENS = 1200


class OpenAIReportGenerator:
    """Generate narrative report sections from structured retrieval outputs."""

    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
    ) -> None:
        _load_local_env()
        self.model = model if model is not None else os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else os.getenv("OPENAI_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
        )
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))

    def generate(self, report_input: ReportInput) -> GeneratedReport:
        warnings: list[str] = []
        if not self.api_key:
            warnings.append("OPENAI_API_KEY not set; deterministic fallback content was used.")
            content = deterministic_report_content(report_input)
        else:
            try:
                content = self._generate_with_openai(report_input)
            except Exception as exc:  # pragma: no cover - network/API dependent
                warnings.append(f"OpenAI generation failed; deterministic fallback content was used: {exc}")
                content = deterministic_report_content(report_input)

        return GeneratedReport(
            title=report_input.title,
            language=report_input.language,
            generated_at=report_input.generated_at,
            report_input=report_input,
            ai_content_markdown=content,
            warnings=[*report_input.warnings, *warnings],
        )

    def _generate_with_openai(self, report_input: ReportInput) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        system_content = _system_prompt(report_input.language)
        user_content = json.dumps(_compact_report_payload(report_input), ensure_ascii=False)
        if _uses_chat_completions(self.model):
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        request = {
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "input": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }
        if _supports_reasoning(self.model):
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = client.responses.create(**request)
        return getattr(response, "output_text", None) or _response_text(response)


def deterministic_report_content(report_input: ReportInput) -> str:
    metrics = public_metrics(report_input.structured_results)
    insights = _resolved_insights(report_input, metrics)
    recommendations = _resolved_recommendations(report_input, metrics, insights)
    coverage_notes = _public_coverage_notes(report_input.warnings, report_input.language)
    if _is_english(report_input.language):
        return _deterministic_report_content_en(metrics, insights, recommendations, coverage_notes)

    return _deterministic_report_content_tr(metrics, insights, recommendations, coverage_notes)


def _deterministic_report_content_en(
    metrics: list[dict],
    insights: dict[str, Any],
    recommendations: dict[str, Any],
    coverage_notes: list[str],
) -> str:
    municipality = insights.get("municipality") or {}
    water_metrics = [item for item in metrics if float(item.get("water_consumption") or 0.0) > 0.0]
    water_section_lines = _section_metric_lines(metrics, "Water Overview", exclude={"water", "electricity", "natural_gas"})
    resource_lines = _section_metric_lines(metrics, "Resource Overview", exclude={"water", "electricity", "natural_gas"})
    context_lines = _section_metric_lines(metrics, "District Context and Sustainability Signals", exclude={"water", "electricity", "natural_gas"})
    district_commentary = list(recommendations.get("priority_district_commentary") or [])
    focus_lines = list((recommendations.get("municipality_focus") or {}).get("en") or [])
    strategy_lines = list(recommendations.get("strategic_recommendations") or [])
    data_quality_notes = [item.get("note_en") for item in (recommendations.get("data_quality_notes") or []) if item.get("note_en")]
    lines = ["# Executive Summary"]
    headlines = list(insights.get("headlines") or [])
    if headlines:
        lines.extend(headlines[:5])
    else:
        lines.extend(
            [
                "This sustainability report summarizes district-level energy, water, emissions, and contextual sustainability indicators for municipal decision-making.",
                "The assessment highlights high-emission districts, water-use patterns, detected resource metrics, contextual sustainability signals, and priority actions for resource management.",
            ]
        )
    if coverage_notes:
        lines.append(f"Coverage remains partially constrained: {coverage_notes[0]}")

    lines.extend(
        [
            "",
            "# Municipality-Wide Assessment",
        ]
    )
    if municipality.get("district_count"):
        lines.append(f"- The current assessment covers {int(municipality['district_count'])} districts.")
        if float(municipality.get("total_emissions") or 0.0) > 0.0:
            lines.append(
                f"- Combined reported emissions across the covered districts are {float(municipality['total_emissions']):,.2f}."
            )
        if float(municipality.get("total_water_consumption") or 0.0) > 0.0:
            lines.append(
                f"- Combined reported water consumption reaches {float(municipality['total_water_consumption']):,.2f} m3."
            )
        highest_emission = municipality.get("highest_emission_district")
        if highest_emission:
            lines.append(
                f"- {highest_emission['district']} is the highest-emission district in the uploaded dataset at {float(highest_emission['value']):,.2f}."
            )
        highest_water = municipality.get("highest_water_district")
        if highest_water:
            lines.append(
                f"- {highest_water['district']} has the highest recorded water demand at {float(highest_water['value']):,.2f} m3."
            )
        lines.append(f"- Overall reporting coverage is assessed as {municipality.get('coverage_level', 'moderate')}.")
        for focus in focus_lines[:3]:
            lines.append(f"- {focus}")
    else:
        lines.append("- No municipality-wide structured dataset was available for assessment.")

    lines.extend(["", "# Emissions Overview"])
    emission_rank = list(insights.get("highest_emission_districts") or [])
    if emission_rank:
        for item in emission_rank[:5]:
            metric = _metric_by_district(metrics, item.get("district"))
            lines.append(
                "- {district}: total emissions {total:,.2f}, growth {growth}.".format(
                    district=item["district"],
                    total=float(item.get("value") or 0.0),
                    growth=_format_growth(metric.get("growth") if metric else item.get("growth")),
                )
            )
    elif metrics:
        for item in metrics[:10]:
            lines.append(
                "- {district}: total emissions {total:,.2f}, growth {growth}.".format(
                    district=item["district"],
                    total=item["total_emission"],
                    growth=_format_growth(item.get("growth")),
                )
            )
    else:
        lines.append("- No district-level structured metrics were available.")

    emission_outliers = [item for item in (insights.get("outliers") or []) if item.get("metric") == "total_emission"]
    if emission_outliers:
        labels = ", ".join(f"{item['district']} ({float(item['value']):,.2f})" for item in emission_outliers[:4])
        lines.append(f"- Emission outliers requiring closer review include {labels}.")

    lines.extend(["", "# Water Overview"])
    water_rank = list(insights.get("highest_water_districts") or [])
    if water_rank:
        for item in water_rank[:5]:
            metric = _metric_by_district(metrics, item.get("district"))
            lines.append(
                "- {district}: water consumption {water:,.2f} m3, growth {growth}.".format(
                    district=item["district"],
                    water=float(item.get("value") or 0.0),
                    growth=_format_growth(metric.get("water_growth") if metric else None),
                )
            )
        lines.extend(water_section_lines[:6])
    elif water_metrics:
        for item in water_metrics[:10]:
            lines.append(
                "- {district}: water consumption {water:,.2f} m3, growth {growth}.".format(
                    district=item["district"],
                    water=float(item.get("water_consumption") or 0.0),
                    growth=_format_growth(item.get("water_growth")),
                )
            )
        lines.extend(water_section_lines[:6])
    else:
        lines.append("- Water consumption data was not available in the uploaded sustainability dataset.")

    water_outliers = [item for item in (insights.get("outliers") or []) if item.get("metric") == "water_consumption"]
    if water_outliers:
        labels = ", ".join(f"{item['district']} ({float(item['value']):,.2f} m3)" for item in water_outliers[:4])
        lines.append(f"- Water-demand outliers include {labels}.")

    lines.extend(["", "# Resource Overview"])
    if resource_lines:
        lines.extend(resource_lines)
    else:
        lines.append("- No additional district-level resource metrics were detected beyond the core energy and water inputs.")

    lines.extend(["", "# District Context and Sustainability Signals"])
    context_highlights = list(insights.get("context_highlights") or [])
    if context_highlights:
        for item in context_highlights[:6]:
            rendered_value = f"{float(item['value']):,.2f}"
            unit = f" {item['unit']}" if item.get("unit") else ""
            lines.append(f"- {item['district']}: {item['metric_label']} {rendered_value}{unit}.")
    elif context_lines:
        lines.extend(context_lines)
    else:
        lines.append("- No additional contextual sustainability indicators were detected in the uploaded files.")

    lines.extend(["", "# Priority Districts"])
    priority_districts = list(insights.get("priority_districts") or [])
    if priority_districts:
        for item in priority_districts[:6]:
            lines.append(f"- {item['district']}: {_priority_reason_en(item)}.")
    else:
        lines.append("- No district-level priority ranking could be derived from the current dataset.")

    lines.extend(["", "# District Commentary"])
    if district_commentary:
        for item in district_commentary[:6]:
            lines.append(f"## {item['district']} - {item['archetype_label_en']}")
            lines.append(f"- {item['headline_en']}")
            lines.append(f"- {item['summary_en']}")
            actions = list(item.get("recommended_actions_en") or [])
            if actions:
                lines.append(f"- Recommended next steps: {'; '.join(actions[:2])}.")
    else:
        improving = [item for item in (insights.get("improving_districts") or []) if float(item.get("value") or 0.0) < 0.0]
        if improving:
            improving_labels = ", ".join(
                f"{item['district']} ({float(item['value']) * 100:.2f}%)" for item in improving[:4]
            )
            lines.append(f"- Districts showing the strongest improvement trend include {improving_labels}.")
        else:
            lines.append("- Most districts require continued monitoring because clear improvement trends are limited or not yet available.")

    lines.extend(["", "# Strategic Recommendations"])
    if strategy_lines:
        for item in strategy_lines[:5]:
            lines.append(f"- {item['title_en']}: {item['rationale_en']}")
    else:
        lines.extend(_priority_actions_en(insights, coverage_notes))

    lines.extend(["", "# Data Quality and Coverage Notes"])
    if coverage_notes or data_quality_notes:
        for note in [*coverage_notes, *data_quality_notes]:
            lines.append(f"- {note}")
    else:
        lines.append("- Coverage is sufficient for a directional district comparison, but continued multi-period collection will strengthen future reporting.")
    return "\n".join(lines)


def _deterministic_report_content_tr(
    metrics: list[dict],
    insights: dict[str, Any],
    recommendations: dict[str, Any],
    coverage_notes: list[str],
) -> str:
    municipality = insights.get("municipality") or {}
    water_metrics = [item for item in metrics if float(item.get("water_consumption") or 0.0) > 0.0]
    water_section_lines = _section_metric_lines(metrics, "Water Overview", exclude={"water", "electricity", "natural_gas"})
    resource_lines = _section_metric_lines(metrics, "Resource Overview", exclude={"water", "electricity", "natural_gas"})
    context_lines = _section_metric_lines(metrics, "District Context and Sustainability Signals", exclude={"water", "electricity", "natural_gas"})
    district_commentary = list(recommendations.get("priority_district_commentary") or [])
    focus_lines = list((recommendations.get("municipality_focus") or {}).get("tr") or [])
    strategy_lines = list(recommendations.get("strategic_recommendations") or [])
    data_quality_notes = [item.get("note_tr") for item in (recommendations.get("data_quality_notes") or []) if item.get("note_tr")]
    lines = [
        "# Yönetici Özeti",
        "Bu sürdürülebilirlik raporu, ilçe bazlı enerji, su, emisyon ve bağlamsal sürdürülebilirlik göstergelerini karar vericiler için özetler.",
        "Analiz, yüksek emisyonlu alanları, su kullanım örüntülerini, tespit edilen ek kaynak metriklerini, bağlamsal göstergeleri ve öncelikli iyileştirme başlıklarını görünür kılar.",
    ]
    if coverage_notes:
        lines.append(f"Kapsam sınırlılığı notu: {coverage_notes[0]}")

    lines.extend(["", "# Belediye Geneli Değerlendirme"])
    if municipality.get("district_count"):
        lines.append(f"- Mevcut değerlendirme {int(municipality['district_count'])} ilçeyi kapsamaktadır.")
        if float(municipality.get("total_emissions") or 0.0) > 0.0:
            lines.append(f"- Kapsanan ilçelerde birleşik toplam emisyon {float(municipality['total_emissions']):,.2f} düzeyindedir.")
        if float(municipality.get("total_water_consumption") or 0.0) > 0.0:
            lines.append(f"- Birleşik toplam su tüketimi {float(municipality['total_water_consumption']):,.2f} m3 seviyesindedir.")
        highest_emission = municipality.get("highest_emission_district")
        if highest_emission:
            lines.append(f"- {highest_emission['district']}, {float(highest_emission['value']):,.2f} ile veri setindeki en yüksek emisyonlu ilçedir.")
        highest_water = municipality.get("highest_water_district")
        if highest_water:
            lines.append(f"- {highest_water['district']}, {float(highest_water['value']):,.2f} m3 ile en yüksek su talebine sahiptir.")
        lines.append(
            f"- Genel veri kapsayıcılığı seviyesi {_coverage_level_label_tr(str(municipality.get('coverage_level') or 'moderate'))} olarak değerlendirilmektedir."
        )
        for focus in focus_lines[:3]:
            lines.append(f"- {focus}")
    else:
        lines.append("- Belediye geneli değerlendirme için yeterli yapılandırılmış veri bulunamadı.")

    lines.extend(["", "# Emisyon Görünümü"])
    emission_rank = list(insights.get("highest_emission_districts") or [])
    if emission_rank:
        for item in emission_rank[:5]:
            metric = _metric_by_district(metrics, item.get("district"))
            lines.append(
                "- {district}: toplam emisyon {total:,.2f}, büyüme {growth}.".format(
                    district=item["district"],
                    total=float(item.get("value") or 0.0),
                    growth=_format_growth(metric.get("growth") if metric else item.get("growth")),
                )
            )
    elif metrics:
        for item in metrics[:8]:
            lines.append(
                "- {district}: toplam emisyon {total:,.2f}, büyüme {growth}.".format(
                    district=item["district"],
                    total=item["total_emission"],
                    growth=_format_growth(item.get("growth")),
                )
            )
    else:
        lines.append("- İlçe bazlı yapılandırılmış metrik bulunamadı.")

    emission_outliers = [item for item in (insights.get("outliers") or []) if item.get("metric") == "total_emission"]
    if emission_outliers:
        labels = ", ".join(f"{item['district']} ({float(item['value']):,.2f})" for item in emission_outliers[:4])
        lines.append(f"- Yakın inceleme gerektiren emisyon aykırı değerleri: {labels}.")

    lines.extend(["", "# Su Görünümü"])
    water_rank = list(insights.get("highest_water_districts") or [])
    if water_rank:
        for item in water_rank[:5]:
            metric = _metric_by_district(metrics, item.get("district"))
            lines.append(
                "- {district}: su tüketimi {water:,.2f} m3, büyüme {growth}.".format(
                    district=item["district"],
                    water=float(item.get("value") or 0.0),
                    growth=_format_growth(metric.get("water_growth") if metric else None),
                )
            )
        lines.extend(water_section_lines[:6])
    elif water_metrics:
        for item in water_metrics[:8]:
            lines.append(
                "- {district}: su tüketimi {water:,.2f} m3, büyüme {growth}.".format(
                    district=item["district"],
                    water=float(item.get("water_consumption") or 0.0),
                    growth=_format_growth(item.get("water_growth")),
                )
            )
        lines.extend(water_section_lines[:6])
    else:
        lines.append("- Yüklenen veri setinde su tüketimi bilgisi bulunmadı.")

    water_outliers = [item for item in (insights.get("outliers") or []) if item.get("metric") == "water_consumption"]
    if water_outliers:
        labels = ", ".join(f"{item['district']} ({float(item['value']):,.2f} m3)" for item in water_outliers[:4])
        lines.append(f"- Su talebinde öne çıkan aykırı ilçeler: {labels}.")

    lines.extend(["", "# Kaynak Görünümü"])
    if resource_lines:
        lines.extend(resource_lines)
    else:
        lines.append("- Çekirdek enerji ve su verileri dışında ek ilçe bazlı kaynak metriği tespit edilmedi.")

    lines.extend(["", "# İlçe Bağlamı ve Sürdürülebilirlik Sinyalleri"])
    context_highlights = list(insights.get("context_highlights") or [])
    if context_highlights:
        for item in context_highlights[:6]:
            rendered_value = f"{float(item['value']):,.2f}"
            unit = f" {item['unit']}" if item.get("unit") else ""
            lines.append(f"- {item['district']}: {item['metric_label']} {rendered_value}{unit}.")
    elif context_lines:
        lines.extend(context_lines)
    else:
        lines.append("- Yüklenen dosyalarda ek bağlamsal sürdürülebilirlik göstergesi tespit edilmedi.")

    lines.extend(["", "# Öncelikli İlçeler"])
    priority_districts = list(insights.get("priority_districts") or [])
    if priority_districts:
        for item in priority_districts[:6]:
            lines.append(f"- {item['district']}: {_priority_reason_tr(item)}.")
    else:
        lines.append("- Mevcut veriyle güvenilir bir ilçe önceliklendirmesi üretilemedi.")

    lines.extend(["", "# İlçe Yorumları"])
    if district_commentary:
        for item in district_commentary[:6]:
            lines.append(f"## {item['district']} - {item['archetype_label_tr']}")
            lines.append(f"- {item['headline_tr']}")
            lines.append(f"- {item['summary_tr']}")
            actions = list(item.get("recommended_actions_tr") or [])
            if actions:
                lines.append(f"- Önerilen adımlar: {'; '.join(actions[:2])}.")
    else:
        improving = [item for item in (insights.get("improving_districts") or []) if float(item.get("value") or 0.0) < 0.0]
        if improving:
            improving_labels = ", ".join(
                f"{item['district']} ({float(item['value']) * 100:.2f}%)" for item in improving[:4]
            )
            lines.append(f"- En belirgin iyileşme eğilimi görülen ilçeler: {improving_labels}.")
        else:
            lines.append("- Açık iyileşme eğilimi gösteren ilçe sayısı sınırlı olduğundan izleme ihtiyacı sürmektedir.")

    lines.extend(["", "# Stratejik Öneriler"])
    if strategy_lines:
        for item in strategy_lines[:5]:
            lines.append(f"- {item['title_tr']}: {item['rationale_tr']}")
    else:
        lines.extend(_priority_actions_tr(insights, coverage_notes))

    lines.extend(["", "# Veri Kalitesi ve Kapsam Notları"])
    if coverage_notes or data_quality_notes:
        for note in [*coverage_notes, *data_quality_notes]:
            lines.append(f"- {note}")
    else:
        lines.append("- Mevcut kapsam yön gösterici ilçe karşılaştırması için yeterlidir; ancak düzenli çok dönemli veri toplama gelecekteki raporları güçlendirecektir.")
    return "\n".join(lines)


def _system_prompt(language: str) -> str:
    headings = (
        "Executive Summary, Municipality-Wide Assessment, Emissions Overview, Water Overview, Resource Overview, District Context and Sustainability Signals, Priority Districts, District Commentary, Strategic Recommendations, Data Quality and Coverage Notes"
        if _is_english(language)
        else "Yönetici Özeti, Belediye Geneli Değerlendirme, Emisyon Görünümü, Su Görünümü, Kaynak Görünümü, İlçe Bağlamı ve Sürdürülebilirlik Sinyalleri, Öncelikli İlçeler, İlçe Yorumları, Stratejik Öneriler, Veri Kalitesi ve Kapsam Notları"
    )
    return (
        "You are an expert sustainability report writer. "
        f"Write the report in {language}. "
        "Write as a public-facing municipal sustainability report, not as a technical system report. "
        "Keep the narrative concise but substantive; the report should read like a professional municipal assessment, not a generic summary. "
        "Use public_metrics as the source of truth for all numeric values. "
        "Use the supplied insights and recommendations to explain the municipality-wide picture, identify priority districts, and justify why those districts matter. "
        "Write Municipality-Wide Assessment as an analytical synthesis of concentration, trend direction, and coverage limits, not as a restatement of the table. "
        "Resource Overview must use only resource_metrics. If resource_metrics is empty, explicitly say that no additional district-level resource metrics were detected beyond the core energy and water inputs. "
        "District Context and Sustainability Signals must use only context_metrics and must not be merged into Resource Overview. "
        "In District Commentary, write concrete district-level interpretation instead of repeating raw numbers alone. "
        "Choose districts that illustrate different patterns when possible: current emissions core, fast-growth frontier, ecological context, lower-pressure baseline, direct-emission profile, or data gaps. "
        "Do not repeat the same explanation template across district commentaries; each district should have a distinct analytical angle and a distinct practical implication. "
        "Use additional non-emission sustainability metrics when they are present; contextual indicators should shape the narrative even when they are not part of the carbon formula. "
        "Comment on general municipal status, district-level implications, practical recommendations, and any observable tradeoffs between emissions, water, and contextual indicators. "
        "For growth values, use growth_display exactly as provided when present; never reinterpret raw growth as a different percentage. "
        "Do not mention source filenames, PDFs, Excel files, parsers, retrieval, RAG, DataEngine, extraction, chunks, prompts, or methodology. "
        "Do not invent numbers, districts, years, or recommendations unsupported by the input. "
        "Do not recalculate emissions. "
        "Avoid saying where a value was extracted from. "
        "If data coverage warnings exist, convert them into natural municipal reporting caveats without technical details. "
        f"Return Markdown with these headings only: {headings}."
    )


def _compact_report_payload(report_input: ReportInput) -> dict[str, Any]:
    metrics = public_metrics(report_input.structured_results)
    insights = _resolved_insights(report_input, metrics)
    recommendations = _resolved_recommendations(report_input, metrics, insights)
    active_metric_keys = _active_metric_keys(metrics)
    return {
        "title": report_input.title,
        "language": report_input.language,
        "district_count": len(metrics),
        "public_metrics_summary": _public_metrics_summary(metrics),
        "insights": _compact_insights(insights),
        "recommendations": _compact_recommendations(recommendations),
        "detected_metrics": _detected_metric_catalog(report_input.detected_metrics, active_metric_keys),
        "resource_metrics": _section_payload(metrics, "Resource Overview")[:8],
        "context_metrics": _section_payload(metrics, "District Context and Sustainability Signals")[:8],
        "coverage_notes": _public_coverage_notes(report_input.warnings, report_input.language),
    }


def _public_coverage_notes(warnings: list[str], language: str) -> list[str]:
    notes = []
    joined = " ".join(warnings)
    english = _is_english(language)
    if "electricity_consumption_not_found" in joined:
        notes.append(
            "Some district assessments are limited to the available energy categories because electricity consumption was not available."
            if english
            else "Bazı değerlendirmelerde elektrik tüketimi bulunmadığı için analiz mevcut enerji kalemleriyle sınırlıdır."
        )
    if "natural_gas_consumption_not_found" in joined:
        notes.append(
            "Some district assessments rely on electricity-focused indicators because natural gas consumption was not available."
            if english
            else "Bazı değerlendirmelerde doğalgaz tüketimi bulunmadığı için elektrik odaklı analiz yapılmıştır."
        )
    if "water_consumption_not_found" in joined:
        notes.append(
            "Some district assessments include limited water reporting because water consumption values were incomplete or missing."
            if english
            else "Bazı ilçe değerlendirmelerinde su tüketimi değerleri eksik olduğu için su raporlaması sınırlı kalmıştır."
        )
    if "direct_emissions_reported_separately" in joined:
        notes.append(
            "Directly reported emissions are treated separately to avoid double counting."
            if english
            else "Doğrudan raporlanan emisyon değerleri çift sayımı önlemek için ayrı değerlendirilmiştir."
        )
    if "custom_formula_missing_variable_definition" in joined or "custom_formula_missing_variable_value" in joined:
        notes.append(
            "A custom emissions formula was provided but could not be fully applied because some required variables were undefined or missing, so the standard calculation remained in use."
            if english
            else "Özel bir emisyon formülü sağlandı; ancak bazı gerekli değişkenler tanımlı olmadığı veya eksik olduğu için standart hesap yöntemi kullanılmaya devam edildi."
        )
    return notes


def _resolved_insights(report_input: ReportInput, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if report_input.insights:
        return report_input.insights
    return build_report_insights(
        metrics,
        warnings=report_input.warnings,
        detected_metrics=report_input.detected_metrics,
    )


def _resolved_recommendations(
    report_input: ReportInput,
    metrics: list[dict[str, Any]],
    insights: dict[str, Any],
) -> dict[str, Any]:
    if report_input.recommendations:
        return report_input.recommendations
    return build_report_recommendations(metrics, insights, warnings=report_input.warnings)


def _is_english(language: str) -> bool:
    return language.strip().lower().startswith("en")


def _format_growth(value) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value) * 100:.2f}%"


def _coverage_level_label_tr(value: str) -> str:
    mapping = {
        "high": "yüksek",
        "moderate": "orta",
        "low": "düşük",
    }
    return mapping.get(value.strip().lower(), value)


def _response_text(response) -> str:
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _metric_by_district(metrics: list[dict[str, Any]], district: str | None) -> dict[str, Any] | None:
    if not district:
        return None
    for item in metrics:
        if item.get("district") == district:
            return item
    return None


def _priority_reason_en(item: dict[str, Any]) -> str:
    reasons: list[str] = []
    total = float(item.get("total_emission") or 0.0)
    water = float(item.get("water_consumption") or 0.0)
    growth = item.get("growth")
    if total > 0.0:
        reasons.append(f"high total emissions ({total:,.2f})")
    if water > 0.0:
        reasons.append(f"high water demand ({water:,.2f} m3)")
    if growth is not None and float(growth) > 0.0:
        reasons.append(f"an increasing trend ({float(growth) * 100:.2f}%)")
    return ", ".join(reasons[:3]) or "requires closer monitoring"


def _priority_reason_tr(item: dict[str, Any]) -> str:
    reasons: list[str] = []
    total = float(item.get("total_emission") or 0.0)
    water = float(item.get("water_consumption") or 0.0)
    growth = item.get("growth")
    if total > 0.0:
        reasons.append(f"yüksek toplam emisyon ({total:,.2f})")
    if water > 0.0:
        reasons.append(f"yüksek su talebi ({water:,.2f} m3)")
    if growth is not None and float(growth) > 0.0:
        reasons.append(f"artan eğilim ({float(growth) * 100:.2f}%)")
    return ", ".join(reasons[:3]) or "yakın izleme gerektiriyor"


def _priority_actions_en(insights: dict[str, Any], coverage_notes: list[str]) -> list[str]:
    actions = [
        "- Prioritize efficiency and demand-management measures in the highest-emission districts.",
        "- Pair water-efficiency and leakage-reduction programs with districts showing the highest water demand.",
    ]
    context_highlights = list(insights.get("context_highlights") or [])
    if context_highlights:
        example = context_highlights[0]
        actions.append(
            f"- Use contextual indicators such as {example['metric_label']} in {example['district']} to tailor district-specific sustainability interventions."
        )
    else:
        actions.append("- Expand contextual sustainability indicators to better explain district-level performance differences.")
    if coverage_notes:
        actions.append("- Close remaining data coverage gaps before the next reporting cycle to improve confidence in district comparisons.")
    else:
        actions.append("- Maintain regular multi-period data collection so district trends remain comparable over time.")
    return actions


def _priority_actions_tr(insights: dict[str, Any], coverage_notes: list[str]) -> list[str]:
    actions = [
        "- En yüksek emisyonlu ilçelerde verimlilik ve talep yönetimi önlemleri önceliklendirilmelidir.",
        "- Su talebi en yüksek ilçelerde su verimliliği ve kaçak azaltımı programları birlikte ele alınmalıdır.",
    ]
    context_highlights = list(insights.get("context_highlights") or [])
    if context_highlights:
        example = context_highlights[0]
        actions.append(
            f"- {example['district']} ilçesindeki {example['metric_label']} gibi bağlamsal göstergeler, ilçe bazlı müdahaleleri özelleştirmek için kullanılmalıdır."
        )
    else:
        actions.append("- İlçeler arası performans farklarını daha iyi açıklamak için bağlamsal sürdürülebilirlik göstergeleri genişletilmelidir.")
    if coverage_notes:
        actions.append("- Bir sonraki raporlama döneminden önce veri kapsamı boşlukları kapatılarak ilçe karşılaştırmalarının güvenilirliği artırılmalıdır.")
    else:
        actions.append("- İlçe trendlerinin karşılaştırılabilir kalması için düzenli çok dönemli veri toplama sürdürülmelidir.")
    return actions


def _section_metric_lines(metrics: list[dict], section: str, *, exclude: set[str]) -> list[str]:
    lines: list[str] = []
    for item in metrics[:15]:
        summaries = []
        for metric_key, summary in (item.get("metric_summaries") or {}).items():
            if metric_key in exclude:
                continue
            if summary.get("report_section") != section:
                continue
            value = float(summary.get("value") or 0.0)
            if value <= 0.0:
                continue
            unit = str(summary.get("unit") or "").strip()
            label = str(summary.get("label") or metric_key.replace("_", " ").title())
            rendered_value = f"{value:,.2f}"
            summaries.append(f"{label} {rendered_value}{(' ' + unit) if unit else ''}")
        if summaries:
            lines.append(f"- {item['district']}: {'; '.join(summaries[:3])}.")
    return lines[:10]


def _section_payload(metrics: list[dict], section: str) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in metrics:
        section_metrics = []
        for metric_key, summary in (item.get("metric_summaries") or {}).items():
            if summary.get("report_section") != section:
                continue
            value = float(summary.get("value") or 0.0)
            if value <= 0.0:
                continue
            section_metrics.append(
                {
                    "metric_key": metric_key,
                    "label": summary.get("label"),
                    "value": value,
                    "unit": summary.get("unit"),
                    "role": summary.get("role"),
                    "category": summary.get("category"),
                }
            )
        if section_metrics:
            payload.append({"district": item["district"], "metrics": section_metrics})
    return payload


def _public_metrics_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "top_emission_districts": [
            {
                "district": item.get("district"),
                "total_emission": item.get("total_emission"),
                "growth_display": item.get("growth_display"),
            }
            for item in metrics
            if float(item.get("total_emission") or 0.0) > 0.0
        ][:8],
        "top_water_districts": [
            {
                "district": item.get("district"),
                "water_consumption": item.get("water_consumption"),
                "water_growth": item.get("water_growth"),
            }
            for item in sorted(metrics, key=lambda item: float(item.get("water_consumption") or 0.0), reverse=True)
            if float(item.get("water_consumption") or 0.0) > 0.0
        ][:8],
        "district_snapshots": [
            {
                "district": item.get("district"),
                "total_emission": item.get("total_emission"),
                "water_consumption": item.get("water_consumption"),
                "growth_display": item.get("growth_display"),
                "available_metric_keys": item.get("available_metric_keys"),
            }
            for item in metrics[:10]
        ],
    }


def _compact_insights(insights: dict[str, Any]) -> dict[str, Any]:
    return {
        "municipality": insights.get("municipality"),
        "headlines": list(insights.get("headlines") or [])[:5],
        "priority_districts": list(insights.get("priority_districts") or [])[:8],
        "highest_emission_districts": list(insights.get("highest_emission_districts") or [])[:5],
        "highest_water_districts": list(insights.get("highest_water_districts") or [])[:5],
        "context_highlights": list(insights.get("context_highlights") or [])[:6],
        "coverage": insights.get("coverage"),
    }


def _compact_recommendations(recommendations: dict[str, Any]) -> dict[str, Any]:
    return {
        "municipality_focus": recommendations.get("municipality_focus"),
        "strategic_recommendations": list(recommendations.get("strategic_recommendations") or [])[:5],
        "priority_district_commentary": [
            {
                "district": item.get("district"),
                "archetype_label_en": item.get("archetype_label_en"),
                "archetype_label_tr": item.get("archetype_label_tr"),
                "headline_en": item.get("headline_en"),
                "headline_tr": item.get("headline_tr"),
                "summary_en": item.get("summary_en"),
                "summary_tr": item.get("summary_tr"),
                "recommended_actions_en": list(item.get("recommended_actions_en") or [])[:3],
                "recommended_actions_tr": list(item.get("recommended_actions_tr") or [])[:3],
                "signals": list(item.get("signals") or [])[:6],
                "severity": item.get("severity"),
                "commentary_angle": item.get("commentary_angle"),
                "emission_rank": item.get("emission_rank"),
                "growth_rank": item.get("growth_rank"),
                "tree_rank": item.get("tree_rank"),
                "total_emission": item.get("total_emission"),
            }
            for item in list(recommendations.get("priority_district_commentary") or [])[:6]
        ],
        "data_quality_notes": list(recommendations.get("data_quality_notes") or [])[:5],
    }


def _detected_metric_catalog(detected_metrics: list[dict[str, Any]], active_metric_keys: set[str]) -> list[dict[str, Any]]:
    catalog = []
    for item in detected_metrics:
        if not item.get("sustainability_related"):
            continue
        metric_key = str(item.get("metric_key") or "")
        if metric_key not in active_metric_keys and not bool(item.get("used_in_calculation")):
            continue
        if float(item.get("numeric_availability") or 0.0) <= 0.0 and metric_key not in active_metric_keys:
            continue
        catalog.append(
            {
                "metric_key": metric_key,
                "display_name": item.get("display_name"),
                "category": item.get("category"),
                "role": item.get("role"),
                "report_section": item.get("report_section"),
            }
        )
    return catalog[:12]


def _active_metric_keys(metrics: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for item in metrics:
        if float(item.get("total_emission") or 0.0) > 0.0:
            keys.update({"electricity", "natural_gas"})
        if float(item.get("water_consumption") or 0.0) > 0.0:
            keys.add("water")
        for metric_key, summary in (item.get("metric_summaries") or {}).items():
            if float(summary.get("value") or 0.0) > 0.0:
                keys.add(str(metric_key))
    return keys


def _supports_reasoning(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def _uses_chat_completions(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith("gpt-4")


def _load_local_env(path: str = ".env") -> None:
    env_path = os.path.abspath(path)
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
