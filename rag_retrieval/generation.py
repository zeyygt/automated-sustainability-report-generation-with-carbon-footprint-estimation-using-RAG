"""Source-grounded AI report content generation."""

from __future__ import annotations

import json
import os
from typing import Any

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
    if _is_english(report_input.language):
        return _deterministic_report_content_en(metrics)

    return _deterministic_report_content_tr(metrics)


def _deterministic_report_content_en(metrics: list[dict]) -> str:
    lines = [
        "# Executive Summary",
        "This sustainability report summarizes district-level energy consumption and emissions indicators for municipal decision-making.",
        "The assessment highlights high-emission districts, observed trends, and priority actions for emissions reduction.",
        "",
        "# Emissions Overview",
    ]
    if metrics:
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

    lines.extend(
        [
            "",
            "# District Analysis",
            "Districts with the highest total emissions should be prioritized for reduction planning and operational review.",
            "Districts with positive growth require closer monitoring to identify the drivers of increasing consumption.",
            "",
            "# Trend Assessment",
            "Observed reductions should be supported with continued tracking, while missing data categories should be completed in future reporting cycles.",
            "",
            "# Priority Actions",
            "- Expand district-level energy monitoring to improve coverage and continuity.",
            "- Prioritize efficiency measures in districts with high total, per-capita, or per-household emissions.",
            "- Track natural gas and electricity consumption separately to measure annual reduction impact.",
        ]
    )
    return "\n".join(lines)


def _deterministic_report_content_tr(metrics: list[dict]) -> str:
    lines = [
        "# Yönetici Özeti",
        "Bu sürdürülebilirlik raporu, ilçe bazlı enerji tüketimi ve emisyon göstergelerini karar vericiler için özetler.",
        "Analiz, yüksek emisyonlu alanları, değişim eğilimlerini ve öncelikli iyileştirme başlıklarını görünür kılar.",
        "",
        "# Emisyon Görünümü",
    ]
    if metrics:
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

    lines.extend(
        [
            "",
            "# İlçe Analizi",
            "En yüksek toplam emisyon değerine sahip ilçeler, azaltım önceliklendirmesinde ilk incelenmesi gereken alanlardır.",
            "Pozitif büyüme gösteren ilçelerde tüketim artışının operasyonel nedenleri ayrıca değerlendirilmelidir.",
            "",
            "# Öncelikli Aksiyonlar",
            "- Emisyon artışı görülen ilçelerde enerji verimliliği ve altyapı iyileştirme programları önceliklendirilmelidir.",
            "- Kişi başı ve hane başı emisyonu yüksek ilçelerde hedefli azaltım planları hazırlanmalıdır.",
            "- Doğalgaz ve elektrik tüketimi ayrı ayrı izlenerek azaltım etkisi yıllık bazda takip edilmelidir.",
            "",
            "# Değerlendirme Notları",
        ]
    )
    lines.append("- Bazı ilçelerde değerlendirme yalnızca mevcut enerji kalemleri üzerinden yapılmıştır.")
    lines.append("- Doğrudan raporlanan emisyon değerleri çift sayımı önlemek için tüketimden hesaplanan emisyonlardan ayrı değerlendirilmiştir.")
    return "\n".join(lines)


def _system_prompt(language: str) -> str:
    headings = (
        "Executive Summary, Emissions Overview, District Analysis, Trend Assessment, Priority Actions"
        if _is_english(language)
        else "Yönetici Özeti, Emisyon Görünümü, İlçe Analizi, Trend Değerlendirmesi, Öncelikli Aksiyonlar"
    )
    return (
        "You are an expert sustainability report writer. "
        f"Write the report in {language}. "
        "Write as a public-facing municipal sustainability report, not as a technical system report. "
        "Keep the narrative concise; charts and the district indicator table will present detailed rows. "
        "Use public_metrics as the source of truth for all numeric values. "
        "For growth values, use growth_display exactly as provided; never reinterpret raw growth as a percentage. "
        "Do not mention source filenames, PDFs, Excel files, parsers, retrieval, RAG, DataEngine, extraction, chunks, prompts, or methodology. "
        "Do not invent numbers, districts, years, or recommendations unsupported by the input. "
        "Do not recalculate emissions. "
        "Avoid saying where a value was extracted from. "
        "If data coverage warnings exist, convert them into natural municipal reporting caveats without technical details. "
        f"Return Markdown with these headings only: {headings}."
    )


def _compact_report_payload(report_input: ReportInput) -> dict[str, Any]:
    return {
        "title": report_input.title,
        "language": report_input.language,
        "district_count": len(public_metrics(report_input.structured_results)),
        "public_metrics": public_metrics(report_input.structured_results),
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
    if "direct_emissions_reported_separately" in joined:
        notes.append(
            "Directly reported emissions are treated separately to avoid double counting."
            if english
            else "Doğrudan raporlanan emisyon değerleri çift sayımı önlemek için ayrı değerlendirilmiştir."
        )
    return notes


def _is_english(language: str) -> bool:
    return language.strip().lower().startswith("en")


def _format_growth(value) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value) * 100:.2f}%"


def _response_text(response) -> str:
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


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
