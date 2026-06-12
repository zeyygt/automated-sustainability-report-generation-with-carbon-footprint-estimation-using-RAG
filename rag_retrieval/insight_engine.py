"""Deterministic municipal insight extraction for sustainability reporting."""

from __future__ import annotations

from statistics import median, mean, pstdev
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

    analytics = _analytics(metrics, emission_metrics)
    plausibility = _plausibility_flags(metrics, emission_metrics, analytics)
    emission_unit = _emission_unit(emission_metrics)

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
        "emission_unit": emission_unit,
        "top3_emission_share": analytics["concentration"].get("top3_share"),
        "concentration_class": analytics["concentration"].get("classification"),
        "dominant_energy_lever": analytics["energy_lever"].get("dominant"),
        "highest_intensity_district": analytics["intensity"].get("top"),
    }

    analytical_findings = _analytical_findings(
        analytics, municipality, plausibility, emission_unit
    )

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
        "analytics": analytics,
        "analytical_findings": analytical_findings,
        "coverage": {
            "level": municipality["coverage_level"],
            "flags": low_coverage,
            "plausibility_flags": plausibility,
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


# ---------------------------------------------------------------------------
# Analytical layer: turns the per-district table into municipality-level
# findings (concentration, dominant lever, per-capita intensity, correlations)
# and surfaces plausibility flags. Everything here is deterministic so the
# narrative layer can cite it without re-deriving numbers.
# ---------------------------------------------------------------------------

CORE_METRIC_KEYS = {"electricity", "natural_gas", "water"}


def _emission_unit(emission_metrics: list[dict[str, Any]]) -> str:
    for item in emission_metrics:
        unit = str(item.get("emission_unit") or "").strip()
        if unit:
            return unit
    return "kgCO2e"


def _analytics(metrics: list[dict[str, Any]], emission_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "concentration": _concentration(emission_metrics),
        "energy_lever": _energy_lever(metrics),
        "intensity": _intensity(metrics),
        "distribution": _distribution_shape(emission_metrics),
        "correlations": _metric_correlations(metrics, emission_metrics),
    }


def _concentration(emission_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        (float(item.get("total_emission") or 0.0) for item in emission_metrics),
        reverse=True,
    )
    total = sum(values)
    if total <= 0.0 or not values:
        return {"top1_share": None, "top3_share": None, "hhi": None, "classification": "unknown"}
    shares = [value / total for value in values]
    top1_share = shares[0]
    top3_share = sum(shares[:3])
    hhi = sum(share * share for share in shares)
    if hhi >= 0.15 or top3_share >= 0.5:
        classification = "concentrated"
    elif top3_share <= 0.3:
        classification = "dispersed"
    else:
        classification = "moderate"
    return {
        "top1_share": round(top1_share, 4),
        "top3_share": round(top3_share, 4),
        "hhi": round(hhi, 4),
        "classification": classification,
        "district_count": len(values),
    }


def _energy_lever(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    electricity = sum(float(item.get("electricity_emission") or 0.0) for item in metrics)
    gas = sum(float(item.get("gas_emission") or 0.0) for item in metrics)
    energy_total = electricity + gas
    if energy_total <= 0.0:
        return {"electricity_share": None, "gas_share": None, "dominant": None}
    electricity_share = electricity / energy_total
    gas_share = gas / energy_total
    dominant = "electricity" if electricity_share >= gas_share else "natural_gas"
    return {
        "electricity_share": round(electricity_share, 4),
        "gas_share": round(gas_share, 4),
        "dominant": dominant,
    }


def _intensity(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        (
            {
                "district": item.get("district"),
                "per_capita": float(item.get("per_capita")),
                "total_emission": float(item.get("total_emission") or 0.0),
            }
            for item in metrics
            if item.get("per_capita") is not None and float(item.get("per_capita")) > 0.0
        ),
        key=lambda entry: entry["per_capita"],
        reverse=True,
    )
    if not ranked:
        return {"available": False, "ranked": [], "top": None, "spread_ratio": None, "diverges_from_absolute": False}

    absolute_top = max(metrics, key=lambda item: float(item.get("total_emission") or 0.0)).get("district")
    intensity_top = ranked[0]["district"]
    spread_ratio = ranked[0]["per_capita"] / ranked[-1]["per_capita"] if ranked[-1]["per_capita"] > 0 else None
    return {
        "available": True,
        "ranked": ranked[:8],
        "top": {"district": intensity_top, "per_capita": round(ranked[0]["per_capita"], 2)},
        "bottom": {"district": ranked[-1]["district"], "per_capita": round(ranked[-1]["per_capita"], 2)},
        "spread_ratio": round(spread_ratio, 1) if spread_ratio is not None else None,
        "diverges_from_absolute": bool(absolute_top and intensity_top and absolute_top != intensity_top),
        "absolute_top": absolute_top,
    }


def _distribution_shape(emission_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        (float(item.get("total_emission") or 0.0) for item in emission_metrics if float(item.get("total_emission") or 0.0) > 0.0),
        reverse=True,
    )
    if len(values) < 4:
        return {"coefficient_of_variation": None, "near_uniform": False, "near_linear": False}
    avg = mean(values)
    cv = (pstdev(values) / avg) if avg > 0 else None
    diffs = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    diff_mean = mean(diffs) if diffs else 0.0
    # Real-world emission tables are rarely evenly spaced; near-constant steps
    # between ranked districts usually mean synthetic or modeled figures.
    near_linear = bool(diffs and diff_mean > 0 and (pstdev(diffs) / diff_mean) < 0.1)
    return {
        "coefficient_of_variation": round(cv, 4) if cv is not None else None,
        "near_uniform": bool(cv is not None and cv < 0.35),
        "near_linear": near_linear,
    }


def _metric_correlations(metrics: list[dict[str, Any]], emission_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(emission_metrics) < 5:
        return []
    metric_keys: set[str] = set()
    for item in metrics:
        for metric_key, summary in (item.get("metric_summaries") or {}).items():
            if metric_key in CORE_METRIC_KEYS:
                continue
            if float(summary.get("value") or 0.0) > 0.0:
                metric_keys.add(metric_key)

    correlations: list[dict[str, Any]] = []
    for metric_key in sorted(metric_keys):
        pairs = []
        label = metric_key.replace("_", " ").title()
        for item in metrics:
            summary = (item.get("metric_summaries") or {}).get(metric_key) or {}
            value = float(summary.get("value") or 0.0)
            emission = float(item.get("total_emission") or 0.0)
            if value > 0.0 and emission > 0.0:
                pairs.append((value, emission))
                label = str(summary.get("label") or label)
        if len(pairs) < 5:
            continue
        coefficient = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
        if coefficient is None:
            continue
        correlations.append(
            {
                "metric_key": metric_key,
                "metric_label": label,
                "coefficient": round(coefficient, 3),
                "sample_size": len(pairs),
                "strong": abs(coefficient) >= 0.9,
            }
        )
    return sorted(correlations, key=lambda entry: abs(entry["coefficient"]), reverse=True)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def _plausibility_flags(
    metrics: list[dict[str, Any]],
    emission_metrics: list[dict[str, Any]],
    analytics: dict[str, Any],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    growth_values = [float(item.get("growth")) for item in metrics if item.get("growth") is not None]
    if growth_values:
        positive = [value for value in growth_values if value > 0.0]
        if len(positive) >= max(3, len(growth_values) // 2) and median(growth_values) > 0.75:
            flags.append(
                {
                    "code": "implausible_growth",
                    "note_en": f"Reported year-over-year growth is implausibly high across most districts (median {median(growth_values) * 100:.0f}%); the trend column should be treated as directional until its unit is confirmed.",
                    "note_tr": f"Raporlanan yıllık büyüme ilçelerin çoğunda makul olmayacak kadar yüksektir (medyan %{median(growth_values) * 100:.0f}); birimi doğrulanana kadar bu eğilim yalnızca yön gösterici olarak ele alınmalıdır.",
                }
            )

    intensity = analytics.get("intensity") or {}
    spread_ratio = intensity.get("spread_ratio")
    if spread_ratio is not None and spread_ratio > 50:
        flags.append(
            {
                "code": "implausible_intensity_spread",
                "note_en": f"Per-capita emissions span a {spread_ratio:.0f}x range across districts, which is unusually wide and suggests the consumption figures may not be normalized to population.",
                "note_tr": f"Kişi başı emisyon ilçeler arasında {spread_ratio:.0f} kat değişmektedir; bu olağandışı geniş aralık tüketim değerlerinin nüfusa göre normalize edilmemiş olabileceğini düşündürmektedir.",
            }
        )

    distribution = analytics.get("distribution") or {}
    if distribution.get("near_linear"):
        flags.append(
            {
                "code": "near_linear_distribution",
                "note_en": "District emission totals are almost perfectly evenly spaced, which is rare in real measured data and may indicate modeled or synthetic figures.",
                "note_tr": "İlçe emisyon toplamları neredeyse eşit aralıklarla dağılmıştır; gerçek ölçüm verilerinde nadir görülen bu durum, değerlerin modellenmiş veya sentetik olabileceğine işaret edebilir.",
            }
        )

    for correlation in analytics.get("correlations") or []:
        if correlation.get("strong"):
            flags.append(
                {
                    "code": "spurious_context_correlation",
                    "note_en": f"{correlation['metric_label']} tracks total emissions almost perfectly (r={correlation['coefficient']:.2f}); it largely reflects district size rather than an independent sustainability signal.",
                    "note_tr": f"{correlation['metric_label']} toplam emisyonla neredeyse birebir hareket etmektedir (r={correlation['coefficient']:.2f}); bu gösterge bağımsız bir sürdürülebilirlik sinyalinden çok ilçe büyüklüğünü yansıtmaktadır.",
                }
            )
            break

    return flags


def _analytical_findings(
    analytics: dict[str, Any],
    municipality: dict[str, Any],
    plausibility: list[dict[str, str]],
    emission_unit: str,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    concentration = analytics.get("concentration") or {}
    top3 = concentration.get("top3_share")
    if top3 is not None:
        classification = concentration.get("classification")
        if classification == "concentrated":
            detail_en = f"The top three districts account for {top3 * 100:.0f}% of reported emissions, so pressure is concentrated in a narrow core."
            detail_tr = f"İlk üç ilçe raporlanan emisyonların %{top3 * 100:.0f} kadarını oluşturmaktadır; baskı dar bir çekirdekte yoğunlaşmıştır."
        else:
            detail_en = f"The top three districts hold only {top3 * 100:.0f}% of reported emissions, so the burden is spread across a broad district base rather than a single hotspot."
            detail_tr = f"İlk üç ilçe raporlanan emisyonların yalnızca %{top3 * 100:.0f} kadarını taşımaktadır; yük tek bir sıcak nokta yerine geniş bir ilçe tabanına yayılmıştır."
        findings.append(
            {
                "id": "concentration",
                "headline_en": "Emission burden is " + ("concentrated" if classification == "concentrated" else "broadly distributed"),
                "headline_tr": "Emisyon yükü " + ("yoğunlaşmış" if classification == "concentrated" else "geniş tabana yayılmış"),
                "detail_en": detail_en,
                "detail_tr": detail_tr,
                "severity": "high" if classification == "concentrated" else "medium",
            }
        )

    lever = analytics.get("energy_lever") or {}
    if lever.get("dominant"):
        share = lever.get("electricity_share") if lever["dominant"] == "electricity" else lever.get("gas_share")
        lever_label_en = "electricity" if lever["dominant"] == "electricity" else "natural gas"
        lever_label_tr = "elektrik" if lever["dominant"] == "electricity" else "doğalgaz"
        if share is not None:
            findings.append(
                {
                    "id": "energy_lever",
                    "headline_en": f"{lever_label_en.title()} is the primary decarbonization lever",
                    "headline_tr": f"Karbonsuzlaşmanın ana kaldıracı {lever_label_tr}",
                    "detail_en": f"{lever_label_en.title()} drives about {share * 100:.0f}% of energy-related emissions, so measures targeting it will move the municipal total the most.",
                    "detail_tr": f"Enerji kaynaklı emisyonların yaklaşık %{share * 100:.0f} kadarını {lever_label_tr} sürüklemektedir; bu kaleme yönelik önlemler belediye toplamını en çok değiştirecektir.",
                    "severity": "high",
                }
            )

    intensity = analytics.get("intensity") or {}
    if intensity.get("available") and intensity.get("top"):
        top = intensity["top"]
        spread = intensity.get("spread_ratio")
        implausible_spread = spread is not None and spread > 50
        if intensity.get("diverges_from_absolute"):
            detail_en = f"On a per-capita basis {top['district']} leads at {top['per_capita']:,.0f} {emission_unit}/person, a different picture from the absolute ranking led by {intensity.get('absolute_top')}; normalized intensity should guide fairness in prioritization."
            detail_tr = f"Kişi başına bakıldığında {top['district']} {top['per_capita']:,.0f} {emission_unit}/kişi ile öne çıkmaktadır; bu, {intensity.get('absolute_top')} öncülüğündeki mutlak sıralamadan farklı bir tablodur ve önceliklendirmede adaleti normalize yoğunluk belirlemelidir."
        elif implausible_spread:
            detail_en = f"On a per-capita basis {top['district']} sits highest at {top['per_capita']:,.0f} {emission_unit}/person, but the spread across districts is so wide that the per-capita ranking should be read cautiously rather than as a settled intensity order."
            detail_tr = f"Kişi başına bakıldığında {top['district']} {top['per_capita']:,.0f} {emission_unit}/kişi ile en üstte yer almaktadır; ancak ilçeler arası dağılım o kadar geniştir ki kişi başı sıralama kesin bir yoğunluk düzeni değil, temkinli okunması gereken bir sinyal olarak değerlendirilmelidir."
        else:
            detail_en = f"{top['district']} also leads on per-capita emissions at {top['per_capita']:,.0f} {emission_unit}/person, confirming the absolute ranking reflects genuine intensity rather than only size."
            detail_tr = f"{top['district']} kişi başına emisyonda da {top['per_capita']:,.0f} {emission_unit}/kişi ile öndedir; bu, mutlak sıralamanın yalnızca büyüklüğü değil gerçek yoğunluğu yansıttığını teyit etmektedir."
        findings.append(
            {
                "id": "intensity",
                "headline_en": "Per-capita intensity reframes the ranking",
                "headline_tr": "Kişi başı yoğunluk sıralamayı yeniden çerçeveler",
                "detail_en": detail_en,
                "detail_tr": detail_tr,
                "severity": "medium",
            }
        )

    for flag in plausibility:
        findings.append(
            {
                "id": flag.get("code"),
                "headline_en": "Data signal to verify",
                "headline_tr": "Doğrulanması gereken veri sinyali",
                "detail_en": flag.get("note_en"),
                "detail_tr": flag.get("note_tr"),
                "severity": "data_quality",
            }
        )

    return findings
