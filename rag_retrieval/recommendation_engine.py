"""Rule-based recommendation layer for sustainability reporting."""

from __future__ import annotations

from statistics import median
from typing import Any


def build_report_recommendations(
    metrics: list[dict[str, Any]],
    insights: dict[str, Any],
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    thresholds = _thresholds(metrics)
    benchmarks = _benchmark_context(metrics)
    priority_lookup = {
        item.get("district"): float(item.get("score") or 0.0)
        for item in (insights.get("priority_districts") or [])
        if item.get("district")
    }

    district_profiles = [
        _district_profile(
            metric,
            thresholds,
            benchmarks,
            priority_score=priority_lookup.get(metric.get("district"), 0.0),
        )
        for metric in metrics
        if metric.get("district")
    ]
    district_profiles.sort(
        key=lambda item: (
            _severity_rank(item["severity"]),
            float(item.get("priority_score") or 0.0),
            float(item.get("total_emission") or 0.0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )
    priority_order = _priority_order(insights, district_profiles)
    commentary_profiles = _select_commentary_profiles(district_profiles, priority_order)

    municipality_focus = _municipality_focus(metrics, district_profiles, insights, commentary_profiles, benchmarks)
    strategic_recommendations = _strategic_recommendations(metrics, district_profiles, warnings, benchmarks)
    data_quality_notes = _data_quality_notes(metrics, warnings)

    return {
        "municipality_focus": municipality_focus,
        "district_archetypes": district_profiles,
        "priority_district_commentary": commentary_profiles,
        "strategic_recommendations": strategic_recommendations,
        "data_quality_notes": data_quality_notes,
    }


def _thresholds(metrics: list[dict[str, Any]]) -> dict[str, float]:
    emission_values = [float(item.get("total_emission") or 0.0) for item in metrics if float(item.get("total_emission") or 0.0) > 0.0]
    water_values = [float(item.get("water_consumption") or 0.0) for item in metrics if float(item.get("water_consumption") or 0.0) > 0.0]
    growth_values = [
        float(item.get("growth") or 0.0)
        for item in metrics
        if item.get("growth") is not None and float(item.get("growth") or 0.0) > 0.0
    ]
    water_growth_values = [
        float(item.get("water_growth") or 0.0)
        for item in metrics
        if item.get("water_growth") is not None and float(item.get("water_growth") or 0.0) > 0.0
    ]
    return {
        "emission_high": float(median(emission_values)) if emission_values else 0.0,
        "water_high": float(median(water_values)) if water_values else 0.0,
        "growth_positive": float(median(growth_values)) if growth_values else 0.0,
        "water_growth_positive": float(median(water_growth_values)) if water_growth_values else 0.0,
    }


def _benchmark_context(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    emission_rank = _rank_map(metrics, key=lambda item: float(item.get("total_emission") or 0.0))
    growth_rank = _rank_map(
        [item for item in metrics if item.get("growth") is not None],
        key=lambda item: float(item.get("growth") or 0.0),
    )
    tree_values = []
    tree_rank_input = []
    direct_shares = []
    for item in metrics:
        tree_value = _metric_value(item, "tree_count")
        if tree_value > 0.0:
            tree_values.append(tree_value)
            tree_rank_input.append({"district": item.get("district"), "value": tree_value})
        total = float(item.get("total_emission") or 0.0)
        direct = float(item.get("direct_emissions") or 0.0)
        if total > 0.0 and direct > 0.0:
            direct_shares.append(direct / total)
    tree_rank = _rank_map(tree_rank_input, key=lambda item: float(item.get("value") or 0.0))
    return {
        "district_count": len(metrics),
        "emission_rank": emission_rank,
        "growth_rank": growth_rank,
        "tree_rank": tree_rank,
        "tree_median": float(median(tree_values)) if tree_values else 0.0,
        "tree_high": _upper_half_threshold(tree_values),
        "direct_share_median": float(median(direct_shares)) if direct_shares else 0.0,
    }


def _district_profile(
    metric: dict[str, Any],
    thresholds: dict[str, float],
    benchmarks: dict[str, Any],
    *,
    priority_score: float,
) -> dict[str, Any]:
    district = str(metric.get("district") or "")
    total_emission = float(metric.get("total_emission") or 0.0)
    water = float(metric.get("water_consumption") or 0.0)
    growth = _float_or_none(metric.get("growth"))
    water_growth = _float_or_none(metric.get("water_growth"))
    electricity_emission = float(metric.get("electricity_emission") or 0.0)
    gas_emission = float(metric.get("gas_emission") or 0.0)
    direct_emissions = float(metric.get("direct_emissions") or 0.0)
    warnings = list(metric.get("warnings") or [])
    resource_metrics = _section_metrics(metric, "Resource Overview")
    context_metrics = _section_metrics(metric, "District Context and Sustainability Signals")
    energy_total = electricity_emission + gas_emission
    electricity_share = (electricity_emission / energy_total) if energy_total > 0.0 else None
    gas_share = (gas_emission / energy_total) if energy_total > 0.0 else None
    direct_share = (direct_emissions / total_emission) if total_emission > 0.0 else 0.0
    tree_value = _metric_value(metric, "tree_count")
    emission_rank = int((benchmarks.get("emission_rank") or {}).get(district) or 0)
    growth_rank = int((benchmarks.get("growth_rank") or {}).get(district) or 0)
    tree_rank = int((benchmarks.get("tree_rank") or {}).get(district) or 0)
    tree_median = float(benchmarks.get("tree_median") or 0.0)
    district_count = int(benchmarks.get("district_count") or 0)

    high_emission = total_emission > 0.0 and total_emission >= thresholds["emission_high"] > 0.0
    high_water = water > 0.0 and water >= thresholds["water_high"] > 0.0
    rising_emission = growth is not None and growth > max(thresholds["growth_positive"], 0.0)
    rising_water = water_growth is not None and water_growth > max(thresholds["water_growth_positive"], 0.0)
    improving = (growth is not None and growth < 0.0) or (water_growth is not None and water_growth < 0.0)
    top_emission_core = emission_rank and emission_rank <= min(3, max(district_count, 1))
    rapid_growth_frontier = growth_rank and growth_rank <= max(3, district_count // 8 or 1) and emission_rank > max(8, district_count // 4)
    ecological_context = tree_rank and tree_rank <= max(4, district_count // 10 or 1) and 4 <= emission_rank <= max(12, district_count // 2)
    lower_pressure = emission_rank and emission_rank >= max(district_count - 2, 1)
    transition_watch = (
        growth is not None
        and growth > max(thresholds["growth_positive"], 0.0)
        and emission_rank >= max(8, district_count // 4)
        and emission_rank <= max(24, (district_count * 3) // 4)
    )

    if warnings and total_emission == 0.0 and water == 0.0 and not resource_metrics and not context_metrics:
        archetype_key = "data_gap_watchlist"
    elif high_emission and high_water:
        archetype_key = "multi_pressure_hotspot"
    elif top_emission_core:
        archetype_key = "carbon_pressure_core"
    elif rapid_growth_frontier:
        archetype_key = "rapid_growth_frontier"
    elif high_emission and direct_share >= max(float(benchmarks.get("direct_share_median") or 0.0) * 1.2, 0.05):
        archetype_key = "direct_emission_watch"
    elif ecological_context:
        archetype_key = "ecological_context_district"
    elif transition_watch:
        archetype_key = "transition_watchlist"
    elif high_water and (rising_water or total_emission == 0.0):
        archetype_key = "water_pressure_hotspot"
    elif improving and (total_emission > 0.0 or water > 0.0):
        archetype_key = "efficiency_transition"
    elif lower_pressure:
        archetype_key = "lower_pressure_baseline"
    elif context_metrics:
        archetype_key = "ecology_signal_district"
    else:
        archetype_key = "baseline_monitor"

    archetype = _archetype_definition(archetype_key)
    score = _score_profile(
        total_emission=total_emission,
        water=water,
        growth=growth,
        water_growth=water_growth,
        context_metric_count=len(context_metrics),
        resource_metric_count=len(resource_metrics),
        warning_count=len(warnings),
        archetype_key=archetype_key,
    )
    severity = _severity_for(archetype_key, score)
    signals = _signals_for(metric, resource_metrics=resource_metrics, context_metrics=context_metrics)
    profile_context = {
        "district_count": int(benchmarks.get("district_count") or 0),
        "emission_rank": emission_rank,
        "growth_rank": growth_rank,
        "tree_rank": tree_rank,
        "tree_value": tree_value,
        "tree_median": tree_median,
        "electricity_share": electricity_share,
        "gas_share": gas_share,
        "direct_share": direct_share,
        "electricity_emission": electricity_emission,
        "gas_emission": gas_emission,
        "direct_emissions": direct_emissions,
    }
    commentary_angle = _commentary_angle(archetype_key)

    return {
        "district": district,
        "archetype_key": archetype_key,
        "archetype_label_en": archetype["label_en"],
        "archetype_label_tr": archetype["label_tr"],
        "severity": severity,
        "priority_score": round(priority_score, 3),
        "score": round(score, 3),
        "total_emission": total_emission,
        "emission_rank": emission_rank,
        "growth_rank": growth_rank,
        "tree_rank": tree_rank,
        "commentary_angle": commentary_angle,
        "headline_en": _headline_en(district, archetype_key, total_emission, water, growth, water_growth, context_metrics, profile_context),
        "headline_tr": _headline_tr(district, archetype_key, total_emission, water, growth, water_growth, context_metrics, profile_context),
        "summary_en": _summary_en(district, archetype_key, total_emission, water, growth, water_growth, resource_metrics, context_metrics, warnings, profile_context),
        "summary_tr": _summary_tr(district, archetype_key, total_emission, water, growth, water_growth, resource_metrics, context_metrics, warnings, profile_context),
        "recommended_actions_en": _actions_en(archetype_key, context_metrics, profile_context),
        "recommended_actions_tr": _actions_tr(archetype_key, context_metrics, profile_context),
        "signals": signals,
        "warnings": warnings,
        "resource_metrics": resource_metrics,
        "context_metrics": context_metrics,
        "profile_context": profile_context,
    }


def _municipality_focus(
    metrics: list[dict[str, Any]],
    district_profiles: list[dict[str, Any]],
    insights: dict[str, Any],
    commentary_profiles: list[dict[str, Any]],
    benchmarks: dict[str, Any],
) -> dict[str, list[str]]:
    highest_emission = list(insights.get("highest_emission_districts") or [])
    profile_lookup = {item["district"]: item for item in district_profiles if item.get("district")}
    fastest_growth = sorted(
        [item for item in district_profiles if item.get("growth_rank")],
        key=lambda item: int(item.get("growth_rank") or 999),
    )[:3]
    ecological = [item for item in district_profiles if item.get("archetype_key") == "ecological_context_district"][:3]
    priorities = [item.get("district") for item in (insights.get("priority_districts") or [])[:3] if item.get("district")]
    if not priorities:
        priorities = [item["district"] for item in commentary_profiles[:3] if item.get("district")]

    lines_en = []
    lines_tr = []
    total_emissions = sum(float(item.get("total_emission") or 0.0) for item in district_profiles)
    if priorities and total_emissions > 0.0:
        joined = ", ".join(priorities)
        priority_total = sum(float(profile_lookup.get(name, {}).get("total_emission") or 0.0) for name in priorities)
        share = (priority_total / total_emissions) * 100 if total_emissions > 0.0 else 0.0
        if share >= 20.0:
            lines_en.append(
                f"{joined} together account for {share:.1f}% of reported municipal emissions, so the municipality is dealing with a concentrated pressure core."
            )
            lines_tr.append(
                f"{joined} birlikte raporlanan belediye emisyonlarının %{share:.1f} kadarını oluşturduğundan belediye daha yoğunlaşmış bir baskı çekirdeğiyle karşı karşıyadır."
            )
        else:
            lines_en.append(
                f"{joined} together account for {share:.1f}% of reported municipal emissions, which means the burden is spread across a broad district base rather than concentrated in one narrow core."
            )
            lines_tr.append(
                f"{joined} birlikte raporlanan belediye emisyonlarının yalnızca %{share:.1f} kadarını oluşturduğundan yük tek bir dar çekirdekte değil daha geniş bir ilçe tabanına yayılmış durumdadır."
            )
    if highest_emission:
        top = highest_emission[0]
        lines_en.append(f"{top['district']} sets the municipality's current emissions ceiling and is the clearest reference point for decarbonization planning.")
        lines_tr.append(f"{top['district']}, belediyenin mevcut emisyon tavanını belirlediği için karbonsuzlaşma planlamasının en net referans noktasıdır.")
    if fastest_growth:
        joined = ", ".join(item["district"] for item in fastest_growth)
        growth_overlap = {item["district"] for item in fastest_growth} & set(priorities)
        if growth_overlap:
            lines_en.append(
                f"Growth pressure is also visible inside the current priority set, led by {joined}, which means the municipality is facing both present burden and continued acceleration in the same districts."
            )
            lines_tr.append(
                f"Büyüme baskısı mevcut öncelik setinin içinde de {joined} ile görünmektedir; bu durum belediyenin aynı ilçelerde hem bugünkü yükü hem de süren hızlanmayı birlikte yönettiğini göstermektedir."
            )
        else:
            lines_en.append(
                f"The fastest growth is surfacing in a different district group, led by {joined}, suggesting that future pressure may shift beyond today's top emitters."
            )
            lines_tr.append(
                f"En hızlı büyüme {joined} ile farklı bir ilçe grubunda ortaya çıkmaktadır; bu da gelecekteki baskının yalnızca bugünün en yüksek emisyonlu ilçeleriyle sınırlı kalmayabileceğini göstermektedir."
            )
    if ecological:
        joined = ", ".join(item["district"] for item in ecological)
        lines_en.append(
            f"Districts such as {joined} add stronger ecological context to the emissions picture, so intervention packages should combine carbon action with place-based resilience and public-space planning."
        )
        lines_tr.append(
            f"{joined} gibi ilçeler emisyon görünümüne daha güçlü bir ekolojik bağlam kattığından, müdahale paketleri karbon eylemini mekansal dayanıklılık ve kamusal alan planlamasıyla birlikte ele almalıdır."
        )
    lower_pressure = [item for item in district_profiles if item.get("archetype_key") == "lower_pressure_baseline"][:3]
    if lower_pressure:
        joined = ", ".join(item["district"] for item in lower_pressure)
        lines_en.append(
            f"{joined} sit closer to the lower-pressure edge of the portfolio and can be used as reference districts when calibrating what better operational performance looks like."
        )
        lines_tr.append(
            f"{joined}, portföyün daha düşük baskılı kenarında yer aldığı için daha iyi operasyonel performansın nasıl göründüğünü kalibre ederken referans ilçe olarak kullanılabilir."
        )
    if not lines_en:
        lines_en.append("The current dataset supports a baseline sustainability assessment, but stronger prioritization will depend on richer district coverage.")
        lines_tr.append("Mevcut veri seti temel bir sürdürülebilirlik değerlendirmesi sağlıyor; ancak daha güçlü önceliklendirme için ilçe kapsayıcılığının artması gerekiyor.")
    return {"en": lines_en[:4], "tr": lines_tr[:4]}


def _strategic_recommendations(
    metrics: list[dict[str, Any]],
    district_profiles: list[dict[str, Any]],
    warnings: list[str],
    benchmarks: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    emission_districts = [item["district"] for item in district_profiles if item["archetype_key"] == "carbon_pressure_core"][:5]
    growth_districts = [item["district"] for item in district_profiles if item["archetype_key"] == "rapid_growth_frontier"][:5]
    transition_districts = [item["district"] for item in district_profiles if item["archetype_key"] == "transition_watchlist"][:5]
    ecology_districts = [item["district"] for item in district_profiles if item["archetype_key"] in {"ecological_context_district", "ecology_signal_district"}][:5]
    data_gap_districts = [item["district"] for item in district_profiles if item["archetype_key"] == "data_gap_watchlist"][:5]

    if emission_districts:
        joined = ", ".join(emission_districts)
        actions.append(
            {
                "priority": "high",
                "title_en": "Stabilize the current emissions core",
                "title_tr": "Mevcut emisyon çekirdeğini stabilize et",
                "rationale_en": f"{joined} already define the municipality's current carbon ceiling, so near-term operational and retrofit measures should start there.",
                "rationale_tr": f"{joined} belediyenin mevcut karbon tavanını belirlediğinden, yakın dönem operasyonel ve retrofit önlemleri ilk olarak burada başlamalıdır.",
                "districts": emission_districts,
            }
        )
    if growth_districts:
        joined = ", ".join(growth_districts)
        actions.append(
            {
                "priority": "high" if not actions else "medium",
                "title_en": "Monitor the fast-growth frontier before it becomes the next emissions core",
                "title_tr": "Hızlı büyüme hattını yeni emisyon çekirdeğine dönüşmeden izle",
                "rationale_en": f"{joined} are not today's heaviest emitters, but their growth trajectory suggests where future pressure may accumulate first.",
                "rationale_tr": f"{joined} bugün en yüksek emisyonlu ilçe grubu değildir; ancak büyüme eğrileri gelecekte baskının ilk birikeceği alanları işaret etmektedir.",
                "districts": growth_districts,
            }
        )
    if transition_districts:
        joined = ", ".join(transition_districts)
        actions.append(
            {
                "priority": "medium",
                "title_en": "Build a transition watchlist for mid-tier districts",
                "title_tr": "Orta bant ilçeler için geçiş izleme listesi oluştur",
                "rationale_en": f"{joined} sit between the highest-pressure core and the lowest-pressure edge, so they are the best place to test scalable district programs.",
                "rationale_tr": f"{joined}, en yüksek baskı çekirdeği ile en düşük baskı çevresi arasında yer aldığı için ölçeklenebilir ilçe programlarını denemek açısından en uygun grubu oluşturmaktadır.",
                "districts": transition_districts,
            }
        )
    if ecology_districts:
        joined = ", ".join(ecology_districts)
        actions.append(
            {
                "priority": "medium",
                "title_en": "Use ecological context as a targeting layer, not a substitute for decarbonization",
                "title_tr": "Ekolojik bağlamı, karbonsuzlaşmanın yerine değil onu hedeflemeyi güçlendiren bir katman olarak kullan",
                "rationale_en": f"{joined} show stronger contextual sustainability signals, which should refine district packages rather than replace emissions action.",
                "rationale_tr": f"{joined} daha güçlü bağlamsal sürdürülebilirlik sinyalleri göstermektedir; bu göstergeler emisyon aksiyonunun yerine geçmeden ilçe paketlerini daha hassas hale getirmelidir.",
                "districts": ecology_districts,
            }
        )
    if data_gap_districts or warnings or not any(float(item.get("water_consumption") or 0.0) > 0.0 for item in metrics):
        joined = ", ".join(data_gap_districts) if data_gap_districts else "the municipality"
        actions.append(
            {
                "priority": "medium",
                "title_en": "Close the water and coverage gap before the next cycle",
                "title_tr": "Bir sonraki döngü öncesinde su ve kapsama boşluğunu kapat",
                "rationale_en": f"Coverage gaps remain visible across {joined}, and the absence of water data limits the municipality's ability to produce a fuller sustainability picture.",
                "rationale_tr": f"{joined} genelinde kapsama boşlukları sürmektedir ve su verisinin yokluğu belediyenin daha bütünlüklü bir sürdürülebilirlik resmi üretmesini sınırlandırmaktadır.",
                "districts": data_gap_districts,
            }
        )
    if not actions and metrics:
        actions.append(
            {
                "priority": "medium",
                "title_en": "Maintain a regular municipal monitoring cycle",
                "title_tr": "Düzenli belediye izleme döngüsünü sürdür",
                "rationale_en": "The current dataset supports baseline reporting; repeat measurement will make district comparisons materially stronger.",
                "rationale_tr": "Mevcut veri taban düzeyi raporlamayı destekliyor; düzenli tekrar ölçümler ilçe karşılaştırmalarını anlamlı biçimde güçlendirir.",
                "districts": [],
            }
        )
    return actions[:5]


def _data_quality_notes(metrics: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    if not metrics:
        notes.append(
            {
                "note_en": "No structured district dataset was available, so findings remain limited to narrative interpretation.",
                "note_tr": "Yapılandırılmış ilçe veri seti bulunmadığı için bulgular anlatısal yorumla sınırlı kalmaktadır.",
            }
        )
    joined = " ".join(warnings)
    if "electricity_consumption_not_found" in joined:
        notes.append(
            {
                "note_en": "Some districts lack electricity coverage, so energy comparisons should be read as partial.",
                "note_tr": "Bazı ilçelerde elektrik verisi bulunmadığından enerji karşılaştırmaları kısmi olarak değerlendirilmelidir.",
            }
        )
    if "natural_gas_consumption_not_found" in joined:
        notes.append(
            {
                "note_en": "Natural gas coverage is incomplete in parts of the dataset, which weakens emissions comparability.",
                "note_tr": "Veri setinin bazı bölümlerinde doğalgaz kapsamı eksik olduğu için emisyon karşılaştırılabilirliği zayıflamaktadır.",
            }
        )
    if "water_consumption_not_found" in joined:
        notes.append(
            {
                "note_en": "Water reporting is incomplete for some districts, so demand patterns should be treated as directional.",
                "note_tr": "Bazı ilçelerde su raporlaması eksik olduğu için talep örüntüleri yön gösterici olarak ele alınmalıdır.",
            }
        )
    if "custom_formula_missing_variable" in joined:
        notes.append(
            {
                "note_en": "A custom methodology was supplied but remained incomplete, so standard calculation logic stayed active for at least part of the report.",
                "note_tr": "Özel bir metodoloji sağlandı ancak tamamlanmadığı için raporun en az bir bölümünde standart hesap mantığı kullanılmaya devam edildi.",
            }
        )
    return notes[:5]


def _archetype_definition(key: str) -> dict[str, str]:
    mapping = {
        "multi_pressure_hotspot": {"label_en": "Multi-pressure hotspot", "label_tr": "Çoklu baskı odağı"},
        "carbon_pressure_core": {"label_en": "Carbon pressure core", "label_tr": "Karbon baskı çekirdeği"},
        "rapid_growth_frontier": {"label_en": "Rapid-growth frontier", "label_tr": "Hızlı büyüme hattı"},
        "transition_watchlist": {"label_en": "Transition watchlist district", "label_tr": "Geçiş izleme ilçesi"},
        "ecological_context_district": {"label_en": "Ecological context district", "label_tr": "Ekolojik bağlam ilçesi"},
        "lower_pressure_baseline": {"label_en": "Lower-pressure baseline", "label_tr": "Düşük baskı taban ilçesi"},
        "direct_emission_watch": {"label_en": "Direct-emission watch district", "label_tr": "Doğrudan emisyon izleme ilçesi"},
        "emission_growth_hotspot": {"label_en": "Emission growth hotspot", "label_tr": "Emisyon artış odağı"},
        "water_pressure_hotspot": {"label_en": "Water pressure hotspot", "label_tr": "Su baskısı odağı"},
        "efficiency_transition": {"label_en": "Efficiency transition district", "label_tr": "Verimlilik geçiş ilçesi"},
        "ecology_signal_district": {"label_en": "Context-rich sustainability district", "label_tr": "Bağlamı güçlü sürdürülebilirlik ilçesi"},
        "data_gap_watchlist": {"label_en": "Data gap watchlist", "label_tr": "Veri boşluğu izleme ilçesi"},
        "baseline_monitor": {"label_en": "Baseline monitoring district", "label_tr": "Temel izleme ilçesi"},
    }
    return mapping[key]


def _score_profile(
    *,
    total_emission: float,
    water: float,
    growth: float | None,
    water_growth: float | None,
    context_metric_count: int,
    resource_metric_count: int,
    warning_count: int,
    archetype_key: str,
) -> float:
    score = 0.0
    if total_emission > 0.0:
        score += 2.0
    if water > 0.0:
        score += 1.5
    if growth is not None and growth > 0.0:
        score += min(growth * 5.0, 1.5)
    if water_growth is not None and water_growth > 0.0:
        score += min(water_growth * 4.0, 1.0)
    score += min(context_metric_count * 0.3, 0.9)
    score += min(resource_metric_count * 0.25, 0.75)
    score += min(warning_count * 0.15, 0.45)
    if archetype_key == "multi_pressure_hotspot":
        score += 1.5
    elif archetype_key == "carbon_pressure_core":
        score += 1.1
    elif archetype_key == "rapid_growth_frontier":
        score += 0.9
    elif archetype_key == "transition_watchlist":
        score += 0.5
    elif archetype_key == "ecological_context_district":
        score += 0.4
    elif archetype_key in {"emission_growth_hotspot", "water_pressure_hotspot"}:
        score += 0.8
    elif archetype_key == "data_gap_watchlist":
        score += 0.4
    return score


def _severity_for(archetype_key: str, score: float) -> str:
    if archetype_key in {"multi_pressure_hotspot", "carbon_pressure_core", "rapid_growth_frontier", "emission_growth_hotspot"} or score >= 4.5:
        return "high"
    if archetype_key in {"water_pressure_hotspot", "data_gap_watchlist", "direct_emission_watch", "transition_watchlist", "ecological_context_district"} or score >= 2.5:
        return "medium"
    return "low"


def _signals_for(
    metric: dict[str, Any],
    *,
    resource_metrics: list[dict[str, Any]],
    context_metrics: list[dict[str, Any]],
) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    total_emission = float(metric.get("total_emission") or 0.0)
    water = float(metric.get("water_consumption") or 0.0)
    growth = _float_or_none(metric.get("growth"))
    water_growth = _float_or_none(metric.get("water_growth"))

    if total_emission > 0.0:
        signals.append({"label_en": "Total emissions", "label_tr": "Toplam emisyon", "value": f"{total_emission:,.2f}"})
    if growth is not None:
        signals.append({"label_en": "Emissions trend", "label_tr": "Emisyon eğilimi", "value": _percent_text(growth)})
    if water > 0.0:
        signals.append({"label_en": "Water consumption", "label_tr": "Su tüketimi", "value": f"{water:,.2f} m3"})
    if water_growth is not None:
        signals.append({"label_en": "Water trend", "label_tr": "Su eğilimi", "value": _percent_text(water_growth)})

    for summary in [*resource_metrics[:2], *context_metrics[:2]]:
        unit = f" {summary['unit']}" if summary.get("unit") else ""
        signals.append(
            {
                "label_en": summary["label"],
                "label_tr": summary["label"],
                "value": f"{float(summary.get('value') or 0.0):,.2f}{unit}",
            }
        )
    return signals[:6]


def _section_metrics(metric: dict[str, Any], section: str) -> list[dict[str, Any]]:
    items = []
    for metric_key, summary in sorted((metric.get("metric_summaries") or {}).items()):
        if metric_key in {"electricity", "natural_gas", "water"}:
            continue
        if summary.get("report_section") != section:
            continue
        value = float(summary.get("value") or 0.0)
        if value <= 0.0:
            continue
        items.append(
            {
                "metric_key": metric_key,
                "label": str(summary.get("label") or metric_key.replace("_", " ").title()),
                "value": value,
                "unit": str(summary.get("unit") or ""),
                "category": str(summary.get("category") or ""),
                "role": str(summary.get("role") or ""),
            }
        )
    return items


def _headline_en(
    district: str,
    archetype_key: str,
    total_emission: float,
    water: float,
    growth: float | None,
    water_growth: float | None,
    context_metrics: list[dict[str, Any]],
    profile_context: dict[str, Any],
) -> str:
    rank = _rank_text_en(profile_context.get("emission_rank"), profile_context.get("district_count"))
    if archetype_key == "multi_pressure_hotspot":
        return f"{district} ranks {rank} on total emissions while also carrying water pressure, making it a cross-cutting operational priority."
    if archetype_key == "carbon_pressure_core":
        return f"{district} sits in the municipality's top emissions tier and should remain one of the anchor districts for near-term decarbonization action."
    if archetype_key == "rapid_growth_frontier":
        growth_text = _percent_text(growth) if growth is not None else "an upward trend"
        return f"{district} is not the largest current emitter, but its growth trajectory at {growth_text} marks it as a likely next-wave pressure district."
    if archetype_key == "transition_watchlist":
        return f"{district} sits in the middle of the emissions distribution, making it a useful test case for scalable district programs before pressure intensifies further."
    if archetype_key == "ecological_context_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "ecological context"
        return f"{district} combines meaningful emissions pressure with stronger {metric_name.lower()}, making it a district where carbon action and ecological planning should be read together."
    if archetype_key == "lower_pressure_baseline":
        return f"{district} currently sits closer to the lower end of the emissions distribution, so it is better treated as a baseline control district than as an immediate hotspot."
    if archetype_key == "direct_emission_watch":
        return f"{district} stands out because direct emissions carry an unusually visible share of its total profile and deserve targeted review."
    if archetype_key == "emission_growth_hotspot":
        trend = _percent_text(growth) if growth is not None else "an upward trend"
        return f"{district} ranks {rank} on total emissions and is still rising at {trend}, so near-term decarbonization measures matter here."
    if archetype_key == "water_pressure_hotspot":
        trend = _percent_text(water_growth) if water_growth is not None else "persistent demand pressure"
        return f"{district} stands out on water demand with {trend}, pointing to immediate efficiency opportunities."
    if archetype_key == "efficiency_transition":
        return f"{district} shows signs of operational improvement and may offer practices worth replicating elsewhere."
    if archetype_key == "ecology_signal_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "context indicators"
        return f"{district} carries additional sustainability signals such as {metric_name}, which enrich district interpretation."
    if archetype_key == "data_gap_watchlist":
        return f"{district} cannot yet be interpreted confidently because the structured sustainability record is still incomplete."
    return f"{district} currently serves as a baseline monitoring district within the uploaded sustainability dataset."


def _headline_tr(
    district: str,
    archetype_key: str,
    total_emission: float,
    water: float,
    growth: float | None,
    water_growth: float | None,
    context_metrics: list[dict[str, Any]],
    profile_context: dict[str, Any],
) -> str:
    rank = _rank_text_tr(profile_context.get("emission_rank"), profile_context.get("district_count"))
    if archetype_key == "multi_pressure_hotspot":
        return f"{district}, toplam emisyon sıralamasında {rank} konumunda olup su baskısını da taşıdığı için kesişen baskıların görüldüğü öncelikli ilçedir."
    if archetype_key == "carbon_pressure_core":
        return f"{district}, belediyenin en yüksek emisyon bandında yer aldığı için yakın dönem karbonsuzlaşma müdahalelerinin ana ilçelerinden biri olmalıdır."
    if archetype_key == "rapid_growth_frontier":
        growth_text = _percent_text(growth) if growth is not None else "yukarı yönlü eğilim"
        return f"{district}, bugün en büyük yayıcılar arasında yer almasa da {growth_text} ile geleceğin baskı hattına dönüşebilecek bir ilçeyi temsil etmektedir."
    if archetype_key == "transition_watchlist":
        return f"{district}, emisyon dağılımının orta bandında yer aldığı için ölçeklenebilir ilçe programlarını denemek açısından uygun bir geçiş alanıdır."
    if archetype_key == "ecological_context_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "ekolojik bağlam"
        return f"{district}, anlamlı emisyon baskısını daha güçlü {metric_name.lower()} ile birlikte taşıdığı için karbon eylemi ile ekolojik planlama birlikte okunmalıdır."
    if archetype_key == "lower_pressure_baseline":
        return f"{district}, mevcut durumda emisyon dağılımının alt ucuna daha yakın olduğundan acil sıcak nokta değil, karşılaştırma için taban ilçe gibi ele alınmalıdır."
    if archetype_key == "direct_emission_watch":
        return f"{district}, toplam profil içinde doğrudan emisyon payının görünür biçimde öne çıkması nedeniyle hedefli bir kaynak incelemesi gerektirmektedir."
    if archetype_key == "emission_growth_hotspot":
        trend = _percent_text(growth) if growth is not None else "yukarı yönlü eğilim"
        return f"{district}, toplam emisyon sıralamasında {rank} konumunda ve {trend} ile yükselmeye devam ettiği için yakın dönem karbonsuzlaşma önlemleri burada önemlidir."
    if archetype_key == "water_pressure_hotspot":
        trend = _percent_text(water_growth) if water_growth is not None else "süregelen talep baskısı"
        return f"{district}, {trend} ile su talebinde öne çıkmakta ve hızlı verimlilik fırsatları sunmaktadır."
    if archetype_key == "efficiency_transition":
        return f"{district}, operasyonel iyileşme sinyalleri göstermekte ve başka ilçelere aktarılabilecek uygulamalar barındırabilir."
    if archetype_key == "ecology_signal_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "bağlamsal göstergeler"
        return f"{district}, {metric_name} gibi ek sürdürülebilirlik sinyalleri taşıyarak ilçe yorumunu zenginleştirmektedir."
    if archetype_key == "data_gap_watchlist":
        return f"{district} için yapılandırılmış sürdürülebilirlik kaydı eksik kaldığından güvenli yorum üretmek henüz mümkün değildir."
    return f"{district}, yüklenen sürdürülebilirlik veri seti içinde şu an için temel izleme ilçesi niteliğindedir."


def _summary_en(
    district: str,
    archetype_key: str,
    total_emission: float,
    water: float,
    growth: float | None,
    water_growth: float | None,
    resource_metrics: list[dict[str, Any]],
    context_metrics: list[dict[str, Any]],
    warnings: list[str],
    profile_context: dict[str, Any],
) -> str:
    direct_share = float(profile_context.get("direct_share") or 0.0)
    tree_value = float(profile_context.get("tree_value") or 0.0)
    tree_median = float(profile_context.get("tree_median") or 0.0)
    tree_rank = profile_context.get("tree_rank")
    growth_rank = profile_context.get("growth_rank")
    emission_rank = profile_context.get("emission_rank")
    context_labels = _metric_labels(context_metrics)
    resource_labels = _metric_labels(resource_metrics)
    tree_note = _tree_note_en(tree_value, tree_median, tree_rank)

    if archetype_key == "multi_pressure_hotspot":
        parts = [f"It combines {total_emission:,.2f} reported emissions with {water:,.2f} m3 of water demand"]
        if growth is not None:
            parts.append(f"emissions are still moving at {_percent_text(growth)}")
        if context_labels:
            parts.append(f"context signals such as {context_labels} mean operational fixes should be paired with resilience-oriented district planning")
        return "; ".join(parts) + "."

    if archetype_key == "carbon_pressure_core":
        parts = [f"It sits at rank #{int(emission_rank)} in the municipal emissions table" if emission_rank else "It sits in the municipal top emissions tier"]
        if growth is not None:
            parts.append(f"the current trend remains {_percent_text(growth)}")
        if direct_share >= 0.03:
            parts.append(f"direct emissions contribute a visible {direct_share * 100:.2f}% share")
        if context_labels:
            parts.append(f"context indicators such as {context_labels} should shape how decarbonization measures are packaged locally")
        return "; ".join(parts) + "."

    if archetype_key == "rapid_growth_frontier":
        parts = [
            f"Its emissions growth of {_percent_text(growth)} places it among the municipality's fastest-rising districts"
            if growth is not None
            else "Its recent trajectory suggests rising pressure"
        ]
        if emission_rank:
            parts.append(f"even though its current emissions rank is #{int(emission_rank)} rather than the very top tier")
        if resource_labels:
            parts.append(f"supporting resource metrics such as {resource_labels} should be monitored before this trend hardens")
        elif context_labels:
            parts.append(f"context signals such as {context_labels} can help explain why pressure is building")
        return "; ".join(parts) + "."

    if archetype_key == "transition_watchlist":
        parts = ["It sits in the mid-band of the municipal portfolio rather than at either extreme"]
        if growth is not None:
            parts.append(f"its emissions trend is already positive at {_percent_text(growth)}")
        if resource_labels:
            parts.append(f"resource signals such as {resource_labels} make it a practical pilot ground for scalable programs")
        elif context_labels:
            parts.append(f"context indicators such as {context_labels} can be used to tune district-level intervention design")
        return "; ".join(parts) + "."

    if archetype_key == "ecological_context_district":
        parts = ["It combines material emissions pressure with stronger ecological context"]
        if tree_note:
            parts.append(tree_note.rstrip("."))
        if context_labels:
            parts.append(f"the additional signals around {context_labels} mean district action should go beyond a pure carbon ranking")
        return "; ".join(parts) + "."

    if archetype_key == "lower_pressure_baseline":
        parts = ["It sits near the lower-pressure end of the current district set and is more useful as a control case than as an immediate hotspot"]
        if growth is not None and growth > 0.0:
            parts.append(f"even so, the {_percent_text(growth)} growth rate should be tracked so the district does not quietly move up the risk ladder")
        if context_labels:
            parts.append(f"context signals such as {context_labels} still matter for place-specific planning")
        return "; ".join(parts) + "."

    if archetype_key == "direct_emission_watch":
        parts = [f"Direct emissions represent {direct_share * 100:.2f}% of the total profile" if direct_share else "Direct emissions are more visible here than in peer districts"]
        if growth is not None:
            parts.append(f"the overall trend is {_percent_text(growth)}")
        parts.append("so source-specific operational review matters as much as generic energy efficiency")
        return "; ".join(parts) + "."

    if archetype_key == "water_pressure_hotspot":
        parts = [f"Water demand reaches {water:,.2f} m3"]
        if water_growth is not None:
            parts.append(f"the water trend is {_percent_text(water_growth)}")
        if total_emission > 0.0:
            parts.append(f"this sits alongside {total_emission:,.2f} reported emissions")
        return "; ".join(parts) + "."

    if archetype_key == "efficiency_transition":
        parts = ["The profile suggests an improving operational direction rather than escalating pressure"]
        if growth is not None:
            parts.append(f"with emissions trend at {_percent_text(growth)}")
        if water_growth is not None:
            parts.append(f"and water trend at {_percent_text(water_growth)}")
        return "; ".join(parts) + "."

    if archetype_key == "ecology_signal_district":
        parts = [f"It adds contextual sustainability evidence through {context_labels}" if context_labels else "It adds contextual sustainability evidence beyond the core emissions fields"]
        if tree_note:
            parts.append(tree_note.rstrip("."))
        return "; ".join(parts) + "."

    if archetype_key == "data_gap_watchlist":
        return "The available record is too incomplete for confident comparative interpretation."

    parts = ["The current dataset provides a baseline monitoring view"]
    if growth_rank and int(growth_rank) <= 10 and growth is not None:
        parts.append(f"with a growth trend of {_percent_text(growth)}")
    if context_labels:
        parts.append(f"and contextual signals such as {context_labels}")
    return "; ".join(parts) + "."


def _summary_tr(
    district: str,
    archetype_key: str,
    total_emission: float,
    water: float,
    growth: float | None,
    water_growth: float | None,
    resource_metrics: list[dict[str, Any]],
    context_metrics: list[dict[str, Any]],
    warnings: list[str],
    profile_context: dict[str, Any],
) -> str:
    direct_share = float(profile_context.get("direct_share") or 0.0)
    tree_value = float(profile_context.get("tree_value") or 0.0)
    tree_median = float(profile_context.get("tree_median") or 0.0)
    tree_rank = profile_context.get("tree_rank")
    growth_rank = profile_context.get("growth_rank")
    emission_rank = profile_context.get("emission_rank")
    context_labels = _metric_labels(context_metrics)
    resource_labels = _metric_labels(resource_metrics)
    tree_note = _tree_note_tr(tree_value, tree_median, tree_rank)

    if archetype_key == "multi_pressure_hotspot":
        parts = [f"{total_emission:,.2f} düzeyindeki emisyonu {water:,.2f} m3 su talebiyle birlikte taşımaktadır"]
        if growth is not None:
            parts.append(f"emisyon eğilimi halen {_percent_text(growth)} düzeyindedir")
        if context_labels:
            parts.append(f"{context_labels} gibi bağlamsal sinyaller operasyonel önlemlerin dirençlilik odaklı planlamayla birlikte kurgulanması gerektiğini göstermektedir")
        return "; ".join(parts) + "."

    if archetype_key == "carbon_pressure_core":
        parts = [f"belediye emisyon sıralamasında {int(emission_rank)}. sıradadır" if emission_rank else "belediyenin üst emisyon bandında yer almaktadır"]
        if growth is not None:
            parts.append(f"mevcut eğilim {_percent_text(growth)} seviyesindedir")
        if direct_share >= 0.03:
            parts.append(f"doğrudan emisyonların payı görünür biçimde %{direct_share * 100:.2f} düzeyindedir")
        if context_labels:
            parts.append(f"{context_labels} gibi bağlamsal göstergeler karbonsuzlaşma paketinin ilçeye özgü tasarlanmasını gerektirmektedir")
        return "; ".join(parts) + "."

    if archetype_key == "rapid_growth_frontier":
        parts = [
            f"emisyon büyümesi {_percent_text(growth)} ile belediyenin en hızlı yükselen ilçeleri arasına girmektedir"
            if growth is not None
            else "yakın dönem eğilim yükselen bir baskıya işaret etmektedir"
        ]
        if emission_rank:
            parts.append(f"buna karşın mevcut emisyon sıralaması en üst çekirdekte değil {int(emission_rank)}. basamaktadır")
        if resource_labels:
            parts.append(f"{resource_labels} gibi destekleyici kaynak metrikleri bu baskı sertleşmeden izlenmelidir")
        elif context_labels:
            parts.append(f"{context_labels} gibi bağlamsal sinyaller baskının neden biriktiğini açıklamaya yardımcı olabilir")
        return "; ".join(parts) + "."

    if archetype_key == "transition_watchlist":
        parts = ["belediye portföyünün iki ucu arasında orta bantta yer almaktadır"]
        if growth is not None:
            parts.append(f"buna rağmen emisyon eğilimi {_percent_text(growth)} ile yukarı yönlüdür")
        if resource_labels:
            parts.append(f"{resource_labels} gibi metrikler burayı ölçeklenebilir programlar için uygun bir pilot alan haline getirmektedir")
        elif context_labels:
            parts.append(f"{context_labels} gibi bağlamsal göstergeler ilçe bazlı tasarımı keskinleştirebilir")
        return "; ".join(parts) + "."

    if archetype_key == "ecological_context_district":
        parts = ["anlamlı emisyon baskısını daha güçlü bir ekolojik bağlamla birlikte taşımaktadır"]
        if tree_note:
            parts.append(tree_note.rstrip("."))
        if context_labels:
            parts.append(f"{context_labels} gibi ek sinyaller ilçe aksiyonunun yalnızca karbon sıralamasına indirgenmemesi gerektiğini göstermektedir")
        return "; ".join(parts) + "."

    if archetype_key == "lower_pressure_baseline":
        parts = ["mevcut ilçe setinin daha düşük baskılı ucunda yer almakta ve acil sıcak noktadan çok karşılaştırma amaçlı bir kontrol örneği sunmaktadır"]
        if growth is not None and growth > 0.0:
            parts.append(f"buna rağmen {_percent_text(growth)} büyüme eğilimi sessiz bir risk birikimini dışlamamaktadır")
        if context_labels:
            parts.append(f"{context_labels} gibi bağlamsal göstergeler yine de ilçe planlaması için önemlidir")
        return "; ".join(parts) + "."

    if archetype_key == "direct_emission_watch":
        parts = [f"doğrudan emisyonların toplam içindeki payı %{direct_share * 100:.2f} düzeyindedir" if direct_share else "doğrudan emisyon bileşeni benzer ilçelere göre daha görünür durumdadır"]
        if growth is not None:
            parts.append(f"genel eğilim {_percent_text(growth)} düzeyindedir")
        parts.append("bu nedenle kaynak bazlı operasyonel inceleme, genel enerji verimliliği kadar kritik hale gelmektedir")
        return "; ".join(parts) + "."

    if archetype_key == "water_pressure_hotspot":
        parts = [f"su talebi {water:,.2f} m3 düzeyine ulaşmaktadır"]
        if water_growth is not None:
            parts.append(f"su eğilimi {_percent_text(water_growth)} seviyesindedir")
        if total_emission > 0.0:
            parts.append(f"bu görünüm {total_emission:,.2f} düzeyindeki emisyonla birlikte okunmalıdır")
        return "; ".join(parts) + "."

    if archetype_key == "efficiency_transition":
        parts = ["tırmanan baskıdan çok iyileşen bir operasyonel yön işaret etmektedir"]
        if growth is not None:
            parts.append(f"emisyon eğilimi {_percent_text(growth)} seviyesindedir")
        if water_growth is not None:
            parts.append(f"su eğilimi {_percent_text(water_growth)} seviyesindedir")
        return "; ".join(parts) + "."

    if archetype_key == "ecology_signal_district":
        parts = [f"{context_labels} üzerinden çekirdek emisyon alanlarının ötesine geçen bağlamsal sürdürülebilirlik kanıtı üretmektedir" if context_labels else "çekirdek emisyon alanlarının ötesine geçen bağlamsal sürdürülebilirlik kanıtı üretmektedir"]
        if tree_note:
            parts.append(tree_note.rstrip("."))
        return "; ".join(parts) + "."

    if archetype_key == "data_gap_watchlist":
        return "Mevcut kayıt karşılaştırmalı yorum üretmek açısından fazla eksiktir."

    parts = ["mevcut veri seti yalnızca temel bir izleme görünümü sunmaktadır"]
    if growth_rank and int(growth_rank) <= 10 and growth is not None:
        parts.append(f"büyüme eğilimi {_percent_text(growth)} düzeyindedir")
    if context_labels:
        parts.append(f"{context_labels} gibi bağlamsal sinyaller de mevcuttur")
    return "; ".join(parts) + "."


def _actions_en(archetype_key: str, context_metrics: list[dict[str, Any]], profile_context: dict[str, Any]) -> list[str]:
    if archetype_key == "multi_pressure_hotspot":
        return [
            "Combine building energy efficiency, fuel switching, and district-scale demand management.",
            "Pair emissions reduction with leakage control or water-efficiency measures.",
            "Track progress with a district action plan and periodic performance review.",
        ]
    if archetype_key == "carbon_pressure_core":
        return [
            "Start with the largest municipal and building-energy loads because this district defines the current emissions ceiling.",
            "Set a district-level reduction pathway with quarterly tracking rather than relying on one-off measures.",
            "Use any contextual sustainability indicators to target where decarbonization packages should land first.",
        ]
    if archetype_key == "rapid_growth_frontier":
        return [
            "Investigate what is driving the recent growth before the district enters the top-emissions core.",
            "Deploy early efficiency and demand-management measures while the district is still more preventable than entrenched.",
            "Track the next reporting cycle closely and escalate if growth remains structurally high.",
        ]
    if archetype_key == "transition_watchlist":
        return [
            "Use the district as a pilot for measures that should later scale across similar mid-band districts.",
            "Test a combined package of operational efficiency, monitoring, and targeted capital upgrades.",
            "Review whether the district is moving upward toward the pressure core or stabilizing in place.",
        ]
    if archetype_key == "direct_emission_watch":
        return [
            "Review the operational sources behind the direct-emissions share and verify whether they can be reduced at source.",
            "Separate direct process emissions from building-energy actions so mitigation planning stays targeted.",
            "Use site-level audits to understand whether a small number of sources are driving the district profile.",
        ]
    if archetype_key == "ecological_context_district":
        return [
            "Protect and expand ecological assets while pursuing the same decarbonization discipline applied in high-emission districts.",
            "Use public-space, tree, or resilience indicators to decide where the district package should be spatially concentrated.",
            "Treat ecological strength as a targeting advantage, not as a substitute for emissions reduction.",
        ]
    if archetype_key == "lower_pressure_baseline":
        return [
            "Preserve consistent monitoring so the district can serve as a baseline comparator for higher-pressure peers.",
            "Investigate any emerging growth signal early, before it becomes a visible ranking problem.",
            "Use the district to benchmark what stable operational performance looks like.",
        ]
    if archetype_key == "water_pressure_hotspot":
        return [
            "Launch water-efficiency and leakage-reduction measures in the highest-demand assets.",
            "Pair operational monitoring with reuse or conservation messaging where feasible.",
            "Track whether water growth moderates after the first intervention cycle.",
        ]
    if archetype_key == "efficiency_transition":
        return [
            "Stabilize and document the practices behind the improving trend.",
            "Replicate successful measures in comparable districts before conditions change.",
            "Keep monitoring so temporary gains are not mistaken for structural improvement.",
        ]
    if archetype_key == "ecology_signal_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "context indicators"
        return [
            f"Use {metric_name} to sharpen district-specific sustainability planning.",
            "Connect ecological or resilience indicators with resource and emissions decisions.",
            "Protect and expand contextual sustainability assets where they support long-term resilience.",
        ]
    if archetype_key == "data_gap_watchlist":
        return [
            "Close the missing dataset fields before using the district for comparative decisions.",
            "Standardize data collection and validate district-level source records.",
            "Re-run the report after coverage improves so this district can be assessed reliably.",
        ]
    return [
        "Maintain regular monitoring and preserve comparability across reporting periods.",
        "Use the district as a baseline for peer comparison and trend tracking.",
    ]


def _actions_tr(archetype_key: str, context_metrics: list[dict[str, Any]], profile_context: dict[str, Any]) -> list[str]:
    if archetype_key == "multi_pressure_hotspot":
        return [
            "Bina enerji verimliliği, yakıt dönüşümü ve ilçe ölçekli talep yönetimi birlikte ele alınmalıdır.",
            "Emisyon azaltımı, kaçak kontrolü veya su verimliliği önlemleriyle eşleştirilmelidir.",
            "İlerleme ilçe eylem planı ve periyodik performans değerlendirmesiyle izlenmelidir.",
        ]
    if archetype_key == "carbon_pressure_core":
        return [
            "En büyük belediye ve bina enerji yüklerinden başlanmalı; çünkü bu ilçe mevcut emisyon tavanını belirlemektedir.",
            "Tek seferlik önlemler yerine çeyreklik izleme içeren ilçe düzeyi azaltım patikası kurulmalıdır.",
            "Varsa bağlamsal sürdürülebilirlik göstergeleri, karbonsuzlaşma paketinin önce nereye ineceğini belirlemede kullanılmalıdır.",
        ]
    if archetype_key == "rapid_growth_frontier":
        return [
            "İlçe üst emisyon çekirdeğine girmeden önce son büyüme ivmesini üreten sürücüler incelenmelidir.",
            "Baskı yerleşik hale gelmeden erken verimlilik ve talep yönetimi önlemleri devreye alınmalıdır.",
            "Bir sonraki raporlama döngüsü yakından izlenmeli; büyüme yapısal kalıyorsa müdahale seviyesi yükseltilmelidir.",
        ]
    if archetype_key == "transition_watchlist":
        return [
            "İlçe, benzer orta bant ilçelere daha sonra ölçeklenebilecek programlar için pilot alan olarak kullanılmalıdır.",
            "Operasyonel verimlilik, izleme ve hedefli sermaye iyileştirmelerini birleştiren paket test edilmelidir.",
            "İlçenin baskı çekirdeğine doğru yükselip yükselmediği ya da bulunduğu yerde dengelenip dengelenmediği gözden geçirilmelidir.",
        ]
    if archetype_key == "direct_emission_watch":
        return [
            "Doğrudan emisyon payını oluşturan operasyonel kaynaklar gözden geçirilmeli ve mümkünse kaynağında azaltım hedeflenmelidir.",
            "Doğrudan proses emisyonları, bina enerji önlemlerinden ayrı ele alınarak daha hedefli bir planlama yapılmalıdır.",
            "Az sayıda noktasal kaynağın profili sürükleyip sürüklemediği saha denetimleriyle doğrulanmalıdır.",
        ]
    if archetype_key == "ecological_context_district":
        return [
            "Ekolojik varlıklar korunup güçlendirilirken yüksek emisyonlu ilçelerde uygulanan karbonsuzlaşma disiplini burada da sürdürülmelidir.",
            "Kamusal alan, ağaç veya dirençlilik göstergeleri, ilçe paketinin mekansal olarak nereye yoğunlaşacağını belirlemede kullanılmalıdır.",
            "Ekolojik güç, emisyon azaltımının yerine geçen değil onu daha doğru hedefleyen bir avantaj olarak ele alınmalıdır.",
        ]
    if archetype_key == "lower_pressure_baseline":
        return [
            "İlçenin daha yüksek baskılı akranlara referans olabilmesi için tutarlı izleme korunmalıdır.",
            "Ortaya çıkan büyüme sinyalleri, görünür bir sıralama sorununa dönüşmeden erken aşamada incelenmelidir.",
            "İlçe, dengeli operasyonel performansın nasıl göründüğünü benchmark etmek için kullanılmalıdır.",
        ]
    if archetype_key == "water_pressure_hotspot":
        return [
            "En yüksek talebe sahip varlıklarda su verimliliği ve kaçak azaltımı başlatılmalıdır.",
            "Uygun alanlarda yeniden kullanım veya tasarruf odaklı operasyonel izleme kurulmalıdır.",
            "İlk müdahale döngüsünden sonra su eğiliminin yavaşlayıp yavaşlamadığı izlenmelidir.",
        ]
    if archetype_key == "efficiency_transition":
        return [
            "İyileşen eğilimi oluşturan uygulamalar sabitlenmeli ve belgelenmelidir.",
            "Başarılı önlemler benzer ilçelere aktarılmalıdır.",
            "Geçici kazanımların yapısal iyileşme sanılmaması için izleme sürdürülmelidir.",
        ]
    if archetype_key == "ecology_signal_district":
        metric_name = context_metrics[0]["label"] if context_metrics else "bağlamsal göstergeler"
        return [
            f"{metric_name} gibi göstergeler ilçe bazlı planlamayı keskinleştirmek için kullanılmalıdır.",
            "Ekolojik veya dirençlilik göstergeleri kaynak ve emisyon kararlarıyla birlikte okunmalıdır.",
            "Uzun vadeli dayanıklılığı destekleyen bağlamsal sürdürülebilirlik varlıkları korunmalı ve güçlendirilmelidir.",
        ]
    if archetype_key == "data_gap_watchlist":
        return [
            "Karşılaştırmalı kararlar için kullanılmadan önce eksik veri alanları tamamlanmalıdır.",
            "Veri toplama standardize edilmeli ve ilçe kaynak kayıtları doğrulanmalıdır.",
            "Kapsam iyileştikten sonra rapor yeniden çalıştırılmalıdır.",
        ]
    return [
        "Düzenli izleme sürdürülmeli ve raporlama dönemleri arası karşılaştırılabilirlik korunmalıdır.",
        "İlçe, akran karşılaştırması ve trend izlemesi için temel referans olarak kullanılmalıdır.",
    ]


def _commentary_angle(archetype_key: str) -> str:
    mapping = {
        "multi_pressure_hotspot": "cross_pressure",
        "carbon_pressure_core": "emission_concentration",
        "rapid_growth_frontier": "growth_risk",
        "transition_watchlist": "mid_band_transition",
        "ecological_context_district": "ecological_targeting",
        "lower_pressure_baseline": "baseline_control",
        "direct_emission_watch": "source_profile",
        "water_pressure_hotspot": "resource_pressure",
        "efficiency_transition": "operational_improvement",
        "ecology_signal_district": "context_signal",
        "data_gap_watchlist": "coverage_gap",
        "baseline_monitor": "baseline_monitoring",
        "emission_growth_hotspot": "growth_risk",
    }
    return mapping.get(archetype_key, "baseline_monitoring")


def _select_commentary_profiles(
    district_profiles: list[dict[str, Any]],
    priority_order: dict[str, int],
) -> list[dict[str, Any]]:
    if not district_profiles:
        return []

    def ordered_candidates(archetypes: set[str] | None = None) -> list[dict[str, Any]]:
        items = [
            item
            for item in district_profiles
            if archetypes is None or str(item.get("archetype_key")) in archetypes
        ]
        return sorted(
            items,
            key=lambda item: (
                priority_order.get(str(item.get("district")), 999),
                -_severity_rank(item.get("severity")),
                -float(item.get("priority_score") or 0.0),
                -float(item.get("total_emission") or 0.0),
                -float(item.get("score") or 0.0),
            ),
        )

    selected: list[dict[str, Any]] = []
    selected_districts: set[str] = set()
    selected_angles: set[str] = set()

    buckets = [
        {"multi_pressure_hotspot", "carbon_pressure_core"},
        {"rapid_growth_frontier"},
        {"transition_watchlist"},
        {"ecological_context_district", "ecology_signal_district"},
        {"lower_pressure_baseline", "baseline_monitor"},
        {"direct_emission_watch", "water_pressure_hotspot", "efficiency_transition", "data_gap_watchlist"},
    ]

    for bucket in buckets:
        for candidate in ordered_candidates(bucket):
            district = str(candidate.get("district") or "")
            angle = str(candidate.get("commentary_angle") or "")
            if not district or district in selected_districts:
                continue
            if angle and angle in selected_angles:
                continue
            selected.append(candidate)
            selected_districts.add(district)
            if angle:
                selected_angles.add(angle)
            break

    for candidate in ordered_candidates():
        if len(selected) >= min(6, len(district_profiles)):
            break
        district = str(candidate.get("district") or "")
        angle = str(candidate.get("commentary_angle") or "")
        if not district or district in selected_districts:
            continue
        if angle and angle in selected_angles and len(selected) >= 4:
            continue
        selected.append(candidate)
        selected_districts.add(district)
        if angle:
            selected_angles.add(angle)

    return selected[:6]


def _metric_labels(items: list[dict[str, Any]], *, limit: int = 2) -> str:
    labels = [str(item.get("label") or "").strip() for item in items if str(item.get("label") or "").strip()]
    return ", ".join(labels[:limit])


def _tree_note_en(tree_value: float, tree_median: float, tree_rank: Any) -> str:
    if tree_value <= 0.0:
        return ""
    qualifier = "above" if tree_median > 0.0 and tree_value >= tree_median else "below"
    rank_fragment = f" and ranks #{int(tree_rank)} on tree count" if tree_rank and int(tree_rank) <= 10 else ""
    return f"Tree Count sits {qualifier} the municipal median{rank_fragment}."


def _tree_note_tr(tree_value: float, tree_median: float, tree_rank: Any) -> str:
    if tree_value <= 0.0:
        return ""
    qualifier = "üzerindedir" if tree_median > 0.0 and tree_value >= tree_median else "altındadır"
    rank_fragment = f" ve ağaç sayısında {int(tree_rank)}. sıradadır" if tree_rank and int(tree_rank) <= 10 else ""
    return f"Ağaç sayısı belediye medyanının {qualifier}{rank_fragment}."


def _severity_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value).strip().lower(), 0)


def _percent_text(value: float) -> str:
    return f"{value * 100:.2f}%"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(metric: dict[str, Any], metric_key: str) -> float:
    summary = (metric.get("metric_summaries") or {}).get(metric_key) or {}
    return float(summary.get("value") or 0.0)


def _priority_order(insights: dict[str, Any], district_profiles: list[dict[str, Any]]) -> dict[str, int]:
    ordered = [
        str(item.get("district"))
        for item in (insights.get("priority_districts") or [])
        if item.get("district")
    ]
    if not ordered:
        ordered = [
            str(item.get("district"))
            for item in (insights.get("highest_emission_districts") or [])
            if item.get("district")
        ]
    if not ordered:
        ordered = [str(item.get("district")) for item in district_profiles if item.get("district")]
    return {district: index for index, district in enumerate(ordered)}


def _rank_map(items: list[dict[str, Any]], key) -> dict[str, int]:
    ranked = sorted(
        [item for item in items if item.get("district")],
        key=key,
        reverse=True,
    )
    return {str(item.get("district")): index + 1 for index, item in enumerate(ranked)}


def _upper_half_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(len(ordered) // 2, 0)])


def _rank_text_en(rank: Any, total: Any) -> str:
    if not rank or not total:
        return "among the covered districts"
    return f"#{int(rank)} of {int(total)}"


def _rank_text_tr(rank: Any, total: Any) -> str:
    if not rank or not total:
        return "kapsanan ilçeler içinde"
    return f"{int(rank)}/{int(total)}"
