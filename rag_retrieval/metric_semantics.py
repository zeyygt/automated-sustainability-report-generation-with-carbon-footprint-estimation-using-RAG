"""Generic semantic profiles for discovered sustainability metrics.

The intelligence layer must not special-case any single metric (e.g. tree
count). Every metric — known or user-supplied (waste, wastewater, recycling,
renewable share, transit ridership, ...) — is interpreted here through its
*category* and *role*, never its literal name. A profile answers three
questions a municipal analyst would ask of any indicator:

- direction: is a higher value good, bad, or neutral?
- relation: how does it connect to emissions (driver, sink/offset, pressure,
  context)?
- significance: why does it matter and how should it be read?

This is the static, knowledge-based half of metric meaning. The empirical half
(does the data actually support the nominal meaning, e.g. a context metric that
merely tracks district size) is supplied separately by the analytics layer and
combined at the point of use.
"""

from __future__ import annotations

from .text import normalize_for_search, search_tokens


# Keyword nuances that override the category default. Order matters: the first
# matching group wins, so genuine assets are checked before generic pressures.
_ASSET_TOKENS = {
    "tree", "trees", "forest", "green", "canopy", "park", "biodiversity", "soil",
    "renewable", "solar", "wind", "pv", "geothermal", "hydro",
    "recycling", "recycled", "recovery", "compost", "reuse", "reused",
    "transit", "ridership", "metro", "tram", "bus", "bicycle", "pedestrian",
    "efficiency", "savings", "offset", "sink", "sequestration",
}
_PRESSURE_TOKENS = {
    "waste", "landfill", "wastewater", "sewage", "leak", "leakage", "loss",
    "losses", "pollution", "emission", "emissions", "noise", "flood", "risk",
    "congestion",
}


def metric_semantic_profile(
    *,
    category: str | None,
    role: str | None,
    metric_key: str | None = None,
    label: str | None = None,
) -> dict[str, object]:
    """Return a generic semantic descriptor for a metric.

    Driven by (category, role) with keyword nuance — never by a hard-coded
    metric name — so an arbitrary user metric is interpreted consistently.
    """
    category = (category or "other").strip().lower()
    role = (role or "context_indicator").strip().lower()
    # Keyword nuance reads the metric's own name/label only; the category is a
    # separate signal so e.g. "recycling rate" in the waste category is still
    # recognised as an asset rather than a waste pressure.
    name_tokens = _tokens(metric_key, label)

    direction, relation = _direction_and_relation(category, role, name_tokens)
    asset = direction == "higher_better" and relation in {"sink_offset", "context"}
    significance_en, significance_tr = _significance(relation, category, asset)

    return {
        "direction": direction,
        "relation": relation,
        "asset": asset,
        "significance_en": significance_en,
        "significance_tr": significance_tr,
    }


def _tokens(*values: str | None) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value:
            tokens.update(search_tokens(normalize_for_search(str(value))))
    return tokens


def _direction_and_relation(category: str, role: str, tokens: set[str]) -> tuple[str, str]:
    # Strongest signal: explicit keyword nuance overrides the category default.
    if tokens & _ASSET_TOKENS and not (tokens & {"waste", "wastewater", "loss", "leak", "leakage"}):
        return "higher_better", "sink_offset"
    if role == "offset_or_sink":
        return "higher_better", "sink_offset"
    if role == "emission_input":
        return "higher_worse", "driver"

    # Category defaults.
    if category in {"energy", "climate"}:
        return "higher_worse", "driver"
    if category in {"water", "waste"}:
        return "higher_worse", "pressure"
    if category == "mobility":
        # Fuel-type mobility is a driver; transit-type mobility is an asset.
        if tokens & {"fuel", "diesel", "gasoline", "petrol", "benzin", "motorin"}:
            return "higher_worse", "driver"
        return "neutral", "context"
    if category == "ecology":
        return "higher_better", "sink_offset"
    if category in {"resilience"} or tokens & _PRESSURE_TOKENS:
        return "neutral", "context"
    return "neutral", "context"


def _significance(relation: str, category: str, asset: bool) -> tuple[str, str]:
    if relation == "driver":
        return (
            "A direct emission driver through its emission factor, and therefore one of the primary reduction levers.",
            "Emisyon faktörü üzerinden doğrudan bir emisyon sürücüsüdür ve bu nedenle başlıca azaltım kaldıraçlarından biridir.",
        )
    if relation == "sink_offset" and category == "ecology":
        return (
            "Acts as a carbon sink and supports heat and air-quality resilience, partly offsetting local emissions.",
            "Bir karbon yutağı işlevi görür; ısı ve hava kalitesi dayanıklılığını destekleyerek yerel emisyonları kısmen dengeler.",
        )
    if relation == "sink_offset" and category == "energy":
        return (
            "On-site renewable generation displaces grid electricity, the dominant emission source, so it offsets emissions directly.",
            "Yerinde üretilen yenilenebilir enerji, baskın emisyon kaynağı olan şebeke elektriğini ikame ederek emisyonları doğrudan azaltır.",
        )
    if relation == "sink_offset" and category == "waste":
        return (
            "Recovery and recycling offset landfill burden and the methane and disposal emissions that come with it.",
            "Geri kazanım ve geri dönüşüm, depolama yükünü ve buna bağlı metan ve bertaraf emisyonlarını dengeler.",
        )
    if relation == "sink_offset":
        return (
            "Acts as an offsetting asset that reduces net emissions or substitutes a higher-emission alternative.",
            "Net emisyonu azaltan veya daha yüksek emisyonlu bir alternatifi ikame eden, dengeleyici bir varlık olarak işlev görür.",
        )
    if relation == "pressure" and category in {"water"}:
        return (
            "Water and wastewater volumes drive pumping and treatment energy, an indirect emission and an infrastructure pressure.",
            "Su ve atıksu hacimleri pompalama ve arıtma enerjisini artırır; bu hem dolaylı bir emisyon hem de altyapı baskısıdır.",
        )
    if relation == "pressure":
        return (
            "Reflects consumption or disposal pressure that compounds the emissions picture and signals where demand management is needed.",
            "Emisyon tablosunu ağırlaştıran ve talep yönetiminin nerede gerektiğini işaret eden bir tüketim veya bertaraf baskısını yansıtır.",
        )
    return (
        "A contextual indicator describing local conditions and exposure rather than a direct emission source.",
        "Doğrudan bir emisyon kaynağından çok yerel koşulları ve maruziyeti tanımlayan bağlamsal bir göstergedir.",
    )
