"""Render generated reports to HTML and PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

from .report_metrics import public_metrics
from .report_models import GeneratedReport, ReportAssets


def render_report(
    report: GeneratedReport,
    output_dir: str | Path,
    assets: ReportAssets | None = None,
    filename_stem: str = "sustainability_report",
) -> GeneratedReport:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    assets = assets or discover_report_assets()

    html_path = output_path / f"{filename_stem}.html"
    pdf_path = output_path / f"{filename_stem}.pdf"

    html_path.write_text(render_html(report, assets), encoding="utf-8")
    render_pdf(report, pdf_path, assets)

    return GeneratedReport(
        title=report.title,
        language=report.language,
        generated_at=report.generated_at,
        report_input=report.report_input,
        ai_content_markdown=report.ai_content_markdown,
        charts=report.charts,
        html_path=html_path,
        pdf_path=pdf_path,
        warnings=report.warnings,
    )


def discover_report_assets(base_dir: str | Path = ".") -> ReportAssets:
    root = Path(base_dir)
    asset_dirs = (root / "assets",)
    return ReportAssets(
        ibb_logo_path=_first_existing(asset_dirs, "ibb_logo.png"),
        cycle_logo_path=_first_existing(asset_dirs, "cycle_logo.png"),
    )


def render_html(report: GeneratedReport, assets: ReportAssets) -> str:
    chart_html = "\n".join(
        f"<section><h2>{html.escape(chart['title'])}</h2><img class=\"chart\" src=\"{html.escape(_image_src(chart['path']))}\" /></section>"
        for chart in report.charts
    )
    metric_rows = "\n".join(_metric_rows(report.report_input.structured_results))
    ibb_logo = _html_logo(assets.ibb_logo_path, "IBB logo")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(report.title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 42px; color: #17212b; line-height: 1.5; }}
    .cover {{ border-bottom: 3px solid #22577a; padding-bottom: 28px; margin-bottom: 32px; }}
    .cover-head {{ display: flex; align-items: center; gap: 26px; min-height: 96px; }}
    .logo {{ max-height: 92px; max-width: 210px; object-fit: contain; }}
    .cover-copy {{ max-width: 780px; }}
    h1 {{ font-size: 34px; margin: 0 0 8px; color: #102a43; }}
    .subtitle {{ font-size: 15px; color: #52616b; margin: 0; }}
    h2 {{ color: #22577a; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; margin-top: 32px; }}
    h3 {{ color: #334e68; margin-top: 22px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 22px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef5f9; }}
    .chart {{ width: 100%; max-width: 920px; border: 1px solid #d9e2ec; }}
    .warning {{ background: #fff7e6; border-left: 4px solid #d97706; padding: 12px 16px; }}
    .muted {{ color: #52616b; }}
  </style>
</head>
<body>
  <section class="cover">
    <div class="cover-head">
      {ibb_logo}
      <div class="cover-copy">
        <h1>{html.escape(report.title)}</h1>
        <p class="subtitle">District-level emissions outlook, trend assessment, and priority sustainability actions.</p>
      </div>
    </div>
  </section>

  <section>
    {_markdown_to_html(report.ai_content_markdown)}
  </section>

  {chart_html}

  <section>
    <h2>District-Level Emissions Indicators</h2>
    <table>
      <tr><th>District</th><th>Total emissions</th><th>Natural gas</th><th>Electricity</th><th>Direct</th><th>Growth</th></tr>
      {metric_rows}
    </table>
  </section>
</body>
</html>
"""


def render_pdf(report: GeneratedReport, output_path: str | Path, assets: ReportAssets) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_name = _register_font()
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = font_name
    styles["Title"].fontName = font_name
    styles["Heading1"].fontName = font_name
    styles["Heading2"].fontName = font_name
    styles["Heading3"].fontName = font_name
    small = ParagraphStyle("Small", parent=styles["Normal"], fontName=font_name, fontSize=8, leading=10)

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=1.4 * cm, leftMargin=1.4 * cm, topMargin=1.4 * cm, bottomMargin=1.2 * cm)
    story = []

    logo = Image(str(assets.ibb_logo_path), width=4.2 * cm, height=2.2 * cm, kind="proportional") if assets.ibb_logo_path and assets.ibb_logo_path.exists() else Paragraph("", styles["Normal"])
    title_block = [
        Paragraph(report.title, styles["Title"]),
        Paragraph("District-level emissions outlook, trend assessment, and priority sustainability actions.", styles["Normal"]),
    ]
    story.append(Table([[logo, title_block]], colWidths=[4.8 * cm, 11.8 * cm]))
    story.append(Spacer(1, 0.6 * cm))

    _append_markdown(story, report.ai_content_markdown, styles)

    if report.charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", styles["Heading1"]))
        for chart in report.charts:
            story.append(Paragraph(chart["title"], styles["Heading2"]))
            story.append(Image(chart["path"], width=16.5 * cm, height=8.8 * cm, kind="proportional"))
            story.append(Spacer(1, 0.3 * cm))

    story.append(PageBreak())
    story.append(Paragraph("District-Level Emissions Indicators", styles["Heading1"]))
    metric_table = _metric_table(report.report_input.structured_results, styles, small)
    if metric_table:
        story.append(metric_table)

    doc.build(story)
    return Path(output_path)


def _metric_rows(structured_results: Iterable[dict]) -> list[str]:
    rows = []
    for data in public_metrics(list(structured_results)):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(data.get('district', '')))}</td>"
            f"<td>{_fmt_number(data.get('total_emission'))}</td>"
            f"<td>{_fmt_number(data.get('gas_emission'))}</td>"
            f"<td>{_fmt_number(data.get('electricity_emission'))}</td>"
            f"<td>{_fmt_number(data.get('direct_emissions'))}</td>"
            f"<td>{_fmt_growth(data.get('growth'))}</td>"
            "</tr>"
        )
    return rows


def _metric_table(structured_results: list[dict], styles, small):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    rows = [["District", "Total", "Natural gas", "Electricity", "Direct", "Growth"]]
    for data in public_metrics(structured_results)[:22]:
        rows.append(
            [
                Paragraph(str(data.get("district", "")), small),
                _fmt_number(data.get("total_emission")),
                _fmt_number(data.get("gas_emission")),
                _fmt_number(data.get("electricity_emission")),
                _fmt_number(data.get("direct_emissions")),
                _fmt_growth(data.get("growth")),
            ]
        )
    if len(rows) == 1:
        return None
    table = Table(rows, repeatRows=1)
    table.setStyle(_table_style(colors))
    return table


def _documents_table(documents: list[dict], styles, small):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table

    rows = [["Filename", "Parser", "Table rows", "Fact rows", "Engine"]]
    for item in documents:
        rows.append(
            [
                Paragraph(str(item.get("filename", "")), small),
                Paragraph(str(item.get("parser", "")), small),
                str(item.get("table_rows", 0)),
                str(item.get("fact_rows", 0)),
                "yes" if item.get("has_data_engine") else "no",
            ]
        )
    table = Table(rows, repeatRows=1)
    table.setStyle(_table_style(colors))
    return table


def _cover_stats_table(report: GeneratedReport, styles, small):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    stats = _cover_stats(report)
    rows = [
        ["Kapsam", "İlçe", "Emisyon göstergesi", "Grafik"],
        [str(stats["documents"]), str(stats["districts"]), str(stats["structured"]), str(stats["charts"])],
    ]
    table = Table(rows, colWidths=[4.1 * cm, 4.1 * cm, 4.1 * cm, 4.1 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f5")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f7fafc")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#102a43")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c7d1")),
                ("FONTNAME", (0, 0), (-1, -1), _register_font()),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, 1), 16),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return table


def _append_markdown(story: list, markdown_text: str, styles) -> None:
    from reportlab.platypus import Paragraph, Spacer

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(_markdown_inline_to_html(stripped[2:]), styles["Heading1"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_markdown_inline_to_html(stripped[3:]), styles["Heading2"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(_markdown_inline_to_html(stripped[4:]), styles["Heading3"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(f"- {_markdown_inline_to_html(stripped[2:])}", styles["Normal"]))
        else:
            story.append(Paragraph(_markdown_inline_to_html(stripped), styles["Normal"]))


def _markdown_to_html(markdown_text: str) -> str:
    lines = []
    list_tag = None
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            if list_tag:
                lines.append(f"</{list_tag}>")
                list_tag = None
            lines.append(f"<h2>{html.escape(stripped[2:])}</h2>")
        elif stripped.startswith("## "):
            if list_tag:
                lines.append(f"</{list_tag}>")
                list_tag = None
            lines.append(f"<h3>{html.escape(stripped[3:])}</h3>")
        elif stripped.startswith("### "):
            if list_tag:
                lines.append(f"</{list_tag}>")
                list_tag = None
            lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if list_tag != "ul":
                if list_tag:
                    lines.append(f"</{list_tag}>")
                lines.append("<ul>")
                list_tag = "ul"
            lines.append(f"<li>{_markdown_inline_to_html(stripped[2:])}</li>")
        elif re.match(r"^\d+\.\s+", stripped):
            if list_tag:
                lines.append(f"</{list_tag}>")
                list_tag = None
            lines.append(f"<p>{_markdown_inline_to_html(stripped)}</p>")
        else:
            if list_tag:
                lines.append(f"</{list_tag}>")
                list_tag = None
            lines.append(f"<p>{_markdown_inline_to_html(stripped)}</p>")
    if list_tag:
        lines.append(f"</{list_tag}>")
    return "\n".join(lines)


def _markdown_inline_to_html(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _table_style(colors):
    from reportlab.lib.units import cm
    from reportlab.platypus import TableStyle

    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102a43")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c7d1")),
            ("FONTNAME", (0, 0), (-1, -1), _register_font()),
            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ]
    )


def _html_logo(path: Path | None, alt: str) -> str:
    if not path or not path.exists():
        return "<span></span>"
    return f"<img class=\"logo\" src=\"{html.escape(_image_src(path))}\" alt=\"{html.escape(alt)}\" />"


def _image_src(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


def _cover_stats(report: GeneratedReport) -> dict[str, int]:
    metrics = public_metrics(report.report_input.structured_results)
    return {
        "documents": len(report.report_input.documents),
        "districts": len(metrics),
        "structured": len(metrics),
        "context": len(report.report_input.retrieval_context),
        "charts": len(report.charts),
    }


def _first_existing(directories: tuple[Path, ...], filename: str) -> Path | None:
    for directory in directories:
        path = directory / filename
        if path.exists():
            return path
    return None


def _fmt_number(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _fmt_growth(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font_name = "DejaVuSans"
    if font_path.exists() and font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    return font_name if font_path.exists() else "Helvetica"
