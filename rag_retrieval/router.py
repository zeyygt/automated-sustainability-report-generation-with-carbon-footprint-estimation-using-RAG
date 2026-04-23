"""Query routing and spreadsheet conversion helpers."""

from __future__ import annotations

from typing import Any

from .models import ElementType, ParsedDocument


def route_query(query: Any) -> str:
    """Route a processed query to the appropriate execution path."""

    intents = set(getattr(query, "intents", ()) or ())
    source_hints = set(getattr(query, "source_hints", ()) or ())

    if "analytical" in intents or "numeric" in intents:
        return "excel"
    if "pdf" in source_hints:
        return "pdf"
    if "spreadsheet" in source_hints:
        return "excel"
    return "hybrid"


def parsed_tables_to_dataframe(parsed_document: ParsedDocument):
    """Convert table elements from any parsed document into one DataFrame."""

    import pandas as pd

    frames = []
    for element in parsed_document.elements:
        element_type = getattr(element, "element_type", getattr(element, "type", None))
        if element_type not in {ElementType.TABLE, ElementType.TABLE.value, "table"}:
            continue

        headers = tuple(element.metadata.get("headers") or ())
        rows = tuple(element.metadata.get("rows") or ())
        if not headers or not rows:
            continue

        frames.append(pd.DataFrame(_pad_rows(rows, len(headers)), columns=headers))

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def spreadsheet_to_dataframe(parsed_document: ParsedDocument):
    """Convert spreadsheet table elements into one DataFrame."""

    return parsed_tables_to_dataframe(parsed_document)


def combine_dataframes(*dataframes):
    """Combine non-empty DataFrames while safely ignoring missing inputs."""

    import pandas as pd

    frames = [dataframe for dataframe in dataframes if dataframe is not None and not dataframe.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False)


def _pad_rows(rows, width: int) -> list[tuple]:
    padded_rows = []
    for row in rows:
        values = tuple(row)
        padded_rows.append((*values, *([""] * max(0, width - len(values))))[:width])
    return padded_rows
