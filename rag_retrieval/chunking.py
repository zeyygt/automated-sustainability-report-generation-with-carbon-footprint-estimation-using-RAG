"""Semantic, structure-aware chunking for sustainability documents."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .config import ChunkingConfig
from .models import Chunk, ChunkType, ElementType, ParsedDocument, ParsedElement


class StructureAwareChunker:
    """Creates chunks around sections, paragraph groups, and table elements."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def chunk_documents(self, documents: Iterable[ParsedDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[ParsedElement] = []

        for element in document.elements:
            if element.element_type == ElementType.HEADING:
                self._flush_text_buffer(document, buffer, chunks)
                buffer = []
                continue

            if element.element_type == ElementType.TABLE:
                self._flush_text_buffer(document, buffer, chunks)
                buffer = []
                chunks.extend(self._table_chunks(document, element, len(chunks)))
                continue

            candidate = [*buffer, element]
            if buffer and self._token_count(self._join_elements(candidate)) > self.config.max_tokens:
                self._flush_text_buffer(document, buffer, chunks)
                buffer = self._overlap_tail(buffer)
            buffer.append(element)

        self._flush_text_buffer(document, buffer, chunks)
        return chunks

    def _flush_text_buffer(self, document: ParsedDocument, buffer: list[ParsedElement], chunks: list[Chunk]) -> None:
        if not buffer:
            return
        text = self._join_elements(buffer)
        section_path = buffer[-1].section_path
        if self.config.include_section_path and section_path:
            text = f"Section: {' > '.join(section_path)}\n{text}"
        chunk_id = f"{document.doc_id}:chunk:{len(chunks) + 1}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=text,
                element_ids=tuple(element.element_id for element in buffer),
                section_path=section_path,
                page_start=min(element.page for element in buffer),
                page_end=max(element.page for element in buffer),
                chunk_type=ChunkType.TEXT,
                metadata={"filename": document.filename},
            )
        )

    def _table_chunks(self, document: ParsedDocument, element: ParsedElement, chunk_offset: int) -> list[Chunk]:
        headers = tuple(str(header) for header in element.metadata.get("headers", ()))
        rows = tuple(tuple(str(cell) for cell in row) for row in element.metadata.get("rows", ()))
        rows_per_chunk = self._rows_per_table_chunk(element)
        row_groups = [
            rows[index : index + rows_per_chunk]
            for index in range(0, len(rows), rows_per_chunk)
        ] or [()]

        chunks: list[Chunk] = []
        for group_index, row_group in enumerate(row_groups, start=1):
            text = self._render_table_text(element, headers, row_group, group_index, len(row_groups))
            chunk_id = f"{document.doc_id}:chunk:{chunk_offset + group_index}"
            first_data_row = element.metadata.get("first_data_row")
            chunk_first_data_row = None
            chunk_last_data_row = None
            if isinstance(first_data_row, int):
                chunk_first_data_row = first_data_row + ((group_index - 1) * rows_per_chunk)
                chunk_last_data_row = chunk_first_data_row + max(len(row_group) - 1, 0)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=text,
                    element_ids=(element.element_id,),
                    section_path=element.section_path,
                    page_start=element.page,
                    page_end=element.page,
                    chunk_type=ChunkType.TABLE,
                    metadata={
                        "filename": document.filename,
                        "headers": headers,
                        "row_count": len(row_group),
                        "table_number": element.metadata.get("table_number"),
                        "sheet_name": element.metadata.get("sheet_name"),
                        "cell_range": element.metadata.get("cell_range"),
                        "source_type": element.metadata.get("source_type"),
                        "first_data_row": chunk_first_data_row or element.metadata.get("first_data_row"),
                        "last_data_row": chunk_last_data_row or element.metadata.get("last_data_row"),
                        "row_group_index": group_index,
                        "row_group_count": len(row_groups),
                    },
                )
            )
        return chunks

    def _rows_per_table_chunk(self, element: ParsedElement) -> int:
        if element.metadata.get("source_type") == "spreadsheet":
            return max(self.config.spreadsheet_table_rows_inline, 1)
        return max(self.config.max_table_rows_inline, 1)

    def _render_table_text(
        self,
        element: ParsedElement,
        headers: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
        group_index: int,
        group_count: int,
    ) -> str:
        prefix_parts = []
        if element.section_path and self.config.include_section_path:
            prefix_parts.append(f"Section: {' > '.join(element.section_path)}")
        if element.metadata.get("source_type") == "spreadsheet":
            sheet_name = element.metadata.get("sheet_name", f"Sheet {element.page}")
            cell_range = element.metadata.get("cell_range")
            table_location = f"Sheet: {sheet_name}"
            if cell_range:
                table_location += f", range {cell_range}"
            prefix_parts.append(table_location)
        else:
            prefix_parts.append(f"Table on page {element.page}")
        if group_count > 1:
            prefix_parts.append(f"rows part {group_index} of {group_count}")

        lines = [". ".join(prefix_parts) + "."]
        if headers:
            lines.append(f"Headers: {', '.join(headers)}.")
        lines.extend(self._table_key_value_sentences(headers, rows))
        lines.append("")
        lines.append(_markdown_table(headers, rows))
        return "\n".join(line for line in lines if line is not None).strip()

    def _table_key_value_sentences(self, headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[str]:
        if not headers or not rows:
            return []
        lines: list[str] = []
        for row in rows:
            padded = [*row, *([""] * max(0, len(headers) - len(row)))]
            label = padded[0] or "Row"
            pairs = [
                f"{header}: {value}"
                for header, value in zip(headers[1:], padded[1:])
                if header and value
            ]
            if pairs:
                lines.append(f"{headers[0] or 'Metric'} {label}; " + "; ".join(pairs) + ".")
        return lines

    def _join_elements(self, elements: list[ParsedElement]) -> str:
        return "\n\n".join(element.text for element in elements if element.text.strip())

    def _overlap_tail(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        if self.config.overlap_tokens <= 0:
            return []
        tail: list[ParsedElement] = []
        total = 0
        for element in reversed(elements):
            count = self._token_count(element.text)
            if total + count > self.config.overlap_tokens and tail:
                break
            tail.append(element)
            total += count
        return list(reversed(tail))

    @staticmethod
    def _token_count(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text))


def _markdown_table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    if not headers:
        return "\n".join(" | ".join(row) for row in rows)
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for row in rows:
        padded = [*row, *([""] * max(0, len(headers) - len(row)))]
        lines.append(" | ".join(padded[: len(headers)]))
    return "\n".join(lines)
