# Retrieval Pipeline Design

## 1. Goal And Scope

This design implements the retrieval layer for a session-based sustainability reporting RAG system. Users upload documents during a live session, the system parses and indexes those files immediately, and all derived data can be discarded when the session ends.

Generation is deliberately out of scope. The retrieval layer returns ranked, provenance-rich chunks that a future report-generation module can cite, summarize, and reconcile.

## 2. Architecture Summary

The pipeline is modular and replaceable:

| Module | Responsibility | Input | Output |
| --- | --- | --- | --- |
| Document ingestion | Validate runtime uploads and assign session document IDs | File paths or upload objects | `DocumentInput[]` |
| Parsing | Extract ordered layout elements, headings, paragraphs, tables, and OCR fallback text | `DocumentInput` | `ParsedDocument` |
| Chunking | Build section-aware text chunks and table chunks | `ParsedDocument` | `Chunk[]` |
| Embedding | Convert chunks and queries into dense vectors | `Chunk.text`, query text | Embedding vectors |
| Indexing | Store temporary dense vectors and lexical term stats | `Chunk[]`, vectors | Session-local index |
| Retrieval | Query processing, hybrid candidate retrieval, fusion, reranking, and context selection | Query string | `RetrievalHit[]` |
| Evaluation | Measure retrieval quality without generation | Queries, relevant chunk IDs | Recall@K, MRR, precision@K |

Current implementation files:

- `rag_retrieval/ingestion.py`
- `rag_retrieval/parsing.py`
- `rag_retrieval/chunking.py`
- `rag_retrieval/embeddings.py`
- `rag_retrieval/index.py`
- `rag_retrieval/retrieval.py`
- `rag_retrieval/session.py`
- `rag_retrieval/evaluation.py`

## 3. Session Lifecycle

Each live user session owns isolated runtime state:

1. Frontend uploads files.
2. Backend creates a `RetrievalSession`.
3. `RetrievalSession.build_index(file_paths)` parses, chunks, embeds, and indexes the uploads.
4. Chatbot or report planner calls `RetrievalSession.search(query)`.
5. On session close, `RetrievalSession.clear()` removes chunks, vectors, lexical stats, and parsed documents.

There is no global corpus, no cross-session search, and no persistent vector collection by default.

## 4. Data Flow

```text
Uploaded files
  -> DocumentIngestor
  -> PyMuPDFLayoutParser / fallback parser / OCR fallback hook
  -> ParsedDocument(elements=[heading, paragraph, table, ...])
  -> StructureAwareChunker
  -> Chunk(text, section_path, page range, element ids, table metadata)
  -> Embedder
  -> Dense vector index + BM25 keyword index
  -> RetrievalPipeline
  -> RetrievalHit(chunk, score breakdown, matched terms)
```

The key contract is that chunks retain provenance:

- `doc_id`
- `filename`
- `page_start` and `page_end`
- `section_path`
- `element_ids`
- table headers and row counts where applicable

This gives the future generator enough metadata to cite pages, avoid hallucinated values, and detect conflicting values across documents.

## 5. Document Ingestion

Primary formats:

- PDF: main target
- XLSX/CSV: structured KPI, emissions, energy, water, waste, and workforce tables
- TXT/Markdown: useful for testing and notes
- Images: supported as an OCR fallback path, not as the default path

The ingestor performs only lightweight validation. It does not parse content and does not persist files outside the session workflow. In a web app, this module should accept temporary upload paths from the API layer and return `DocumentInput` records.

## 6. Spreadsheet Parsing

Spreadsheet uploads are parsed into the same `ParsedElement(TABLE)` representation used by PDF tables.

Current behavior:

- `.xlsx` and `.xlsm` are read with `openpyxl` in `read_only` and `data_only` mode.
- `.csv` is read with Python's standard CSV parser.
- Each sheet becomes a section.
- Contiguous non-empty row regions become tables.
- The first row with at least two non-empty cells is treated as the header row.
- Empty leading/trailing columns are trimmed.
- Metadata preserves `sheet_name`, `cell_range`, headers, rows, and data row bounds.
- Numeric values are rendered with stable text formatting for keyword retrieval.

This is faster and more reliable than extracting equivalent tables from PDFs because the rows and columns are already structured.

## 7. Layout-Aware Parsing

### Fast Default Path

The implemented parser uses PyMuPDF when available:

- `page.get_text("dict", sort=True)` for layout blocks, lines, spans, font sizes, and bounding boxes.
- Heading detection from numbering, font-size deltas, bold-like font flags, and short uppercase headings.
- Reading order from positional sorting.
- `page.find_tables()` for native table detection and extraction.
- Table text blocks are excluded from paragraph extraction when they overlap table bounding boxes.

This is layout-aware but still fast because it uses PDF-native text and vector data instead of rendering every page into images.

### OCR Policy

OCR is used only when extracted text is below a configured threshold or the input is an image. This avoids wasting most of the 30-60 second budget on born-digital PDFs. The current code exposes an `OCRFallback` protocol; a production app can plug in Tesseract, PaddleOCR, a cloud OCR service, or a vision model.

### Higher-Accuracy Alternative

For documents with messy layouts, complex tables, footnotes, or multi-column annual reports, swap the parser with a Docling-based implementation. Docling is designed to convert messy documents into structured data and supports tables, reading order, bounding boxes, headers, paragraphs, captions, and OCR-capable imports. The trade-off is latency and dependency weight.

## 8. Chunking Strategy

The chunker avoids fixed-size chunking as the main unit. It uses document structure first:

- Headings update the active `section_path`.
- Paragraphs are grouped within the current section until a soft token budget is reached.
- Tables become dedicated chunks.
- Large tables are split by row groups, preserving headers in every split.
- Spreadsheet tables use smaller row groups than PDF tables by default, improving exact row retrieval while keeping the pipeline fast.

Text chunks look like:

```text
Section: Environmental Performance > GHG Emissions
The company reduced Scope 1 emissions...
```

Table chunks include both a natural-language representation and a markdown-style table:

```text
Section: GHG Emissions. Table on page 14.
Headers: Metric, 2023, 2024, Unit.
Metric Scope 1 emissions; 2023: 12,400; 2024: 11,900; Unit: tCO2e.

Metric | 2023 | 2024 | Unit
--- | --- | --- | ---
Scope 1 emissions | 12,400 | 11,900 | tCO2e
```

This improves retrieval for both natural language questions and numeric keyword queries like `Scope 1 emissions 2024`.

## 9. Embedding And Indexing

### Current Implementation

The code includes:

- `SentenceTransformerEmbedder` for production neural embeddings when `sentence-transformers` is installed.
- `HashingEmbedder` as a deterministic no-dependency fallback for tests and constrained environments.
- `InMemoryVectorIndex` for cosine search over temporary session chunks.
- `BM25Index` for keyword search.

The no-dependency index is appropriate for small live-session corpora. For larger files or many documents, replace it behind the same interface with FAISS, Qdrant local mode, LanceDB, or another embedded vector store.

### Recommended Production Setup

Fast default:

- Parser: PyMuPDF native extraction.
- Embeddings: small CPU-friendly embedding model such as `BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, or a FastEmbed-supported dense model.
- Vector store: in-memory Qdrant local mode, FAISS, or LanceDB per session.
- Keyword path: BM25, SPLADE/miniCOIL sparse vectors, or Qdrant sparse vectors.

Higher accuracy:

- Parser: Docling for harder PDFs.
- Rerank: cross-encoder reranker or late-interaction reranker over top 20-50 candidates.
- Vector store: Qdrant hybrid collection with dense and sparse vectors.

## 10. Retrieval Pipeline

The retrieval path has four stages:

1. Query processing
   - Normalize whitespace and case.
   - Preserve quoted phrases.
   - Extract numbers, years, and Scope 1/2/3 terms.
   - Normalize Turkish characters for robust matching, so `ZEYTİNBURNU` can match `zeytinburnu`.
   - Expand sustainability concepts such as `carbon footprint`, `greenhouse gas`, `fossil fuel`, and `energy efficiency baseline` into activity-data terms such as natural gas, energy, and consumption.
   - Infer source hints from terms such as PDF, report, annual, spreadsheet, workbook, Excel, monthly, and multi-year annual ranges.
   - This matters because sustainability queries often combine semantics with exact values.

2. Initial retrieval
   - Dense vector search handles semantic queries such as `greenhouse gas reduction initiatives`.
   - BM25 keyword search handles exact terms such as `Scope 1 emissions 2024`.

3. Fusion and reranking
   - Reciprocal Rank Fusion combines dense and lexical candidates.
   - Numeric and scope exact matches receive lightweight boosts.
   - Table chunks receive stronger boosts for numeric, analytical, implicit-energy, monthly, and annual-table queries.
   - Specific entity terms such as district names are boosted above generic words such as report, workbook, natural, gas, and consumption.
   - Source hints boost spreadsheet chunks for monthly/workbook queries and PDF chunks for annual/report queries.
   - A fast heuristic reranker is implemented; a cross-encoder reranker can replace it.

4. Final context selection
   - Select top chunks after reranking.
   - Limit repeated chunks from the same section to improve diversity.
   - Preserve source diversity for explicit cross-document queries without forcing lower-scoring sources to the top rank.
   - Return score breakdowns and matched terms for debugging.

## 11. Latency Design

Target: 30 seconds to 1 minute for parsing, chunking, embedding, and indexing typical session uploads.

Latency tactics:

- Use PDF-native extraction before OCR.
- Parse page text and tables in one pass.
- Keep tables as structured text, not image crops, unless extraction fails.
- Batch embeddings with `EmbeddingConfig.batch_size`.
- Use small embedding models by default.
- Use in-memory session indexes to avoid network and disk round trips.
- Retrieve top 40 dense and top 40 lexical candidates, then rerank only top 20-30.
- Make OCR page-limited through `ParserConfig.max_ocr_pages`.
- Defer heavy Docling or vision parsing to fallback or user-selected high-accuracy mode.

Expected budget for a moderate born-digital PDF bundle:

| Step | Target |
| --- | --- |
| Ingestion and validation | <1s |
| PyMuPDF parsing | 2-15s depending on pages and tables |
| Chunking | <2s |
| CPU embedding with small model | 5-30s depending on chunk count |
| Index build | <2s for small session corpora |

OCR-heavy scanned PDFs can exceed the target. The practical design choice is to process the first N pages quickly, warn the caller about OCR fallback, and optionally continue processing asynchronously.

## 12. Trade-Offs

PyMuPDF versus Docling:

- PyMuPDF is fast and lightweight for born-digital PDFs.
- Docling captures richer structure and harder layouts but costs more time and dependencies.

Small embeddings versus large embeddings:

- Small models keep session startup fast on CPU.
- Larger models improve recall and paraphrase robustness but may violate the 1-minute target without GPU or batching.

Heuristic reranker versus cross-encoder:

- Heuristic reranking is cheap and good for exact sustainability metrics.
- Cross-encoders improve relevance but score query/chunk pairs one by one, so they should only rerank a small candidate set.

In-memory index versus persistent vector DB:

- In-memory indexes are simplest and fastest for session data.
- Qdrant/FAISS/LanceDB are better if sessions contain hundreds of documents, need approximate nearest neighbor search, or need native sparse+dense hybrid querying.

## 13. Evaluation Strategy

Evaluate retrieval without generation.

Ground truth creation:

- Select a representative set of sustainability PDFs.
- Create 30-100 queries per document bundle.
- Include factual metric queries, semantic policy queries, table queries, and ambiguous queries.
- Annotate relevant chunk IDs, not generated answers.
- Include page-level provenance in the label sheet to verify parser and chunk alignment.

Metrics:

- Recall@K: whether relevant chunks are retrieved.
- MRR: whether the first relevant chunk is highly ranked.
- Precision@K: how much final context is useful.
- nDCG@K: optional graded relevance for partially relevant chunks.
- Table Recall@K: separate slice for numeric table queries.
- Latency p50/p95: build time and query time.

Experiment matrix:

- Dense only versus BM25 only versus hybrid.
- Fixed-size chunks versus structure-aware chunks.
- Table markdown only versus table markdown plus key-value sentences.
- Heuristic rerank versus cross-encoder rerank.
- PyMuPDF parser versus Docling parser on hard PDFs.

## 14. Extension Points

The package is intentionally interface-driven:

- Replace `DocumentParser` with a Docling parser.
- Replace `Embedder` with OpenAI embeddings, FastEmbed, or a local ONNX model.
- Replace `InMemoryVectorIndex` with Qdrant, FAISS, or LanceDB.
- Replace `HeuristicReranker` with CrossEncoder, ColBERT, or a hosted rerank API.
- Add metadata filters for document type, reporting year, page range, source organization, or framework tags such as GRI, SASB, ESRS, TCFD, and ISSB.
- Add multimodal retrieval for charts and scanned tables by storing image crops with captions and page coordinates.

## 15. References

- PyMuPDF text extraction docs: https://pymupdf.readthedocs.io/en/latest/app1.html
- PyMuPDF table extraction docs: https://pymupdf.readthedocs.io/en/latest/the-basics.html#extracting-tables-from-a-page
- Docling project overview and features: https://www.docling.ai/
- Qdrant hybrid retrieval and reranking docs: https://qdrant.tech/documentation/search-precision/reranking-hybrid-search/
- Qdrant hybrid search course: https://qdrant.tech/course/essentials/day-3/hybrid-search/
- Sentence Transformers retrieve-and-rerank docs: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Sentence Transformers CrossEncoder docs: https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
