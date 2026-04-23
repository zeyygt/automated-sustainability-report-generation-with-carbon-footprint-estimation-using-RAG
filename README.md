# Automated Sustainability Report Generation with Carbon Footprint Estimation Using Retrieval-Augmented Generation

This project implements a modular sustainability reporting system that combines session-based retrieval, deterministic data analysis, carbon footprint estimation, and AI-assisted report generation for uploaded PDF and spreadsheet sources.

## Quick Start

```python
from rag_retrieval import RetrievalSession

session = RetrievalSession()
stats = session.build_index([
    "uploads/company-sustainability-report.pdf",
    "uploads/kpi-data.xlsx",
])
hits = session.search("Scope 1 emissions 2024", top_k=5)

for hit in hits:
    print(hit.score, hit.chunk.metadata["filename"], hit.chunk.page_start)
    print(hit.chunk.text[:500])
```

The package runs without external dependencies by falling back to a deterministic hashing embedder and in-memory indexes. For production-quality PDF parsing, spreadsheet parsing, and neural embeddings, install extras:

```bash
pip install ".[pdf,spreadsheet,embeddings]"
```

See [docs/retrieval_design.md](docs/retrieval_design.md) for the technical design, latency strategy, trade-offs, and evaluation plan.
