import unittest

from rag_retrieval.embeddings import HashingEmbedder
from rag_retrieval.index import BM25Index, InMemoryVectorIndex
from rag_retrieval.models import Chunk, ChunkType
from rag_retrieval.retrieval import RetrievalPipeline


class RetrievalTests(unittest.TestCase):
    def test_hybrid_retrieval_prioritizes_exact_numeric_table_query(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc",
                text="The company launched supplier engagement programs and biodiversity initiatives.",
                element_ids=("e1",),
                section_path=("Strategy",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TEXT,
            ),
            Chunk(
                chunk_id="c2",
                doc_id="doc",
                text=(
                    "Section: GHG Emissions. Table on page 7. "
                    "Headers: Metric, 2023, 2024, Unit. "
                    "Metric Scope 1 emissions; 2023: 12,400; 2024: 11,900; Unit: tCO2e."
                ),
                element_ids=("e2",),
                section_path=("GHG Emissions",),
                page_start=7,
                page_end=7,
                chunk_type=ChunkType.TABLE,
            ),
        ]
        embedder = HashingEmbedder()
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex()
        keyword_index = BM25Index()
        vector_index.add(chunks, vectors)
        keyword_index.add(chunks)

        hits = RetrievalPipeline(vector_index, keyword_index, embedder).search("Scope 1 emissions 2024", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "c2")
        self.assertIn("scope", hits[0].matched_terms)
        self.assertGreater(hits[0].keyword_score, 0)

    def test_retrieval_matches_turkish_names_with_ascii_query(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc",
                text="Yıl 2023; Ay No: 4; İlçe: ZEYTİNBURNU; Dogalgaz Tüketim Miktarı (m3): 9,465,693.679.",
                element_ids=("e1",),
                section_path=("Tüketim Verileri",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TABLE,
            ),
            Chunk(
                chunk_id="c2",
                doc_id="doc",
                text="Yıl 2023; Ay No: 4; İlçe: ADALAR; Dogalgaz Tüketim Miktarı (m3): 190,000.",
                element_ids=("e2",),
                section_path=("Tüketim Verileri",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TABLE,
            ),
        ]
        embedder = HashingEmbedder()
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex()
        keyword_index = BM25Index()
        vector_index.add(chunks, vectors)
        keyword_index.add(chunks)

        hits = RetrievalPipeline(vector_index, keyword_index, embedder).search("zeytinburnu 2023 dogalgaz", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "c1")
        self.assertIn("zeytinburnu", hits[0].matched_terms)

    def test_implicit_carbon_query_expands_to_energy_activity_data(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc",
                text="District natural gas consumption table with annual energy use values.",
                element_ids=("e1",),
                section_path=("Energy",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TABLE,
            ),
            Chunk(
                chunk_id="c2",
                doc_id="doc",
                text="Employee training attendance and survey participation summary.",
                element_ids=("e2",),
                section_path=("People",),
                page_start=2,
                page_end=2,
                chunk_type=ChunkType.TEXT,
            ),
        ]
        embedder = HashingEmbedder()
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex()
        keyword_index = BM25Index()
        vector_index.add(chunks, vectors)
        keyword_index.add(chunks)

        hits = RetrievalPipeline(vector_index, keyword_index, embedder).search("carbon footprint by district", top_k=2)

        self.assertEqual(hits[0].chunk.chunk_id, "c1")
        self.assertIn("natural", hits[0].matched_terms)

    def test_cross_source_query_preserves_pdf_and_spreadsheet_results(self):
        chunks = [
            Chunk(
                chunk_id="pdf",
                doc_id="pdf-doc",
                text="Annual PDF report table for district natural gas consumption.",
                element_ids=("e1",),
                section_path=("Annual Report",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TABLE,
                metadata={"filename": "report.pdf"},
            ),
            Chunk(
                chunk_id="xlsx",
                doc_id="xlsx-doc",
                text="Spreadsheet monthly district dogalgaz consumption table.",
                element_ids=("e2",),
                section_path=("Workbook",),
                page_start=1,
                page_end=1,
                chunk_type=ChunkType.TABLE,
                metadata={"filename": "data.xlsx", "source_type": "spreadsheet"},
            ),
        ]
        embedder = HashingEmbedder()
        vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        vector_index = InMemoryVectorIndex()
        keyword_index = BM25Index()
        vector_index.add(chunks, vectors)
        keyword_index.add(chunks)

        hits = RetrievalPipeline(vector_index, keyword_index, embedder).search("compare PDF and spreadsheet district gas consumption", top_k=2)
        filenames = {hit.chunk.metadata.get("filename") for hit in hits}

        self.assertEqual(filenames, {"report.pdf", "data.xlsx"})


if __name__ == "__main__":
    unittest.main()
