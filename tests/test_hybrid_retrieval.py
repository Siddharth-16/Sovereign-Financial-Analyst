from langchain_core.documents import Document

from app.retrieval import _bm25_rank, _rrf_fuse, hybrid_retrieve


def test_bm25_prefers_exact_financial_row():
    docs = [
        Document(page_content="Other noncurrent liabilities and interest expense were discussed."),
        Document(page_content="Net income attributable to Pfizer Inc. common shareholders $ 8,031 $ 2,119 $ 31,372"),
        Document(page_content="The company discusses operations and future performance."),
    ]

    ranked = _bm25_rank("net income", docs, limit=3)
    assert ranked[0].page_content.startswith("Net income attributable")


def test_rrf_deduplicates_repeated_chunk_text():
    duplicate_old = Document(
        page_content="We operate in three reportable segments: Commercial Airplanes; Defense, Space & Security; Global Services.",
        metadata={"fiscal_year": 2023},
    )
    duplicate_new = Document(
        page_content=duplicate_old.page_content,
        metadata={"fiscal_year": 2025},
    )
    other = Document(page_content="Unrelated business narrative.", metadata={"fiscal_year": 2025})

    fused = _rrf_fuse([duplicate_old, other], [duplicate_new], limit=3)

    assert len(fused) == 2
    assert fused[0].metadata["fiscal_year"] == 2025


class _FakeVectorStore:
    def similarity_search(self, query, k, filter):
        return [
            Document(
                page_content="General discussion of liabilities and financing activities.",
                metadata={"company_slug": "pfizer", "section": "financial_statements", "fiscal_year": 2025},
            )
        ]

    def get(self, where, include):
        return {
            "documents": [
                "General discussion of liabilities and financing activities.",
                "Net income attributable to Pfizer Inc. common shareholders $ 8,031 $ 2,119 $ 31,372",
            ],
            "metadatas": [
                {"company_slug": "pfizer", "section": "financial_statements", "fiscal_year": 2025},
                {"company_slug": "pfizer", "section": "financial_statements", "fiscal_year": 2025},
            ],
        }


def test_hybrid_retrieval_can_recover_lexical_miss_without_increasing_final_k():
    docs = hybrid_retrieve(
        _FakeVectorStore(),
        "net income",
        k=1,
        filter_dict={"$and": [{"company_slug": "pfizer"}, {"section": "financial_statements"}]},
    )

    assert len(docs) == 1
    assert docs[0].page_content.startswith("Net income attributable")
