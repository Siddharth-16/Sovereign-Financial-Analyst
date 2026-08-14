from langchain_core.documents import Document

from app.retrieval import (
    _mdna_trend_score,
    _query_facets,
    _round_robin_facets,
)


def test_multifacet_queries_are_split_conservatively():
    assert _query_facets("business strategy and store formats") == [
        "business strategy",
        "store formats",
    ]
    assert _query_facets("environmental and regulatory risks") == [
        "environmental risks",
        "regulatory risks",
    ]
    assert _query_facets("manufacturing and production risks discussed") == [
        "manufacturing risks",
        "production risks discussed",
    ]


def test_round_robin_reserves_each_facet():
    a1 = Document(page_content="strategy one")
    a2 = Document(page_content="strategy two")
    b1 = Document(page_content="format one")
    b2 = Document(page_content="format two")
    out = _round_robin_facets([[a1, a2], [b1, b2]], per_facet=2)
    assert [d.page_content for d in out] == [
        "strategy one", "format one", "strategy two", "format two"
    ]


def test_mdna_realized_results_outrank_generic_trend_discussion():
    query = "advertising revenue trends in MD&A"
    realized = (
        "Google Search & other revenues increased $23.1 billion from 2023 to 2024. "
        "YouTube ads revenues increased $4.6 billion, driven by advertiser spending."
    )
    generic = (
        "We expect advertising trends to continue to affect our revenues. "
        "Changes in device mix and product mix may affect monetization."
    )
    assert _mdna_trend_score(query, realized) > _mdna_trend_score(query, generic)
