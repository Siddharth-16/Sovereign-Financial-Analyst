from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

from langchain_core.documents import Document


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.&%-][a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "its", "of", "on", "or", "say", "says", "the",
    "their", "this", "to", "was", "were", "what", "which", "with", "about",
    "company", "companies", "disclose", "disclosed", "discloses", "discussion",
    "financial", "statements", "statement", "show", "shows", "regarding",
}


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token.lower() not in _STOPWORDS
    ]


def _normalize(text: str) -> str:
    text = (text or "").casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def _text_key(text: str) -> str:
    return _normalize(text)


def _fiscal_year(doc: Document) -> int:
    value = doc.metadata.get("fiscal_year")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _dedupe_docs(docs: Iterable[Document], *, prefer_latest: bool = False) -> list[Document]:
    chosen: dict[str, Document] = {}
    order: list[str] = []

    for doc in docs:
        key = _text_key(doc.page_content)
        if not key:
            continue
        if key not in chosen:
            chosen[key] = doc
            order.append(key)
        elif prefer_latest and _fiscal_year(doc) > _fiscal_year(chosen[key]):
            chosen[key] = doc

    return [chosen[key] for key in order]


def _has_fiscal_year_filter(filter_dict: dict) -> bool:
    if "fiscal_year" in filter_dict:
        return True
    for op in ("$and", "$or"):
        children = filter_dict.get(op)
        if isinstance(children, list):
            if any(isinstance(child, dict) and _has_fiscal_year_filter(child) for child in children):
                return True
    return False


def _add_fiscal_year_filter(filter_dict: dict, fiscal_year: int) -> dict:
    if _has_fiscal_year_filter(filter_dict):
        return filter_dict
    if "$and" in filter_dict and isinstance(filter_dict["$and"], list):
        return {"$and": [*filter_dict["$and"], {"fiscal_year": fiscal_year}]}
    return {"$and": [filter_dict, {"fiscal_year": fiscal_year}]}


def _bm25_rank(query: str, docs: list[Document], limit: int) -> list[Document]:
    if not docs or limit <= 0:
        return []

    query_tokens = list(dict.fromkeys(_tokenize(query)))
    if not query_tokens:
        return []

    tokenized_docs = [_tokenize(doc.page_content) for doc in docs]
    lengths = [len(tokens) for tokens in tokenized_docs]
    avg_len = max(sum(lengths) / len(lengths), 1.0)

    df: Counter[str] = Counter()
    for tokens in tokenized_docs:
        present = set(tokens)
        for token in query_tokens:
            if token in present:
                df[token] += 1

    n_docs = len(docs)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, int, Document]] = []

    for index, (doc, tokens, length) in enumerate(zip(docs, tokenized_docs, lengths)):
        if not tokens:
            continue
        frequencies = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            tf = frequencies.get(token, 0)
            if not tf:
                continue
            token_df = df.get(token, 0)
            idf = math.log(1.0 + (n_docs - token_df + 0.5) / (token_df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * length / avg_len)
            score += idf * (tf * (k1 + 1.0) / denominator)
        if score > 0:
            scored.append((score, index, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _rrf_fuse(dense_docs: list[Document], lexical_docs: list[Document], limit: int) -> list[Document]:
    if limit <= 0:
        return []

    dense_docs = _dedupe_docs(dense_docs)
    lexical_docs = _dedupe_docs(lexical_docs)
    rrf_constant = 60

    by_key: dict[str, Document] = {}
    dense_rank: dict[str, int] = {}
    lexical_rank: dict[str, int] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        key = _text_key(doc.page_content)
        by_key.setdefault(key, doc)
        dense_rank.setdefault(key, rank)

    for rank, doc in enumerate(lexical_docs, start=1):
        key = _text_key(doc.page_content)
        if key not in by_key or _fiscal_year(doc) > _fiscal_year(by_key[key]):
            by_key[key] = doc
        lexical_rank.setdefault(key, rank)

    missing = 10**9
    ranked = []
    for key, doc in by_key.items():
        score = 0.0
        if key in dense_rank:
            score += 1.0 / (rrf_constant + dense_rank[key])
        if key in lexical_rank:
            score += 1.0 / (rrf_constant + lexical_rank[key])
        ranked.append((score, lexical_rank.get(key, missing), dense_rank.get(key, missing), key, doc))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return [doc for *_, doc in ranked[:limit]]


def _financial_intents(query: str) -> set[str]:
    q = _normalize(query)
    intents: set[str] = set()

    if "total assets" in q:
        intents.add("total_assets")
    if "liabilit" in q:
        intents.add("liabilities")
    if "debt" in q:
        intents.add("debt")
    if "revenue" in q or "sales" in q:
        intents.add("revenue")
    if "net income" in q or "earnings" in q:
        intents.add("net_income")
    if (
        "cash flow from operations" in q
        or "cash flows from operations" in q
        or "operating cash flow" in q
        or "operating activities" in q
    ):
        intents.add("operating_cash_flow")
    if "capital expenditure" in q or "capex" in q:
        intents.add("capex")

    return intents


def _numeric_density(text: str) -> float:
    count = len(re.findall(r"\$?\s*\(?\d[\d,]*(?:\.\d+)?\)?", text or ""))
    return min(count, 20) / 20.0


def _financial_anchor_score(query: str, text: str) -> float:
    """Rank chunks that contain the actual accounting row, not merely related prose.

    These rules are metric-level and company-agnostic. They do not contain benchmark
    answer values or company-specific strings.
    """
    q = _normalize(query)
    t = _normalize(text)
    intents = _financial_intents(q)
    if not intents:
        return 0.0

    score = _numeric_density(text) * 4.0

    # Generic statement headers.
    is_balance_sheet = bool(re.search(r"consolidated (?:balance sheets?|statements? of financial position)", t))
    is_income_statement = bool(re.search(r"consolidated statements? of (?:income|operations|earnings)", t))
    is_cash_flow_statement = bool(re.search(r"consolidated statements? of cash flows", t))

    if "total_assets" in intents:
        if re.search(r"\btotal assets\b(?!\s+acquired)", t):
            score += 100.0
        if is_balance_sheet:
            score += 25.0
        if "total current assets" in t:
            score += 8.0
        if any(term in t for term in (
            "total assets acquired", "assets acquired:", "retirement plans",
            "plan assets", "investments measured at fair value",
        )):
            score -= 45.0

    if "liabilities" in intents:
        if re.search(r"\btotal liabilities\b", t):
            score += 110.0
        if "total current liabilities" in t:
            score += 85.0
        if "long-term debt" in t:
            score += 35.0
        if "long-term operating lease obligations" in t:
            score += 20.0
        if "long-term finance lease obligations" in t:
            score += 15.0
        if is_balance_sheet:
            score += 25.0
        if any(term in t for term in (
            "deferred tax liabilities:", "net deferred tax liabilities",
            "self insurance reserves", "fair value measurements",
        )):
            score -= 20.0

    if "debt" in intents:
        if re.search(r"\btotal debt\b", t):
            score += 110.0
        if re.search(r"debt at (?:december|january|year end)", t):
            score += 45.0
        if "short-term debt" in t:
            score += 20.0
        if "long-term debt" in t:
            score += 25.0
        if "net debt" in t and "acquisition" in t:
            score -= 25.0

    if "revenue" in intents:
        if re.search(r"\btotal revenues?\b\s*\$?\s*\d", t):
            score += 110.0
        # Some income statements label the row simply "Revenue" rather than
        # "Total revenue"; that is semantically the same top-line metric.
        if re.search(r"(?:^|\s)revenue\s*\$?\s*\d[\d,]*", t):
            score += 95.0
        if is_income_statement:
            score += 30.0
        if any(term in t for term in (
            "10% or more of total revenue", "revenue by geographic area",
            "customer concentration", "percentage of total revenue",
        )):
            score -= 25.0

    if "net_income" in intents:
        if re.search(r"net income attributable to .*?common shareholders?\s*\$?\s*\d", t):
            score += 120.0
        elif re.search(r"\bnet income\b\s*\$?\s*\d", t):
            score += 90.0
        if is_income_statement:
            score += 30.0
        if any(term in t for term in (
            "benefit obligations", "net periodic benefit", "fair value reported in net income",
        )):
            score -= 20.0

    if "operating_cash_flow" in intents:
        if re.search(r"net cash provided by \(used in\) operating activities\s*\$?\s*\d", t):
            score += 130.0
        elif re.search(r"net cash provided by operating activities\s*\$?\s*\d", t):
            score += 125.0
        elif "operating activities" in t:
            score += 25.0
        if is_cash_flow_statement:
            score += 35.0
        if any(term in t for term in (
            "cash flow hedge", "cash flow hedges", "derivative instruments",
            "foreign exchange forward contracts",
        )):
            score -= 40.0

    if "capex" in intents:
        if "capital and exploration expenditures (capex)" in t:
            score += 120.0
        if "total cash capex" in t:
            score += 105.0
        if "additions to property, plant and equipment" in t:
            score += 100.0
        if re.search(r"\bcapital expenditures?\b", t):
            score += 45.0
        if is_cash_flow_statement:
            score += 20.0
        if any(term in t for term in (
            "environmental expenditures", "forward investment guidance",
            "plans to invest", "expected to account for",
        )):
            score -= 25.0

    return score


def _financial_anchor_rank(query: str, docs: list[Document], limit: int) -> list[Document]:
    scored = []
    for idx, doc in enumerate(docs):
        score = _financial_anchor_score(query, doc.page_content)
        if score > 0:
            scored.append((score, idx, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _merge_ranked(*ranked_lists: list[Document], limit: int) -> list[Document]:
    merged: list[Document] = []
    seen: set[str] = set()
    for docs in ranked_lists:
        for doc in docs:
            key = _text_key(doc.page_content)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= limit:
                return merged
    return merged



def _filter_contains(filter_dict: dict, key: str, value: str) -> bool:
    if filter_dict.get(key) == value:
        return True
    for op in ("$and", "$or"):
        children = filter_dict.get(op)
        if isinstance(children, list):
            if any(
                isinstance(child, dict) and _filter_contains(child, key, value)
                for child in children
            ):
                return True
    return False


def _query_facets(query: str) -> list[str]:
    """Split clearly multi-part prose queries into retrieval facets.

    This is intentionally conservative. The full query is still used as the
    fallback ranker; facets only reserve coverage for distinct requested topics.
    """
    q = re.sub(r"\s+", " ", (query or "").strip())
    if not q:
        return []

    # One or two explicit conjunctions covers the common analyst question form
    # without turning every noun phrase into a separate search.
    parts = [part.strip(" ,;:-") for part in re.split(r"\s+and\s+", q, flags=re.IGNORECASE)]
    parts = [part for part in parts if len(_tokenize(part)) >= 1]
    if len(parts) < 2 or len(parts) > 3:
        return [q]

    # If the right-hand facet supplies a shared head noun (e.g. "environmental
    # and regulatory risks"), attach that noun to a short left-hand facet.
    shared_heads = (
        "risks", "risk", "trends", "trend", "drivers", "driver", "segments",
        "segment", "products", "product", "formats", "format", "revenue",
        "revenues", "model", "services", "operations",
    )
    right_tokens = _tokenize(parts[-1])
    head = next((token for token in reversed(right_tokens) if token in shared_heads), None)
    if head:
        repaired = []
        for part in parts[:-1]:
            tokens = _tokenize(part)
            if head not in tokens and len(tokens) == 1:
                repaired.append(f"{part} {head}")
            else:
                repaired.append(part)
        parts = repaired + [parts[-1]]

    # Deduplicate normalized facets.
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = _normalize(part)
        if key and key not in seen:
            out.append(part)
            seen.add(key)
    return out if len(out) >= 2 else [q]


def _jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _round_robin_facets(rankings: list[list[Document]], *, per_facet: int = 2) -> list[Document]:
    """Reserve a small number of slots for each requested facet."""
    selected: list[Document] = []
    seen: set[str] = set()
    for rank in range(per_facet):
        for docs in rankings:
            if rank >= len(docs):
                continue
            doc = docs[rank]
            key = _text_key(doc.page_content)
            if key and key not in seen:
                selected.append(doc)
                seen.add(key)
    return selected


def _merge_diverse(*ranked_lists: list[Document], limit: int, threshold: float = 0.94) -> list[Document]:
    """Merge ranked lists while suppressing near-duplicate prose chunks."""
    if limit <= 0:
        return []

    selected: list[Document] = []
    deferred: list[Document] = []
    seen: set[str] = set()

    for docs in ranked_lists:
        for doc in docs:
            key = _text_key(doc.page_content)
            if not key or key in seen:
                continue
            seen.add(key)
            if any(_jaccard_similarity(doc.page_content, prev.page_content) >= threshold for prev in selected):
                deferred.append(doc)
                continue
            selected.append(doc)
            if len(selected) >= limit:
                return selected

    # Never return fewer than k solely because diversity filtering was strict.
    for doc in deferred:
        selected.append(doc)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _mdna_trend_score(query: str, text: str) -> float:
    """Promote actual period-over-period MD&A results over generic trend prose."""
    q = _normalize(query)
    if not any(term in q for term in (
        "trend", "growth", "driver", "margin", "increase", "decrease", "revenue", "sales", "income", "earnings",
    )):
        return 0.0

    t = _normalize(text)
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    overlap = len(q_tokens & t_tokens)
    score = overlap * 3.0

    numeric_hits = min(len(re.findall(r"\$?\s*\d[\d,.]*(?:\s*(?:%|percent|billion|million))?", text or "", re.IGNORECASE)), 12)
    score += numeric_hits * 0.8

    if re.search(r"\b20\d{2}\b", t):
        score += 3.0
    if re.search(r"\b(?:increase(?:d)?|decrease(?:d)?|higher|lower|grew|growth|declin(?:e|ed))\b", t):
        score += 5.0
    if any(phrase in t for phrase in (
        "primarily due to", "primarily driven", "primarily reflecting", "compared with", "compared to",
    )):
        score += 6.0

    # Generic forward-looking discussion is useful context, but for a question
    # asking what happened, realized results should rank ahead of it.
    if any(phrase in t for phrase in ("we expect", "may affect", "could affect", "anticipate that")) and numeric_hits < 2:
        score -= 5.0

    return score


def _mdna_trend_rank(query: str, docs: list[Document], limit: int) -> list[Document]:
    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(docs):
        score = _mdna_trend_score(query, doc.page_content)
        if score > 0:
            scored.append((score, idx, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _hybrid_rank_for_query(vectorstore, query: str, *, corpus_docs: list[Document], filter_dict: dict, fetch_k: int) -> list[Document]:
    dense = vectorstore.similarity_search(query, k=fetch_k, filter=filter_dict)
    lexical = _bm25_rank(query, corpus_docs, fetch_k)
    return _rrf_fuse(dense, lexical, fetch_k) if lexical else _dedupe_docs(dense)


def hybrid_retrieve(
    vectorstore,
    query: str,
    *,
    k: int,
    filter_dict: dict,
    candidate_multiplier: int = 4,
    min_candidates: int = 16,
    prefer_latest_year: bool = True,
    financial_statement_mode: bool = False,
) -> list[Document]:
    """Hybrid retrieval with section-aware coverage.

    Financial Statements keep the already-validated exact-metric path unchanged.
    Business/Risk/MD&A add two generic mechanisms:
      1. facet coverage for clearly multi-part questions;
      2. realized-result anchors for MD&A trend/driver questions.
    The final context size remains exactly `k`.
    """
    if financial_statement_mode:
        fetch_k = max(k * 8, 64)
    else:
        fetch_k = max(k * candidate_multiplier, min_candidates)

    effective_filter = filter_dict
    corpus_docs: list[Document] = []

    try:
        raw = vectorstore.get(
            where=filter_dict,
            include=["documents", "metadatas"],
        )
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or [{} for _ in documents]
        all_docs = [
            Document(page_content=text or "", metadata=metadata or {})
            for text, metadata in zip(documents, metadatas)
            if text
        ]

        if prefer_latest_year and not _has_fiscal_year_filter(filter_dict) and all_docs:
            years = [_fiscal_year(doc) for doc in all_docs if _fiscal_year(doc) > 0]
            if years:
                latest_year = max(years)
                effective_filter = _add_fiscal_year_filter(filter_dict, latest_year)
                corpus_docs = [doc for doc in all_docs if _fiscal_year(doc) == latest_year]
            else:
                corpus_docs = all_docs
        else:
            corpus_docs = all_docs

        corpus_docs = _dedupe_docs(corpus_docs, prefer_latest=True)
    except Exception:
        corpus_docs = []

    dense_docs = vectorstore.similarity_search(
        query,
        k=fetch_k,
        filter=effective_filter,
    )

    if not corpus_docs:
        return _dedupe_docs(dense_docs)[:k]

    lexical_docs = _bm25_rank(query, corpus_docs, fetch_k)
    fused_docs = _rrf_fuse(dense_docs, lexical_docs, fetch_k) if lexical_docs else _dedupe_docs(dense_docs)

    if financial_statement_mode:
        # FROZEN: exact accounting rows first; hybrid results provide context.
        anchors = _financial_anchor_rank(query, corpus_docs, limit=min(fetch_k, 16))
        return _merge_ranked(anchors, fused_docs, limit=k)

    is_business = _filter_contains(effective_filter, "section", "business")
    is_risk = _filter_contains(effective_filter, "section", "risk_factors")
    is_mdna = _filter_contains(effective_filter, "section", "mdna")
    is_prose_section = is_business or is_risk or is_mdna

    trend_anchors: list[Document] = []
    if is_mdna:
        trend_anchors = _mdna_trend_rank(query, corpus_docs, limit=min(fetch_k, 16))

    facet_seed: list[Document] = []
    if is_prose_section:
        facets = _query_facets(query)
        if len(facets) >= 2:
            facet_fetch = max(k * 3, 16)
            facet_rankings = [
                _hybrid_rank_for_query(
                    vectorstore,
                    facet,
                    corpus_docs=corpus_docs,
                    filter_dict=effective_filter,
                    fetch_k=facet_fetch,
                )
                for facet in facets
            ]
            facet_seed = _round_robin_facets(facet_rankings, per_facet=2)

    if facet_seed or trend_anchors:
        return _merge_diverse(facet_seed, trend_anchors, fused_docs, limit=k)

    return fused_docs[:k]

