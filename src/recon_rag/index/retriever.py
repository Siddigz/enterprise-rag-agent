"""Hybrid retrieval: pgvector cosine search + Postgres full-text search + exact business-key lookup,
fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from recon_rag.config import get_settings
from recon_rag.index.embedder import Embedder

ORDER_ID_RE = re.compile(r"\bSO-?\d{1,8}\b", re.I)
CUSTOMER_ID_RE = re.compile(r"\bC-?\d{3,6}\b", re.I)


@dataclass
class Hit:
    id: str
    kind: str
    content: str
    meta: dict
    score: float


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def or_tsquery(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 1]
    return " | ".join(dict.fromkeys(w.lower() for w in words))


def _filters(kind: str | None, region: str | None, source: str | None, month: str | None) -> tuple[str, dict]:
    clauses, params = [], {}
    if kind:
        clauses.append("kind = :kind")
        params["kind"] = kind
    if region:
        clauses.append("metadata->>'region' = :region")
        params["region"] = region.upper()
    if source:
        clauses.append("metadata->'sources' ? :source")
        params["source"] = source
    if month:
        clauses.append("metadata->>'month' = :month")
        params["month"] = month
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def extract_keys(query: str) -> tuple[list[str], list[str]]:
    from recon_rag.ingest.schema_drift import normalize_customer_id, normalize_order_id

    orders = [normalize_order_id(m) for m in ORDER_ID_RE.findall(query)]
    customers = [normalize_customer_id(m) for m in CUSTOMER_ID_RE.findall(query)]
    return [o for o in orders if o], [c for c in customers if c]


def hybrid_search(
    session: Session,
    embedder: Embedder,
    query: str,
    top_k: int | None = None,
    kind: str | None = None,
    region: str | None = None,
    source: str | None = None,
    month: str | None = None,
) -> list[Hit]:
    s = get_settings()
    top_k = top_k or s.retrieval_top_k
    pool = max(top_k * 4, 20)
    where, params = _filters(kind, region, source, month)

    qvec = embedder.embed_query(query)
    vec_ids = (
        session.execute(
            text(
                f"SELECT id FROM documents WHERE embedding IS NOT NULL{where} "
                f"ORDER BY embedding <=> CAST(:qvec AS vector) LIMIT :pool"
            ),
            {**params, "qvec": str(qvec), "pool": pool},
        )
        .scalars()
        .all()
    )

    fts_ids: list[str] = []
    tsq = or_tsquery(query)
    if tsq:
        fts_ids = (
            session.execute(
                text(
                    f"SELECT id FROM documents WHERE tsv @@ to_tsquery('english', :tsq){where} "
                    # normalization 2|32: divide by document length, then scale to 0..1, so short focused
                    # chunks (drift events, discrepancies) are not drowned out by long order summaries
                    f"ORDER BY ts_rank_cd(tsv, to_tsquery('english', :tsq), 2|32) DESC LIMIT :pool"
                ),
                {**params, "tsq": tsq, "pool": pool},
            )
            .scalars()
            .all()
        )

    key_ids: list[str] = []
    orders, customers = extract_keys(query)
    if orders or customers:
        stmt = text(
            f"SELECT id FROM documents WHERE (metadata->>'order_id' IN :orders "
            f"OR metadata->>'customer_id' IN :customers){where} ORDER BY kind, id LIMIT :pool"
        ).bindparams(bindparam("orders", expanding=True), bindparam("customers", expanding=True))
        key_ids = (
            session.execute(stmt, {**params, "orders": orders or ["-"], "customers": customers or ["-"], "pool": pool})
            .scalars()
            .all()
        )

    # Exact key matches are listed twice so an explicitly named order always outranks fuzzy neighbours.
    fused = rrf([key_ids, key_ids, vec_ids, fts_ids], k=s.rrf_k)[:top_k]
    if not fused:
        return []
    ids = [d for d, _ in fused]
    rows = session.execute(
        text("SELECT id, kind, content, metadata FROM documents WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": ids},
    ).all()
    by_id = {r.id: r for r in rows}
    return [Hit(d, by_id[d].kind, by_id[d].content, by_id[d].metadata, round(sc, 6)) for d, sc in fused if d in by_id]
