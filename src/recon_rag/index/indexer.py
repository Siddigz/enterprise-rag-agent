from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from recon_rag.index.chunker import Chunk, discrepancy_chunks, drift_chunks, order_chunks, schema_chunks
from recon_rag.index.embedder import Embedder
from recon_rag.models import Discrepancy, Document, DriftEvent, SchemaVersion
from recon_rag.reconcile.service import load_records

BATCH = 256


def build_chunks(session: Session) -> list[Chunk]:
    records = load_records(session)
    findings = [
        {
            "order_id": d.order_id,
            "kind": d.kind,
            "severity": d.severity,
            "region": d.region,
            "sources": d.sources,
            "detail": d.detail,
        }
        for d in session.scalars(select(Discrepancy).order_by(Discrepancy.id))
    ]
    events = [
        {
            "source": e.source,
            "from_version": e.from_version,
            "to_version": e.to_version,
            "kind": e.kind,
            "column": e.column,
            "detail": e.detail,
        }
        for e in session.scalars(select(DriftEvent).order_by(DriftEvent.id))
    ]
    versions = [
        {"source": v.source, "version": v.version, "mapping": v.mapping}
        for v in session.scalars(select(SchemaVersion).order_by(SchemaVersion.source, SchemaVersion.version))
    ]
    return (
        order_chunks(records, findings) + discrepancy_chunks(findings) + drift_chunks(events) + schema_chunks(versions)
    )


def build_index(session: Session, embedder: Embedder) -> dict[str, int]:
    chunks = build_chunks(session)
    session.execute(delete(Document))
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = embedder.embed_documents([c.content for c in batch])
        session.execute(
            insert(Document),
            [
                {"id": c.id, "kind": c.kind, "content": c.content, "meta": c.meta, "embedding": v}
                for c, v in zip(batch, vectors, strict=True)
            ],
        )
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    return counts
