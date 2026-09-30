from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from recon_rag.config import get_settings

JSONType = JSONB


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    """A registered schema for one source feed. A new version is recorded whenever drift is detected."""

    __tablename__ = "schema_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    columns: Mapped[list] = mapped_column(JSONType)
    mapping: Mapped[dict] = mapped_column(JSONType)  # source column -> {field, confidence, method, transform}
    profile: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    from_version: Mapped[int | None] = mapped_column(Integer)
    to_version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))  # column_renamed | column_added | column_removed | format_changed ...
    column: Mapped[str] = mapped_column(String(128))
    detail: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RawRecord(Base):
    __tablename__ = "raw_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    batch: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSONType)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CanonicalOrder(Base):
    __tablename__ = "canonical_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    raw_record_id: Mapped[int] = mapped_column(ForeignKey("raw_records.id", ondelete="CASCADE"))
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    customer_id: Mapped[str | None] = mapped_column(String(32), index=True)
    order_date: Mapped[date | None] = mapped_column(Date)
    region: Mapped[str | None] = mapped_column(String(16), index=True)
    sku: Mapped[str | None] = mapped_column(String(32))
    quantity: Mapped[int | None] = mapped_column(Integer)
    unit_price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    amount: Mapped[float | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    status: Mapped[str | None] = mapped_column(String(16))


class Discrepancy(Base):
    __tablename__ = "discrepancies"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(8))
    region: Mapped[str | None] = mapped_column(String(16), index=True)
    sources: Mapped[list] = mapped_column(JSONType)
    detail: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    """A retrievable chunk. `id` doubles as the evidence ID the agent must cite."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # order | discrepancy | drift | schema
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column("metadata", JSONType)
    embedding = mapped_column(Vector(get_settings().embedding_dim))
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))

    __table_args__ = (
        Index(
            "ix_documents_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_documents_tsv", "tsv", postgresql_using="gin"),
        Index("ix_documents_metadata", "metadata", postgresql_using="gin"),
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | completed | failed
    agent_model: Mapped[str] = mapped_column(String(64))
    judge_model: Mapped[str] = mapped_column(String(64))
    n_cases: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict | None] = mapped_column(JSONType)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(32))
    question: Mapped[str] = mapped_column(Text)
    expected: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[list] = mapped_column(JSONType)
    abstained: Mapped[bool] = mapped_column(Boolean)
    grounded: Mapped[bool] = mapped_column(Boolean)
    correct: Mapped[bool] = mapped_column(Boolean)
    verdict: Mapped[dict] = mapped_column(JSONType)
    latency_ms: Mapped[float] = mapped_column(Float)
