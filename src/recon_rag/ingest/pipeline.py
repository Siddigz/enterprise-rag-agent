"""Load every batch, register/compare its schema, log drift, and write raw + canonical rows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import insert, select, text
from sqlalchemy.orm import Session

from recon_rag.config import get_settings
from recon_rag.ingest.loaders import SOURCE_DEFAULTS, discover_batches
from recon_rag.ingest.schema_drift import Resolver, SchemaMapping, diff_schemas, infer_mapping, normalize_row
from recon_rag.models import CanonicalOrder, DriftEvent, RawRecord, SchemaVersion

log = logging.getLogger(__name__)

RESET_TABLES = ["documents", "discrepancies", "canonical_orders", "raw_records", "drift_events", "schema_versions"]


@dataclass
class IngestReport:
    batches: int = 0
    raw_records: int = 0
    canonical_records: int = 0
    rejected: int = 0
    schema_versions: dict[str, int] = field(default_factory=dict)
    drift_events: list[dict] = field(default_factory=list)
    needs_review: list[dict] = field(default_factory=list)


def reset(session: Session) -> None:
    session.execute(text(f"TRUNCATE {', '.join(RESET_TABLES)} RESTART IDENTITY CASCADE"))


def _latest_schema(session: Session, source: str) -> tuple[SchemaVersion | None, SchemaMapping | None]:
    sv = session.scalars(
        select(SchemaVersion).where(SchemaVersion.source == source).order_by(SchemaVersion.version.desc()).limit(1)
    ).first()
    if sv is None:
        return None, None
    return sv, SchemaMapping.from_json(source, sv.columns, sv.mapping, sv.profile)


def ingest_directory(
    session: Session, data_dir: Path, resolver: Resolver | None = None, reset_first: bool = True
) -> IngestReport:
    settings = get_settings()
    report = IngestReport()
    if reset_first:
        reset(session)

    for batch in discover_batches(data_dir):
        report.batches += 1
        mapping = infer_mapping(batch.source, batch.rows, auto_accept=settings.drift_auto_accept, resolver=resolver)
        prev_row, prev = _latest_schema(session, batch.source)

        version = 1
        if prev is None:
            session.add(_schema_row(batch.source, 1, mapping))
        else:
            changes = diff_schemas(prev, mapping)
            version = prev_row.version
            if changes:
                version += 1
                session.add(_schema_row(batch.source, version, mapping))
                for ch in changes:
                    ev = {
                        "source": batch.source,
                        "from_version": prev_row.version,
                        "to_version": version,
                        "kind": ch.kind,
                        "column": ch.column,
                        "detail": {**ch.detail, "batch": batch.name},
                    }
                    session.add(DriftEvent(**ev))
                    report.drift_events.append(ev)
                log.info("drift in %s/%s: %d changes -> v%d", batch.source, batch.name, len(changes), version)
        report.schema_versions[batch.source] = version
        report.needs_review += [
            {
                "source": batch.source,
                "batch": batch.name,
                "column": m.column,
                "candidate": m.candidate,
                "confidence": m.confidence,
            }
            for m in mapping.mappings.values()
            if m.method == "needs_review"
        ]

        raw_ids = session.scalars(
            insert(RawRecord).returning(RawRecord.id),
            [{"source": batch.source, "batch": batch.name, "payload": r} for r in batch.rows],
        ).all()
        report.raw_records += len(raw_ids)

        canonical = []
        defaults = SOURCE_DEFAULTS.get(batch.source)
        for raw_id, row in zip(raw_ids, batch.rows, strict=True):
            rec = normalize_row(row, mapping, defaults)
            if not rec["order_id"]:
                report.rejected += 1
                continue
            canonical.append({"source": batch.source, "raw_record_id": raw_id, **rec})
        if canonical:
            session.execute(insert(CanonicalOrder), canonical)
        report.canonical_records += len(canonical)
        session.flush()
    return report


def _schema_row(source: str, version: int, m: SchemaMapping) -> SchemaVersion:
    return SchemaVersion(
        source=source, version=version, columns=m.columns, mapping=m.mapping_json(), profile=m.profile_json()
    )
