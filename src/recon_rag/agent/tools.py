"""Tools the agent can call. Every tool returns evidence items (id + text); only these IDs may be cited.

``query_sales`` is a whitelisted aggregate-query builder: the model picks a metric, a grouping and filters
from fixed enums, and the SQL is assembled from those choices. It never executes model-written SQL.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recon_rag.index.chunker import describe_discrepancy, describe_drift, discrepancy_id
from recon_rag.index.embedder import Embedder
from recon_rag.index.retriever import hybrid_search
from recon_rag.ingest.schema_drift import normalize_order_id
from recon_rag.models import CanonicalOrder, Discrepancy, DriftEvent, SchemaVersion
from recon_rag.reconcile.engine import reconcile
from recon_rag.reconcile.service import load_records

Region = Literal["NA", "EMEA", "APAC"]
Source = Literal["crm", "erp", "wh_na", "wh_emea", "wh_apac"]
DiscKind = Literal[
    "amount_mismatch",
    "quantity_mismatch",
    "currency_mismatch",
    "status_conflict",
    "missing_record",
    "duplicate_record",
    "id_mismatch",
]
Status = Literal["pending", "shipped", "delivered", "cancelled", "returned"]


@dataclass
class Evidence:
    id: str
    text: str


@dataclass
class ToolResult:
    evidence: list[Evidence]
    data: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> str:
        return json.dumps({"evidence": [{"id": e.id, "text": e.text} for e in self.evidence], **self.data}, default=str)


# ------------------------------------------------------------------------------------------------ schemas


class SearchArgs(BaseModel):
    query: str
    kind: Literal["order", "discrepancy", "drift", "schema"] | None = None
    region: Region | None = None
    source: Source | None = None
    month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    top_k: int = Field(8, ge=1, le=15)


class ReconcileArgs(BaseModel):
    order_id: str


class DiscrepancyArgs(BaseModel):
    kind: DiscKind | None = None
    region: Region | None = None
    source: Source | None = None
    order_id: str | None = None
    customer_id: str | None = None
    severity: Literal["low", "medium", "high"] | None = None
    limit: int = Field(20, ge=1, le=50)


class DriftArgs(BaseModel):
    source: Source | None = None


class QueryArgs(BaseModel):
    metric: Literal["order_count", "total_amount", "avg_amount", "total_quantity", "discrepancy_count"]
    group_by: (
        Literal["region", "month", "currency", "status", "customer_id", "sku", "source", "kind", "severity"] | None
    ) = None
    source: Source = "crm"
    region: Region | None = None
    month: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    status: Status | None = None
    kind: DiscKind | None = None
    order_by: Literal["value_desc", "value_asc", "group"] = "value_desc"
    limit: int = Field(25, ge=1, le=50)


def _schema(model: type[BaseModel]) -> dict:
    s = model.model_json_schema()
    s.pop("title", None)
    for p in s.get("properties", {}).values():
        p.pop("title", None)
    return s


TOOL_SPECS: list[dict] = [
    {
        "name": "search_documents",
        "description": (
            "Hybrid semantic + keyword search over indexed evidence: one document per order (all source records "
            "side by side plus its reconciliation result), one per discrepancy, one per schema-drift event and "
            "one per schema version. Use for lookups of specific orders, customers, or descriptive questions. "
            "Do not use search results to count or total things; use query_sales or get_discrepancies instead."
        ),
        "input_schema": _schema(SearchArgs),
    },
    {
        "name": "reconcile_order",
        "description": "Fetch every source system's normalized record for one order and reconcile them live.",
        "input_schema": _schema(ReconcileArgs),
    },
    {
        "name": "get_discrepancies",
        "description": (
            "List detected cross-source discrepancies with optional filters. Returns the exact total count "
            "matching the filters plus up to `limit` items."
        ),
        "input_schema": _schema(DiscrepancyArgs),
    },
    {
        "name": "get_schema_drift",
        "description": "Schema-drift events (renamed/added/removed columns, format and unit changes) and current "
        "column mappings, optionally for one source.",
        "input_schema": _schema(DriftArgs),
    },
    {
        "name": "query_sales",
        "description": (
            "Exact aggregates over normalized sales records from one source system (default crm, the system of "
            "record; duplicates are de-duplicated by order ID) or over discrepancies (metric=discrepancy_count). "
            "Money metrics are always broken down by currency and are never converted. "
            "Months are YYYY-MM. Data covers 2026-01 to 2026-06."
        ),
        "input_schema": _schema(QueryArgs),
    },
]


# ------------------------------------------------------------------------------------------------ executor


class ToolExecutor:
    def __init__(self, session: Session, embedder: Embedder):
        self.session = session
        self.embedder = embedder

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        handlers = {
            "search_documents": (SearchArgs, self.search_documents),
            "reconcile_order": (ReconcileArgs, self.reconcile_order),
            "get_discrepancies": (DiscrepancyArgs, self.get_discrepancies),
            "get_schema_drift": (DriftArgs, self.get_schema_drift),
            "query_sales": (QueryArgs, self.query_sales),
        }
        if name not in handlers:
            raise ToolError(f"unknown tool {name}")
        model, fn = handlers[name]
        try:
            parsed = model.model_validate(args)
        except ValidationError as e:
            raise ToolError(f"invalid arguments: {e.errors(include_url=False)}") from e
        return fn(parsed)

    def search_documents(self, a: SearchArgs) -> ToolResult:
        hits = hybrid_search(self.session, self.embedder, a.query, a.top_k, a.kind, a.region, a.source, a.month)
        return ToolResult([Evidence(h.id, h.content) for h in hits], {"n_results": len(hits)})

    def reconcile_order(self, a: ReconcileArgs) -> ToolResult:
        oid = normalize_order_id(a.order_id)
        records = load_records(self.session, oid) if oid else []
        if not records:
            return ToolResult(
                [
                    Evidence(
                        f"recon:{oid or a.order_id}", f"No records found for order {a.order_id} in any source system."
                    )
                ],
                {"found": False},
            )
        findings = reconcile(records)
        lines = [f"Live reconciliation of order {oid} across {len(records)} source record(s):"]
        for r in records:
            lines.append(
                f"- {r['source']}: customer {r['customer_id']}, region {r['region']}, date {r['order_date']}, "
                f"sku {r['sku']}, quantity {r['quantity']}, unit price {r['unit_price']}, amount {r['amount']} "
                f"{r['currency']}, status {r['status']}"
            )
        lines += [f"- {describe_discrepancy(f.to_dict())}" for f in findings] or ["- All sources agree."]
        return ToolResult(
            [Evidence(f"recon:{oid}", "\n".join(lines))], {"found": True, "n_discrepancies": len(findings)}
        )

    def get_discrepancies(self, a: DiscrepancyArgs) -> ToolResult:
        stmt = select(Discrepancy)
        conds = []
        if a.kind:
            conds.append(Discrepancy.kind == a.kind)
        if a.region:
            conds.append(Discrepancy.region == a.region)
        if a.severity:
            conds.append(Discrepancy.severity == a.severity)
        if a.order_id:
            conds.append(Discrepancy.order_id == normalize_order_id(a.order_id))
        if a.source:
            conds.append(Discrepancy.sources.contains([a.source]))
        if a.customer_id:
            from recon_rag.ingest.schema_drift import normalize_customer_id

            ids = select(CanonicalOrder.order_id).where(
                CanonicalOrder.customer_id == normalize_customer_id(a.customer_id)
            )
            conds.append(Discrepancy.order_id.in_(ids))
        stmt = stmt.where(*conds)
        total = self.session.scalar(select(func.count()).select_from(stmt.subquery()))
        rows = self.session.scalars(stmt.order_by(Discrepancy.order_id, Discrepancy.kind).limit(a.limit)).all()
        filters = a.model_dump(exclude_none=True, exclude={"limit"})
        summary_id = "q:disc:" + _digest(filters)
        summary = f"get_discrepancies(filters={filters or 'none'}): total matching discrepancies = {total}."
        ev = [Evidence(summary_id, summary)]
        for d in rows:
            f = {
                "order_id": d.order_id,
                "kind": d.kind,
                "severity": d.severity,
                "region": d.region,
                "sources": d.sources,
                "detail": d.detail,
            }
            ev.append(Evidence(discrepancy_id(d.order_id, d.kind, d.sources), describe_discrepancy(f)))
        return ToolResult(ev, {"total": total, "returned": len(rows)})

    def get_schema_drift(self, a: DriftArgs) -> ToolResult:
        q = select(DriftEvent).order_by(DriftEvent.source, DriftEvent.id)
        if a.source:
            q = q.where(DriftEvent.source == a.source)
        events = self.session.scalars(q).all()
        ev = []
        for e in events:
            d = {
                "source": e.source,
                "from_version": e.from_version,
                "to_version": e.to_version,
                "kind": e.kind,
                "column": e.column,
                "detail": e.detail,
            }
            ev.append(Evidence(f"drift:{e.source}:v{e.to_version}:{e.kind}:{e.column}", describe_drift(d)))
        vq = select(SchemaVersion.source, func.max(SchemaVersion.version)).group_by(SchemaVersion.source)
        if a.source:
            vq = vq.where(SchemaVersion.source == a.source)
        versions = dict(self.session.execute(vq).all())
        summary = (
            f"get_schema_drift(source={a.source or 'all'}): {len(events)} drift event(s). Current schema versions: "
            + ", ".join(f"{s} v{v}" for s, v in sorted(versions.items()))
            + ". Sources at v1 have had no schema drift."
        )
        return ToolResult([Evidence("q:drift:" + (a.source or "all"), summary)] + ev, {"n_events": len(events)})

    def query_sales(self, a: QueryArgs) -> ToolResult:
        rows = self._discrepancy_agg(a) if a.metric == "discrepancy_count" else self._order_agg(a)
        params = a.model_dump(exclude_none=True)
        desc = ", ".join(f"{k}={v}" for k, v in params.items())
        if rows:
            body = "; ".join(" | ".join(f"{k}={_fmt(v)}" for k, v in r.items()) for r in rows)
        else:
            body = "no matching records"
        text = f"query_sales({desc}) returned {len(rows)} row(s): {body}."
        return ToolResult([Evidence("sql:" + _digest(params), text)], {"rows": rows})

    def _order_agg(self, a: QueryArgs) -> list[dict]:
        # De-duplicate: one row per order ID within the chosen source.
        base = (
            select(CanonicalOrder)
            .where(CanonicalOrder.source == a.source)
            .distinct(CanonicalOrder.order_id)
            .order_by(CanonicalOrder.order_id, CanonicalOrder.id)
            .subquery()
        )
        month_expr = func.to_char(base.c.order_date, "YYYY-MM")
        group_cols = {
            "region": base.c.region,
            "month": month_expr,
            "currency": base.c.currency,
            "status": base.c.status,
            "customer_id": base.c.customer_id,
            "sku": base.c.sku,
            "source": base.c.source,
        }
        if a.group_by in ("kind", "severity"):
            raise ToolError(f"group_by={a.group_by} only applies to metric=discrepancy_count")
        metric = {
            "order_count": func.count(base.c.order_id),
            "total_amount": func.round(func.sum(base.c.amount), 2),
            "avg_amount": func.round(func.avg(base.c.amount), 2),
            "total_quantity": func.sum(base.c.quantity),
        }[a.metric]
        keys = []
        if a.group_by:
            keys.append((a.group_by, group_cols[a.group_by]))
        if a.metric in ("total_amount", "avg_amount") and a.group_by != "currency":
            keys.append(("currency", base.c.currency))
        stmt = select(*(c.label(n) for n, c in keys), metric.label(a.metric))
        if a.region:
            stmt = stmt.where(base.c.region == a.region)
        if a.month:
            stmt = stmt.where(month_expr == a.month)
        if a.status:
            stmt = stmt.where(base.c.status == a.status)
        if keys:
            stmt = stmt.group_by(*(c for _, c in keys))
        stmt = _order(stmt, a, metric, [c for _, c in keys])
        return [dict(r._mapping) for r in self.session.execute(stmt.limit(a.limit))]

    def _discrepancy_agg(self, a: QueryArgs) -> list[dict]:
        month_expr = func.to_char(
            select(func.min(CanonicalOrder.order_date))
            .where(CanonicalOrder.order_id == Discrepancy.order_id)
            .scalar_subquery(),
            "YYYY-MM",
        )
        cols = {
            "kind": Discrepancy.kind,
            "region": Discrepancy.region,
            "severity": Discrepancy.severity,
            "month": month_expr,
        }
        if a.group_by and a.group_by not in cols:
            raise ToolError(f"group_by={a.group_by} is not supported for discrepancy_count")
        metric = func.count(Discrepancy.id)
        keys = [(a.group_by, cols[a.group_by])] if a.group_by else []
        stmt = select(*(c.label(n) for n, c in keys), metric.label("discrepancy_count"))
        if a.kind:
            stmt = stmt.where(Discrepancy.kind == a.kind)
        if a.region:
            stmt = stmt.where(Discrepancy.region == a.region)
        if a.month:
            stmt = stmt.where(month_expr == a.month)
        if keys:
            stmt = stmt.group_by(*(c for _, c in keys))
        stmt = _order(stmt, a, metric, [c for _, c in keys])
        return [dict(r._mapping) for r in self.session.execute(stmt.limit(a.limit))]


class ToolError(Exception):
    pass


def _order(stmt, a: QueryArgs, metric, group_cols: list):
    if not group_cols:
        return stmt
    if a.order_by == "group":
        return stmt.order_by(*group_cols)
    return stmt.order_by(metric.desc() if a.order_by == "value_desc" else metric.asc(), *group_cols)


def _digest(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float) or type(v).__name__ == "Decimal":
        return f"{float(v):.2f}"
    return str(v)
