"""Turn canonical records, discrepancies and schema metadata into retrievable text chunks.

Chunk IDs are stable (derived from business keys, not row IDs) because they double as the evidence IDs
the agent cites and the eval suite checks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

SOURCE_LABELS = {
    "crm": "CRM",
    "erp": "ERP",
    "wh_na": "Warehouse NA",
    "wh_emea": "Warehouse EMEA",
    "wh_apac": "Warehouse APAC",
}


@dataclass
class Chunk:
    id: str
    kind: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


def _money(v: float | None, ccy: str | None = None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}" + (f" {ccy}" if ccy else "")


def discrepancy_id(order_id: str, kind: str, sources: list[str]) -> str:
    return f"disc:{order_id}:{kind}:{'+'.join(sorted(sources))}"


def describe_discrepancy(d: dict) -> str:
    kind, det, srcs = d["kind"], d["detail"], d["sources"]
    label = lambda s: SOURCE_LABELS.get(s, s)  # noqa: E731
    if kind == "missing_record":
        body = f"the order is missing from {label(det['missing_from'])} ({det['missing_from']})"
    elif kind == "duplicate_record":
        body = f"{label(det['source'])} ({det['source']}) contains {det['count']} copies of the order"
    elif kind == "amount_mismatch":
        a, b = srcs
        body = (
            f"{a} amount {_money(det[a])} vs {b} amount {_money(det[b])}, difference {det['difference']:+.2f} "
            f"({det['pct_difference']:.2f}%)"
        )
    elif kind == "id_mismatch":
        body = f"reported under ID {det['reported_id']}, matched on customer, date and amount"
    else:
        a, b = srcs
        field_name = kind.replace("_mismatch", "").replace("_conflict", "")
        body = f"{a} {field_name} {det[a]} vs {b} {field_name} {det[b]}"
    return (
        f"Discrepancy {kind} (severity {d['severity']}) on order {d['order_id']} in region {d.get('region')} "
        f"between {', '.join(srcs)}: {body}."
    )


def order_chunks(records: list[dict], findings: list[dict]) -> list[Chunk]:
    by_order: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_order[r["order_id"]].append(r)
    f_by_order: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        f_by_order[f["order_id"]].append(f)

    chunks = []
    for oid in sorted(by_order):
        recs = sorted(by_order[oid], key=lambda r: r["source"])
        ref = next((r for r in recs if r["source"] == "crm"), recs[0])
        od = ref.get("order_date")
        lines = [
            f"Order {oid} | customer {ref.get('customer_id')} | region {ref.get('region')} | "
            f"order date {od} | product {ref.get('sku')}."
        ]
        for r in recs:
            lines.append(
                f"{SOURCE_LABELS.get(r['source'], r['source'])} ({r['source']}): quantity {r.get('quantity')}, "
                f"unit price {_money(r.get('unit_price'))}, amount {_money(r.get('amount'), r.get('currency'))}, "
                f"status {r.get('status')}."
            )
        fs = f_by_order.get(oid, [])
        if fs:
            lines.append(
                f"Reconciliation: {len(fs)} discrepancy(ies): " + " ".join(describe_discrepancy(f) for f in fs)
            )
        else:
            lines.append("Reconciliation: all sources agree.")
        chunks.append(
            Chunk(
                id=f"order:{oid}",
                kind="order",
                content="\n".join(lines),
                meta={
                    "order_id": oid,
                    "customer_id": ref.get("customer_id"),
                    "region": ref.get("region"),
                    "month": str(od)[:7] if od else None,
                    "sources": sorted({r["source"] for r in recs}),
                    "discrepancy_kinds": sorted({f["kind"] for f in fs}),
                },
            )
        )
    return chunks


def discrepancy_chunks(findings: list[dict]) -> list[Chunk]:
    return [
        Chunk(
            id=discrepancy_id(f["order_id"], f["kind"], f["sources"]),
            kind="discrepancy",
            content=describe_discrepancy(f),
            meta={
                "order_id": f["order_id"],
                "region": f.get("region"),
                "sources": f["sources"],
                "discrepancy_kind": f["kind"],
                "severity": f["severity"],
            },
        )
        for f in findings
    ]


def describe_drift(e: dict) -> str:
    kind, col, det = e["kind"], e["column"], e["detail"]
    if kind == "column_renamed":
        body = f"column renamed from {det['from']} to {col} (canonical field {det.get('field')}); same data, new name"
    elif kind == "column_added":
        mapped = f"maps to canonical field {det['field']}" if det.get("field") else "not mapped to any canonical field"
        body = f"new column {col} added to the feed ({mapped})"
    elif kind == "column_removed":
        body = (
            f"column {col} removed (dropped, no longer sent); it previously held canonical field "
            f"{det.get('field')}, which is now derived or missing"
        )
    elif kind == "format_changed":
        body = f"date format of column {col} changed from {det['from']} to {det['to']} (format change)"
    elif kind == "unit_changed":
        body = f"unit of column {col} changed from {det['from']} to {det['to']} (unit change)"
    else:
        body = f"{kind} on column {col}: {det}"
    return (
        f"Schema drift in source {e['source']} ({SOURCE_LABELS.get(e['source'], e['source'])}), "
        f"schema v{e.get('from_version')} to v{e['to_version']} in batch {det.get('batch')}: {body}."
    )


def drift_chunks(events: list[dict]) -> list[Chunk]:
    return [
        Chunk(
            id=f"drift:{e['source']}:v{e['to_version']}:{e['kind']}:{e['column']}",
            kind="drift",
            content=describe_drift(e),
            meta={"source": e["source"], "sources": [e["source"]], "drift_kind": e["kind"], "column": e["column"]},
        )
        for e in events
    ]


def schema_chunks(versions: list[dict]) -> list[Chunk]:
    out = []
    for v in versions:
        parts = []
        for col, m in v["mapping"].items():
            target = m.get("field") or "unmapped"
            extra = [x for x in (m.get("date_format"), m.get("unit")) if x]
            parts.append(f"{col} -> {target}" + (f" [{', '.join(extra)}]" if extra else ""))
        out.append(
            Chunk(
                id=f"schema:{v['source']}:v{v['version']}",
                kind="schema",
                content=(
                    f"Schema v{v['version']} of source {v['source']} "
                    f"({SOURCE_LABELS.get(v['source'], v['source'])}) column mapping: " + "; ".join(parts) + "."
                ),
                meta={"source": v["source"], "sources": [v["source"]], "version": v["version"]},
            )
        )
    return out
