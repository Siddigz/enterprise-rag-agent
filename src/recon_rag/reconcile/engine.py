"""Cross-source reconciliation over canonical order records.

The CRM is treated as the system of record. Every order is expected in the CRM, the ERP and the warehouse
feed for its region. Records are matched on normalized order ID. Records whose ID matches nothing are
matched on (customer, date, amount) as a fallback. Each matched source is then compared field by field
against the reference within configurable tolerances.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

REFERENCE_ORDER = ["crm", "erp"]


@dataclass
class Tolerances:
    amount_abs: float = 0.01
    amount_rel: float = 0.005


@dataclass
class Finding:
    order_id: str
    kind: str
    severity: str
    region: str | None
    sources: list[str]
    detail: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def expected_sources(region: str | None) -> list[str]:
    return ["crm", "erp"] + ([f"wh_{region.lower()}"] if region else [])


def _amount_differs(a: float, b: float, tol: Tolerances) -> bool:
    diff = abs(a - b)
    return diff > tol.amount_abs and diff > tol.amount_rel * max(abs(a), abs(b))


def _fuzzy_key(r: dict) -> tuple | None:
    if r.get("customer_id") and r.get("order_date") and r.get("amount") is not None:
        return (r["customer_id"], str(r["order_date"]), round(float(r["amount"]), 2))
    return None


def reconcile(records: Iterable[dict[str, Any]], tol: Tolerances | None = None) -> list[Finding]:
    tol = tol or Tolerances()
    by_order: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_order[r["order_id"]][r["source"]].append(r)

    # Fuzzy fallback: an ID with no CRM/ERP record at all is re-keyed onto a CRM order with the same
    # customer, date and amount (e.g. a warehouse that mistyped the order reference).
    crm_index = {}
    for oid, srcs in by_order.items():
        for r in srcs.get("crm", []):
            if (k := _fuzzy_key(r)) is not None:
                crm_index[k] = oid
    findings: list[Finding] = []
    for oid in [o for o, s in by_order.items() if not any(x in s for x in REFERENCE_ORDER)]:
        srcs = by_order[oid]
        sample = next(iter(srcs.values()))[0]
        target = crm_index.get(_fuzzy_key(sample)) if _fuzzy_key(sample) else None
        if target:
            for src, recs in by_order.pop(oid).items():
                by_order[target][src].extend(recs)
            findings.append(
                Finding(target, "id_mismatch", "medium", sample.get("region"), sorted(srcs),
                        {"reported_id": oid, "matched_on": ["customer_id", "order_date", "amount"]})
            )  # fmt: skip

    for oid in sorted(by_order):
        srcs = by_order[oid]
        ref_src = next((s for s in REFERENCE_ORDER if s in srcs), None)
        ref = srcs[ref_src][0] if ref_src else next(iter(srcs.values()))[0]
        region = ref.get("region")

        for src, recs in sorted(srcs.items()):
            if len(recs) > 1:
                findings.append(
                    Finding(oid, "duplicate_record", "low", region, [src], {"source": src, "count": len(recs)})
                )

        for src in expected_sources(region):
            if src not in srcs:
                findings.append(Finding(oid, "missing_record", "high", region, [src], {"missing_from": src}))

        if ref_src is None:
            continue
        for src, recs in sorted(srcs.items()):
            if src == ref_src:
                continue
            findings.extend(_compare(oid, region, ref_src, ref, src, recs[0], tol))
    return findings


def _compare(
    oid: str, region: str | None, ref_src: str, ref: dict, src: str, rec: dict, tol: Tolerances
) -> list[Finding]:
    out: list[Finding] = []
    qty_mismatch = (
        ref.get("quantity") is not None and rec.get("quantity") is not None and ref["quantity"] != rec["quantity"]
    )
    if qty_mismatch:
        out.append(Finding(oid, "quantity_mismatch", "medium", region, [ref_src, src],
                           {ref_src: ref["quantity"], src: rec["quantity"]}))  # fmt: skip
    a, b = ref.get("amount"), rec.get("amount")
    # A quantity mismatch already explains a different line value; don't double count it.
    if a is not None and b is not None and not qty_mismatch and _amount_differs(float(a), float(b), tol):
        a, b = float(a), float(b)
        pct = abs(a - b) / max(abs(a), 1e-9)
        out.append(Finding(oid, "amount_mismatch", "high" if pct > 0.1 else "medium", region, [ref_src, src],
                           {ref_src: round(a, 2), src: round(b, 2), "difference": round(b - a, 2),
                            "pct_difference": round(pct * 100, 2)}))  # fmt: skip
    if ref.get("currency") and rec.get("currency") and ref["currency"] != rec["currency"]:
        out.append(Finding(oid, "currency_mismatch", "high", region, [ref_src, src],
                           {ref_src: ref["currency"], src: rec["currency"]}))  # fmt: skip
    if ref.get("status") and rec.get("status") and ref["status"] != rec["status"]:
        out.append(Finding(oid, "status_conflict", "medium", region, [ref_src, src],
                           {ref_src: ref["status"], src: rec["status"]}))  # fmt: skip
    return out


def score_against_truth(findings: list[Finding], truth_discrepancies: list[dict]) -> dict:
    """Precision/recall of detected discrepancies against injected ground truth, keyed on (order, kind, sources)."""

    def key(order_id: str, kind: str, sources: list[str]) -> tuple:
        return (order_id, kind, tuple(sorted(sources)))

    got = {key(f.order_id, f.kind, f.sources) for f in findings}
    exp = {key(d["order_id"], d["kind"], d["sources"]) for d in truth_discrepancies}
    tp = len(got & exp)
    by_kind: dict[str, dict] = {}
    for kind in sorted({k[1] for k in got | exp}):
        g = {k for k in got if k[1] == kind}
        e = {k for k in exp if k[1] == kind}
        by_kind[kind] = {"expected": len(e), "detected": len(g), "true_positives": len(g & e)}
    return {
        "expected": len(exp),
        "detected": len(got),
        "true_positives": tp,
        "precision": round(tp / len(got), 4) if got else 1.0,
        "recall": round(tp / len(exp), 4) if exp else 1.0,
        "by_kind": by_kind,
    }
