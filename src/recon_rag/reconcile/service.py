from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from recon_rag.config import get_settings
from recon_rag.models import CanonicalOrder, Discrepancy
from recon_rag.reconcile.engine import Finding, Tolerances, reconcile

FIELDS = ["source", "order_id", "customer_id", "order_date", "region", "sku", "quantity", "unit_price", "amount",
          "currency", "status"]  # fmt: skip


def load_records(session: Session, order_id: str | None = None) -> list[dict]:
    stmt = select(*(getattr(CanonicalOrder, f) for f in FIELDS)).order_by(CanonicalOrder.id)
    if order_id:
        stmt = stmt.where(CanonicalOrder.order_id == order_id)
    rows = session.execute(stmt).all()
    return [
        {
            f: (float(v) if f in ("amount", "unit_price") and v is not None else v)
            for f, v in zip(FIELDS, row, strict=True)
        }
        for row in rows
    ]


def run_reconciliation(session: Session) -> list[Finding]:
    s = get_settings()
    findings = reconcile(load_records(session), Tolerances(s.amount_abs_tolerance, s.amount_rel_tolerance))
    session.execute(delete(Discrepancy))
    if findings:
        session.execute(insert(Discrepancy), [f.to_dict() for f in findings])
    return findings
