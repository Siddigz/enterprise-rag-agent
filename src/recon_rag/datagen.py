"""Seeded synthetic sales data spread across five source systems.

Every order exists in the CRM (system of record), the ERP and the warehouse feed for its region. Each
source uses its own column names, ID formats, units, date formats and status vocabularies, and four of
them change schema between their first and second batch. Discrepancies are injected at known rates and
everything injected is written to ``ground_truth.json`` so reconciliation and the eval suite can be
scored against it.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

REGIONS = ["NA", "EMEA", "APAC"]
REGION_CURRENCY = {"NA": "USD", "EMEA": "EUR", "APAC": "AUD"}
ERP_REGION_NAMES = {"NA": "North America", "EMEA": "Europe", "APAC": "Asia Pacific"}
STATUSES = ["pending", "shipped", "delivered", "cancelled", "returned"]
STATUS_WEIGHTS = [0.12, 0.2, 0.55, 0.08, 0.05]
ERP_STATUS = {
    "pending": "OPEN",
    "shipped": "SHIPPED",
    "delivered": "DELIVERED",
    "cancelled": "CANCELLED",
    "returned": "RETURNED",
}
WH_STATUS = {
    "pending": "awaiting",
    "shipped": "in_transit",
    "delivered": "delivered",
    "cancelled": "cancelled",
    "returned": "returned",
}
START = date(2026, 1, 1)
DAYS = 181  # Jan 1 - Jun 30
V2_CUTOVER = date(2026, 4, 1)  # batches on/after this date use the drifted schemas

# Per-order probability of each injected discrepancy (at most one per order)
DISCREPANCY_RATES = {
    "amount_mismatch": 0.04,
    "status_conflict": 0.03,
    "quantity_mismatch": 0.02,
    "missing_in_erp": 0.015,
    "missing_in_warehouse": 0.015,
    "duplicate_in_crm": 0.01,
    "currency_mismatch": 0.01,
}


@dataclass
class Order:
    order_id: str
    customer_id: str
    order_date: str
    region: str
    sku: str
    quantity: int
    unit_price: float
    amount: float
    currency: str
    status: str


def _num(order_id: str) -> int:
    return int(order_id.split("-")[1])


def _make_orders(rng: random.Random, n: int) -> list[Order]:
    customers = [f"C-{i:04d}" for i in range(1, max(50, n // 7) + 1)]
    customer_region = {c: rng.choice(REGIONS) for c in customers}
    skus = [f"SKU-{i:03d}" for i in range(1, 41)]
    sku_price = {s: round(rng.uniform(5, 500), 2) for s in skus}
    orders = []
    for i in range(1, n + 1):
        cust = rng.choice(customers)
        region = customer_region[cust]
        sku = rng.choice(skus)
        qty = rng.randint(1, 20)
        price = sku_price[sku]
        orders.append(
            Order(
                order_id=f"SO-{i:06d}",
                customer_id=cust,
                order_date=(START + timedelta(days=rng.randrange(DAYS))).isoformat(),
                region=region,
                sku=sku,
                quantity=qty,
                unit_price=price,
                amount=round(qty * price, 2),
                currency=REGION_CURRENCY[region],
                status=rng.choices(STATUSES, STATUS_WEIGHTS)[0],
            )
        )
    orders.sort(key=lambda o: (o.order_date, o.order_id))
    return orders


def _pick_discrepancies(rng: random.Random, orders: list[Order]) -> dict[str, dict]:
    injected: dict[str, dict] = {}
    for o in orders:
        roll = rng.random()
        acc = 0.0
        for kind, rate in DISCREPANCY_RATES.items():
            acc += rate
            if roll < acc:
                injected[o.order_id] = {"kind": kind}
                break
    for o in orders:
        d = injected.get(o.order_id)
        if not d:
            continue
        if d["kind"] == "amount_mismatch":
            factor = rng.choice([-1, 1]) * rng.uniform(0.03, 0.25)
            d["erp_amount"] = round(o.amount * (1 + factor), 2)
        elif d["kind"] == "status_conflict":
            d["wh_status"] = rng.choice([s for s in STATUSES if s != o.status])
        elif d["kind"] == "quantity_mismatch":
            d["wh_quantity"] = max(1, o.quantity + rng.choice([-3, -2, -1, 1, 2, 3]))
            if d["wh_quantity"] == o.quantity:
                d["wh_quantity"] = o.quantity + 1
        elif d["kind"] == "currency_mismatch":
            d["erp_currency"] = "USD" if o.currency != "USD" else "CAD"
    return injected


def _ground_truth_entry(o: Order, d: dict) -> dict:
    wh = f"wh_{o.region.lower()}"
    kind = d["kind"]
    if kind == "amount_mismatch":
        return {
            "order_id": o.order_id,
            "kind": "amount_mismatch",
            "sources": ["crm", "erp"],
            "detail": {"crm": o.amount, "erp": d["erp_amount"]},
        }
    if kind == "status_conflict":
        return {
            "order_id": o.order_id,
            "kind": "status_conflict",
            "sources": ["crm", wh],
            "detail": {"crm": o.status, wh: d["wh_status"]},
        }
    if kind == "quantity_mismatch":
        return {
            "order_id": o.order_id,
            "kind": "quantity_mismatch",
            "sources": ["crm", wh],
            "detail": {"crm": o.quantity, wh: d["wh_quantity"]},
        }
    if kind == "currency_mismatch":
        return {
            "order_id": o.order_id,
            "kind": "currency_mismatch",
            "sources": ["crm", "erp"],
            "detail": {"crm": o.currency, "erp": d["erp_currency"]},
        }
    if kind == "missing_in_erp":
        return {"order_id": o.order_id, "kind": "missing_record", "sources": ["erp"], "detail": {"missing_from": "erp"}}
    if kind == "missing_in_warehouse":
        return {"order_id": o.order_id, "kind": "missing_record", "sources": [wh], "detail": {"missing_from": wh}}
    if kind == "duplicate_in_crm":
        return {
            "order_id": o.order_id,
            "kind": "duplicate_record",
            "sources": ["crm"],
            "detail": {"source": "crm", "count": 2},
        }
    raise ValueError(kind)


def _us_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _eu_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def _crm_rows(orders: list[Order], inj: dict[str, dict], rng: random.Random, v2: bool) -> list[dict]:
    reps = ["A. Okafor", "B. Chen", "C. Martin", "D. Singh", "E. Rossi"]
    rows = []
    for o in orders:
        if v2:
            row = {
                "order_id": o.order_id,
                "cust_id": o.customer_id,
                "order_date": _us_date(o.order_date),
                "region": o.region,
                "sku": o.sku,
                "quantity": o.quantity,
                "unit_price": f"{o.unit_price:.2f}",
                "total_amount": f"{o.amount:.2f}",
                "currency": o.currency,
                "status": o.status,
                "sales_rep": rng.choice(reps),
            }
        else:
            row = {
                "order_id": o.order_id,
                "customer_id": o.customer_id,
                "order_date": o.order_date,
                "region": o.region,
                "sku": o.sku,
                "quantity": o.quantity,
                "unit_price": f"{o.unit_price:.2f}",
                "total_amount": f"{o.amount:.2f}",
                "currency": o.currency,
                "status": o.status,
            }
        rows.append(row)
        if inj.get(o.order_id, {}).get("kind") == "duplicate_in_crm":
            rows.append(dict(row))
    return rows


def _erp_rows(orders: list[Order], inj: dict[str, dict], rng: random.Random, v2: bool) -> list[dict]:
    rows = []
    for o in orders:
        d = inj.get(o.order_id, {})
        if d.get("kind") == "missing_in_erp":
            continue
        amount = d.get("erp_amount", o.amount)
        placed = datetime.fromisoformat(o.order_date).replace(hour=rng.randrange(8, 20), minute=rng.randrange(60))
        row = {
            "orderNumber": _num(o.order_id),
            "customerRef": o.customer_id.replace("-", ""),
            "placedAt": placed.isoformat() + "Z",
            "salesRegion": ERP_REGION_NAMES[o.region],
            "itemCode": o.sku,
            "units": o.quantity,
        }
        if v2:
            row["amount"] = f"{amount:.2f}"
        else:
            row["amountCents"] = int(round(amount * 100))
        row["currencyCode"] = d.get("erp_currency", o.currency)
        row["orderStatus"] = ERP_STATUS[o.status]
        rows.append(row)
    return rows


def _wh_rows(orders: list[Order], inj: dict[str, dict], region: str, v2: bool) -> list[dict]:
    rows = []
    for o in orders:
        d = inj.get(o.order_id, {})
        if d.get("kind") == "missing_in_warehouse":
            continue
        qty = d.get("wh_quantity", o.quantity)
        status = d.get("wh_status", o.status)
        value = round(qty * o.unit_price, 2) if "wh_quantity" in d else o.amount
        order_dt = _eu_date(o.order_date) if region == "EMEA" else o.order_date
        row = {
            "ORDER_REF": o.order_id.replace("-", ""),
            "CUSTOMER": o.customer_id,
            "ORDER_DT": order_dt,
            "PRODUCT": o.sku,
            "UNITS": qty,
        }
        if not (region == "APAC" and v2):
            row["UNIT_PRICE"] = f"{o.unit_price:.2f}"
        if region == "EMEA" and v2:
            row["NET_VALUE"] = f"{value:.2f}"
            row["VAT_AMOUNT"] = f"{value * 0.2:.2f}"
        else:
            row["LINE_VALUE"] = f"{value:.2f}"
        row["CCY"] = o.currency
        row["FULFILMENT_STATUS"] = WH_STATUS[status]
        rows.append(row)
    return rows


EXPECTED_DRIFT = [
    {
        "source": "crm",
        "kind": "column_renamed",
        "column": "cust_id",
        "detail": {"from": "customer_id", "field": "customer_id"},
    },
    {
        "source": "crm",
        "kind": "format_changed",
        "column": "order_date",
        "detail": {"from": "%Y-%m-%d", "to": "%m/%d/%Y"},
    },
    {"source": "crm", "kind": "column_added", "column": "sales_rep", "detail": {}},
    {
        "source": "erp",
        "kind": "column_renamed",
        "column": "amount",
        "detail": {"from": "amountCents", "field": "amount"},
    },
    {"source": "erp", "kind": "unit_changed", "column": "amount", "detail": {"from": "cents", "to": "dollars"}},
    {
        "source": "wh_emea",
        "kind": "column_renamed",
        "column": "NET_VALUE",
        "detail": {"from": "LINE_VALUE", "field": "amount"},
    },
    {"source": "wh_emea", "kind": "column_added", "column": "VAT_AMOUNT", "detail": {}},
    {"source": "wh_apac", "kind": "column_removed", "column": "UNIT_PRICE", "detail": {"field": "unit_price"}},
]


def generate(out_dir: Path, n_orders: int = 2000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    orders = _make_orders(rng, n_orders)
    injected = _pick_discrepancies(rng, orders)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    v1 = [o for o in orders if date.fromisoformat(o.order_date) < V2_CUTOVER]
    v2 = [o for o in orders if date.fromisoformat(o.order_date) >= V2_CUTOVER]

    for batch, subset, is_v2 in (("batch_001", v1, False), ("batch_002", v2, True)):
        _write_csv(out_dir / "crm" / f"{batch}.csv", _crm_rows(subset, injected, rng, is_v2))
        _write_json(out_dir / "erp" / f"{batch}.json", _erp_rows(subset, injected, rng, is_v2))
        for region in REGIONS:
            regional = [o for o in subset if o.region == region]
            # NA's warehouse feed never changes schema; EMEA and APAC drift in batch_002
            _write_csv(
                out_dir / f"wh_{region.lower()}" / f"{batch}.csv",
                _wh_rows(regional, injected, region, is_v2 and region != "NA"),
            )

    truth = {
        "seed": seed,
        "n_orders": n_orders,
        "sources": ["crm", "erp", "wh_na", "wh_emea", "wh_apac"],
        "orders": {o.order_id: {k: v for k, v in asdict(o).items() if k != "order_id"} for o in orders},
        "discrepancies": [_ground_truth_entry(o, injected[o.order_id]) for o in orders if o.order_id in injected],
        "drift": EXPECTED_DRIFT,
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def load_ground_truth(data_dir: Path) -> dict:
    return json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic multi-source sales data")
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--orders", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    t = generate(a.out, a.orders, a.seed)
    print(f"wrote {a.orders} orders, {len(t['discrepancies'])} injected discrepancies to {a.out}")
