"""Build the benchmark question set from generator ground truth.

Every case has a reference answer and `key_facts`: a list of facts, each a list of acceptable spellings.
The LLM judge grades against the reference answer. The offline heuristic judge checks the key facts.
Unanswerable cases expect the agent to abstain.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from recon_rag.index.chunker import SOURCE_LABELS

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
MONTH_NAMES = dict(zip(MONTHS, ["January", "February", "March", "April", "May", "June"], strict=True))


@dataclass
class GoldenCase:
    id: str
    category: str
    question: str
    expected: str
    key_facts: list[list[str]]
    expect_abstain: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _m(v: float) -> list[str]:
    return [f"{v:.2f}", f"{v:,.2f}"]


def _src(s: str) -> list[str]:
    alts = [s, SOURCE_LABELS.get(s, s)]
    if s.startswith("wh_"):
        r = s[3:].upper()
        alts += [f"{r} warehouse", f"warehouse {r}", f"warehouse ({r})"]
    return alts


def build_golden(truth: dict, seed: int = 7) -> list[GoldenCase]:
    rng = random.Random(seed)
    orders: dict[str, dict] = truth["orders"]
    discs: list[dict] = truth["discrepancies"]
    disc_orders = {d["order_id"] for d in discs}
    clean = sorted(o for o in orders if o not in disc_orders)
    cases: list[GoldenCase] = []

    def add(cat: str, q: str, exp: str, facts: list[list[str]], abstain: bool = False) -> None:
        cases.append(GoldenCase(f"{cat}-{sum(c.category == cat for c in cases) + 1:03d}", cat, q, exp, facts, abstain))

    # --- order lookups (30)
    for i, oid in enumerate(rng.sample(clean, 30)):
        o = orders[oid]
        t = i % 4
        if t == 0:
            add(
                "order_lookup",
                f"What is the total amount of order {oid}, and in which currency?",
                f"{o['amount']:.2f} {o['currency']}",
                [_m(o["amount"]), [o["currency"]]],
            )
        elif t == 1:
            add("order_lookup", f"What is the current status of order {oid}?", o["status"], [[o["status"]]])
        elif t == 2:
            add(
                "order_lookup",
                f"Which customer placed order {oid}, and in which region?",
                f"{o['customer_id']} in {o['region']}",
                [[o["customer_id"]], [o["region"]]],
            )
        else:
            add(
                "order_lookup",
                f"What product and how many units were ordered in {oid}?",
                f"{o['quantity']} units of {o['sku']}",
                [[str(o["quantity"])], [o["sku"]]],
            )

    # --- discrepancy lookups (40 injected + 10 clean)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for d in discs:
        by_kind[d["kind"]].append(d)
    quota = {
        "amount_mismatch": 10,
        "status_conflict": 8,
        "quantity_mismatch": 6,
        "missing_record": 8,
        "currency_mismatch": 4,
        "duplicate_record": 4,
    }
    for kind, n in quota.items():
        for d in rng.sample(by_kind[kind], min(n, len(by_kind[kind]))):
            oid, det, srcs = d["order_id"], d["detail"], d["sources"]
            q = f"Do the source systems agree on order {oid}? If not, describe the discrepancy."
            if kind == "missing_record":
                s = det["missing_from"]
                add(
                    "discrepancy",
                    q,
                    f"No: order {oid} is missing from {s}.",
                    [["missing", "not present", "absent"], _src(s)],
                )
            elif kind == "duplicate_record":
                add(
                    "discrepancy",
                    q,
                    f"No: the CRM contains a duplicate record ({det['count']} copies) for {oid}.",
                    [["duplicate", "copies", "twice"], _src("crm")],
                )
            elif kind == "amount_mismatch":
                a, b = srcs
                add(
                    "discrepancy",
                    q,
                    f"No: amount mismatch, {a} {det[a]:.2f} vs {b} {det[b]:.2f}.",
                    [["amount"], _m(det[a]), _m(det[b])],
                )
            else:
                a, b = srcs
                label = kind.split("_")[0]
                add(
                    "discrepancy",
                    q,
                    f"No: {label} mismatch, {a} has {det[a]} but {b} has {det[b]}.",
                    [[label], [str(det[a])], [str(det[b])]],
                )
    for oid in rng.sample(clean, 10):
        add(
            "discrepancy",
            f"Do the source systems agree on order {oid}? If not, describe the discrepancy.",
            "Yes, all source systems agree on this order.",
            [["agree", "consistent", "no discrepanc", "match"]],
        )

    # --- aggregates (25)
    kind_counts = Counter(d["kind"] for d in discs)
    for kind, n in sorted(kind_counts.items()):
        add("aggregate", f"How many {kind.replace('_', ' ')} discrepancies were detected in total?", str(n), [[str(n)]])
    region_kind = Counter((d["kind"], orders[d["order_id"]]["region"]) for d in discs)
    for kind, region in rng.sample(sorted(region_kind), 6):
        n = region_kind[(kind, region)]
        add(
            "aggregate",
            f"How many {kind.replace('_', ' ')} discrepancies were found in the {region} region?",
            str(n),
            [[str(n)]],
        )
    rm_count: Counter = Counter()
    rm_sum: defaultdict[tuple, float] = defaultdict(float)
    for o in orders.values():
        k = (o["region"], o["order_date"][:7])
        rm_count[k] += 1
        rm_sum[k] += o["amount"]
    keys = rng.sample(sorted(rm_count), 10)
    for region, month in keys[:5]:
        n = rm_count[(region, month)]
        add(
            "aggregate",
            f"How many orders were placed in {region} in {MONTH_NAMES[month]} 2026 according to the CRM?",
            str(n),
            [[str(n)]],
        )
    for region, month in keys[5:]:
        total = round(rm_sum[(region, month)], 2)
        ccy = {"NA": "USD", "EMEA": "EUR", "APAC": "AUD"}[region]
        add(
            "aggregate",
            f"What was the total CRM order amount for {region} in {MONTH_NAMES[month]} 2026?",
            f"{total:.2f} {ccy}",
            [_m(total)],
        )
    am_by_region = Counter(orders[d["order_id"]]["region"] for d in by_kind["amount_mismatch"])
    top = am_by_region.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        add(
            "aggregate",
            "Which region has the most amount mismatches between systems?",
            f"{top[0][0]} ({top[0][1]})",
            [[top[0][0]]],
        )

    # --- schema drift (12)
    add(
        "drift",
        "What schema changes were detected in the CRM feed?",
        "customer_id was renamed to cust_id, order_date changed from YYYY-MM-DD to MM/DD/YYYY, and a sales_rep "
        "column was added.",
        [["cust_id"], ["sales_rep"], ["%m/%d/%Y", "MM/DD/YYYY", "date format"]],
    )
    add(
        "drift",
        "What changed in the ERP export's schema?",
        "amountCents was replaced by amount, and the unit changed from cents to dollars.",
        [["amountCents"], ["dollars"], ["cents"]],
    )
    add(
        "drift",
        "What schema changes happened in the EMEA warehouse feed?",
        "LINE_VALUE was renamed to NET_VALUE and a VAT_AMOUNT column was added.",
        [["NET_VALUE"], ["LINE_VALUE"], ["VAT_AMOUNT"]],
    )
    add(
        "drift",
        "Which source dropped a column, and which column was it?",
        "wh_apac dropped UNIT_PRICE.",
        [_src("wh_apac") + ["APAC"], ["UNIT_PRICE", "unit price", "unit_price"]],
    )
    add(
        "drift",
        "Has the schema of the NA warehouse feed changed?",
        "No, wh_na is still on schema v1 with no drift.",
        [["no ", "not changed", "unchanged", "no drift", "no schema", "v1"]],
    )
    add("drift", "Which column did the CRM rename the customer identifier to?", "cust_id", [["cust_id"]])
    add(
        "drift",
        "How did the date format of the CRM order_date column change?",
        "From %Y-%m-%d (ISO) to %m/%d/%Y (US).",
        [["%Y-%m-%d", "YYYY-MM-DD", "ISO"], ["%m/%d/%Y", "MM/DD/YYYY"]],
    )
    add(
        "drift",
        "Which source changed the unit of its amount field, and from what to what?",
        "The ERP changed amount from cents to dollars.",
        [_src("erp"), ["cents"], ["dollars"]],
    )
    add(
        "drift",
        "Which sources added new columns, and what were they?",
        "crm added sales_rep; wh_emea added VAT_AMOUNT.",
        [["sales_rep"], ["VAT_AMOUNT"]],
    )
    add(
        "drift",
        "Is the VAT_AMOUNT column mapped to any canonical field?",
        "No, VAT_AMOUNT is unmapped.",
        [["not mapped", "unmapped", "no canonical", "none", "no "]],
    )
    add("drift", "Which canonical field does the EMEA warehouse's NET_VALUE column map to?", "amount", [["amount"]])
    add(
        "drift",
        "How many schema drift events were detected in total?",
        str(len(truth["drift"])),
        [[str(len(truth["drift"]))]],
    )

    # --- multi-hop (15)
    for d in rng.sample(by_kind["amount_mismatch"], 8):
        o = orders[d["order_id"]]
        diff = round(d["detail"]["erp"] - d["detail"]["crm"], 2)
        add(
            "multi_hop",
            f"Order {d['order_id']} has an amount discrepancy. Which customer placed it, and what is "
            "the difference between the ERP and CRM amounts?",
            f"{o['customer_id']}; ERP minus CRM = {diff:+.2f}",
            [[o["customer_id"]], [f"{abs(diff):.2f}", f"{abs(diff):,.2f}"]],
        )
    cust_disc = Counter(orders[d["order_id"]]["customer_id"] for d in discs)
    for cid, n in rng.sample(sorted(cust_disc.items()), 7):
        add("multi_hop", f"How many discrepancies involve orders placed by customer {cid}?", str(n), [[str(n)]])

    # --- unanswerable (20)
    n = truth["n_orders"]
    for i in range(6):
        ghost = f"SO-{n + 1000 + i * 37:06d}"
        add(
            "unanswerable",
            rng.choice(
                [
                    f"What is the status of order {ghost}?",
                    f"Do the systems agree on order {ghost}?",
                    f"Who placed order {ghost}?",
                ]
            ),
            "No such order exists in the data.",
            [],
            abstain=True,
        )
    for oid in rng.sample(clean, 8):
        q = rng.choice(
            [
                f"What discount code was applied to order {oid}?",
                f"Which shipping carrier delivered order {oid}?",
                f"Who approved the pricing on order {oid}?",
                f"What was the gross margin on order {oid}?",
            ]
        )
        add("unanswerable", q, "The data does not contain this field.", [], abstain=True)
    for q in [
        "What were total LATAM sales in March 2026?",
        "How many orders were placed in December 2025?",
        "What will EMEA revenue be in Q3 2026?",
        "What is the total order amount across all regions converted to USD?",
        "Which sales rep in the ERP handled the most orders?",
        "What was the customer satisfaction score for APAC?",
    ]:
        add("unanswerable", q, "Not answerable from the available data.", [], abstain=True)
    return cases
