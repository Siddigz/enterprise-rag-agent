from datetime import date

from recon_rag.ingest.loaders import SOURCE_DEFAULTS, discover_batches
from recon_rag.ingest.schema_drift import (
    detect_date_format,
    diff_schemas,
    infer_mapping,
    normalize_customer_id,
    normalize_order_id,
    normalize_row,
    profile_column,
)


def _mappings(dataset):
    data_dir, _ = dataset
    out = {}
    for b in discover_batches(data_dir):
        out.setdefault(b.source, []).append((b, infer_mapping(b.source, b.rows)))
    return out


def test_every_canonical_field_is_mapped_for_each_source(dataset):
    for source, batches in _mappings(dataset).items():
        for batch, m in batches:
            fields = set(m.by_field())
            expected = {"order_id", "customer_id", "order_date", "sku", "quantity", "amount", "currency", "status"}
            assert expected <= fields, (source, batch.name, expected - fields)


def test_detects_exactly_the_injected_drift(dataset):
    _, truth = dataset
    found = set()
    for source, batches in _mappings(dataset).items():
        for (_, prev), (_, new) in zip(batches, batches[1:], strict=False):
            found |= {(source, c.kind, c.column) for c in diff_schemas(prev, new)}
    expected = {(d["source"], d["kind"], d["column"]) for d in truth["drift"]}
    assert found == expected


def test_erp_cents_are_converted_and_ids_normalized(dataset):
    _, truth = dataset
    batches = _mappings(dataset)["erp"]
    batch, m = batches[0]
    assert m.mappings["amountCents"].unit == "cents"
    row = normalize_row(batch.rows[0], m)
    expected = truth["orders"][row["order_id"]]
    assert row["amount"] == expected["amount"]
    assert row["region"] == expected["region"]  # "North America" -> NA
    assert row["status"] == expected["status"]  # "DELIVERED" -> delivered


def test_warehouse_defaults_and_derived_unit_price(dataset):
    batch, m = _mappings(dataset)["wh_apac"][1]  # v2 dropped UNIT_PRICE
    assert "UNIT_PRICE" not in m.mappings
    row = normalize_row(batch.rows[0], m, SOURCE_DEFAULTS["wh_apac"])
    assert row["region"] == "APAC"
    assert row["unit_price"] == round(row["amount"] / row["quantity"], 2)


def test_ambiguous_date_format_uses_unambiguous_values():
    assert detect_date_format(["03/04/2026", "25/04/2026"]) == "%d/%m/%Y"
    assert detect_date_format(["03/04/2026", "04/25/2026"]) == "%m/%d/%Y"
    assert detect_date_format(["2026-04-25"]) == "%Y-%m-%d"
    assert detect_date_format(["hello"]) is None


def test_id_normalization():
    assert normalize_order_id("SO000123") == "SO-000123"
    assert normalize_order_id(123) == "SO-000123"
    assert normalize_order_id("so-000123") == "SO-000123"
    assert normalize_customer_id("C0042") == "C-0042"
    assert normalize_order_id("") is None


def test_profile_kinds():
    assert profile_column("x", ["SO-000001", "SO-000002"]).kind == "order_id"
    assert profile_column("x", ["USD", "EUR"]).kind == "currency"
    assert profile_column("amountCents", ["12345", "99"]).unit == "cents"
    assert profile_column("x", ["12.50", "3.00"]).kind == "money"
    assert profile_column("x", ["OPEN", "SHIPPED"]).kind == "status"


def _rows():
    return [
        {"order_ref": f"SO-{i:06d}", "client": f"C-{i:04d}", "when": "2026-01-02", "qty": str(i % 5 + 1),
         "grand_total": f"{i * 10:.2f}", "ccy": "USD", "state": "open", "tax_amt": f"{i:.2f}"}
        for i in range(1, 30)
    ]  # fmt: skip


def test_low_confidence_columns_are_left_for_review_without_resolver():
    m = infer_mapping("new_feed", _rows())
    assert m.mappings["order_ref"].field == "order_id"
    assert m.mappings["tax_amt"].field is None


def test_resolver_decides_ambiguous_columns():
    calls = []

    def resolver(source, column, profile, candidates):
        calls.append(column)
        return ("order_date", 0.9) if column == "when" else (None, 0.8)

    m = infer_mapping("new_feed", _rows(), resolver=resolver)
    assert m.mappings["grand_total"].field == "amount"  # confident enough without the resolver
    assert sorted(calls) == ["tax_amt", "when"]
    assert m.mappings["when"].field == "order_date"
    assert m.mappings["when"].method == "llm"
    assert m.mappings["tax_amt"].field is None


def test_normalize_row_parses_us_dates(dataset):
    batch, m = _mappings(dataset)["crm"][1]
    assert m.mappings["order_date"].date_format == "%m/%d/%Y"
    row = normalize_row(batch.rows[0], m)
    assert isinstance(row["order_date"], date)
    assert row["order_date"] >= date(2026, 4, 1)
