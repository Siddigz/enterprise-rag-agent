from recon_rag.ingest.loaders import SOURCE_DEFAULTS, discover_batches
from recon_rag.ingest.schema_drift import infer_mapping, normalize_row
from recon_rag.reconcile.engine import Tolerances, reconcile, score_against_truth


def rec(source, **kw):
    base = {"order_id": "SO-000001", "customer_id": "C-0001", "order_date": "2026-01-05", "region": "NA",
            "sku": "SKU-001", "quantity": 2, "unit_price": 10.0, "amount": 20.0, "currency": "USD",
            "status": "delivered"}  # fmt: skip
    return {**base, **kw, "source": source}


def kinds(findings):
    return sorted(f.kind for f in findings)


def test_clean_order_has_no_findings():
    assert reconcile([rec("crm"), rec("erp"), rec("wh_na")]) == []


def test_each_discrepancy_kind():
    assert kinds(reconcile([rec("crm"), rec("erp", amount=25.0), rec("wh_na")])) == ["amount_mismatch"]
    assert kinds(reconcile([rec("crm"), rec("erp", currency="CAD"), rec("wh_na")])) == ["currency_mismatch"]
    assert kinds(reconcile([rec("crm"), rec("erp"), rec("wh_na", status="returned")])) == ["status_conflict"]
    assert kinds(reconcile([rec("crm"), rec("wh_na")])) == ["missing_record"]
    assert kinds(reconcile([rec("crm"), rec("crm"), rec("erp"), rec("wh_na")])) == ["duplicate_record"]


def test_quantity_mismatch_does_not_double_count_amount():
    f = reconcile([rec("crm"), rec("erp"), rec("wh_na", quantity=3, amount=30.0)])
    assert kinds(f) == ["quantity_mismatch"]


def test_amount_tolerance():
    assert reconcile([rec("crm"), rec("erp", amount=20.005), rec("wh_na")], Tolerances(0.01, 0.005)) == []
    f = reconcile([rec("crm", amount=1000.0), rec("erp", amount=1200.0), rec("wh_na", amount=1000.0)])
    assert f[0].severity == "high" and f[0].detail["difference"] == 200.0


def test_erp_is_reference_when_crm_missing():
    f = reconcile([rec("erp"), rec("wh_na", status="pending")])
    assert kinds(f) == ["missing_record", "status_conflict"]
    assert next(x for x in f if x.kind == "status_conflict").sources == ["erp", "wh_na"]


def test_fuzzy_fallback_rekeys_mistyped_ids():
    f = reconcile([rec("crm"), rec("erp"), rec("wh_na", order_id="SO-999999")])
    assert kinds(f) == ["id_mismatch"]
    assert f[0].order_id == "SO-000001"


def test_matches_ground_truth_exactly(dataset):
    data_dir, truth = dataset
    records = []
    for b in discover_batches(data_dir):
        m = infer_mapping(b.source, b.rows)
        records += [{**normalize_row(r, m, SOURCE_DEFAULTS.get(b.source)), "source": b.source} for r in b.rows]
    score = score_against_truth(reconcile(records), truth["discrepancies"])
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
