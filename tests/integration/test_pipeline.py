"""End-to-end against a real Postgres + pgvector. Uses the hash embedder and the scripted LLM so it needs no
model downloads or API key. Run with DATABASE_URL set (docker compose up postgres)."""

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    os.environ.update(EMBEDDING_PROVIDER="hash", LLM_PROVIDER="fake")
    data_dir = tmp_path_factory.mktemp("it") / "raw"
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["REPORTS_DIR"] = str(tmp_path_factory.mktemp("reports"))
    from recon_rag.config import get_settings
    from recon_rag.index.embedder import get_embedder

    get_settings.cache_clear()
    get_embedder.cache_clear()
    from recon_rag import services
    from recon_rag.db import init_db

    init_db()
    return services.run_pipeline(generate_first=True, n_orders=400, seed=5)


def test_pipeline_matches_ground_truth(pipeline):
    assert pipeline["ingest"]["rejected"] == 0
    assert len(pipeline["ingest"]["drift_events"]) == 8
    gt = pipeline["reconcile"]["vs_ground_truth"]
    assert gt["precision"] == 1.0 and gt["recall"] == 1.0
    assert pipeline["index"]["by_kind"]["order"] == 400


def test_tools_against_database(pipeline):
    from recon_rag.agent.tools import ToolExecutor
    from recon_rag.db import session_scope
    from recon_rag.index.embedder import get_embedder

    with session_scope() as s:
        t = ToolExecutor(s, get_embedder())
        hits = t.run("search_documents", {"query": "order SO-000010"}).evidence
        assert hits[0].id == "order:SO-000010"
        agg = t.run("query_sales", {"metric": "order_count", "group_by": "region"})
        assert sum(r["order_count"] for r in agg.data["rows"]) == 400
        disc = t.run("get_discrepancies", {"kind": "missing_record"})
        assert disc.data["total"] > 0 and disc.evidence[0].id.startswith("q:disc:")
        drift = t.run("get_schema_drift", {"source": "crm"})
        assert drift.data["n_events"] == 3
        assert t.run("reconcile_order", {"order_id": "SO-999999"}).data["found"] is False


def test_api_endpoints(pipeline):
    from fastapi.testclient import TestClient

    from recon_rag.api.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["counts"]["canonical_orders"] > 0
        assert c.get("/drift", params={"source": "erp"}).status_code == 200
        r = c.post("/query", json={"question": "What is the status of order SO-000010?"}).json()
        assert r["citations"] and r["grounding"]["ok"]


def test_eval_run_end_to_end(pipeline):
    from recon_rag.eval import runner

    run_id = runner.start_run()
    metrics = runner.execute_run(run_id, sample_size=12)
    assert metrics["n_cases"] >= 10
    assert 0 <= metrics["answer_accuracy"] <= 1
