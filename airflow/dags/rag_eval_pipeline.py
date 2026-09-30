"""Daily RAG pipeline + LLM-as-a-judge benchmark with a grounding quality gate.

generate_data → ingest → detect_drift → reconcile → index → run_eval → wait_for_eval → publish_report → quality_gate

Airflow only orchestrates: every stage is an HTTP call to the recon-rag API, so the Airflow image needs
no project dependencies. The eval runs asynchronously in the API and is awaited by a rescheduling sensor.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.sensors.base import PokeReturnValue

API = os.environ.get("RECON_API_URL", "http://api:8000")
REPORTS = Path("/opt/airflow/reports")
log = logging.getLogger(__name__)


def _call(method: str, path: str, timeout: int = 1800, **kw) -> dict | str:
    r = requests.request(method, f"{API}{path}", timeout=timeout, **kw)
    r.raise_for_status()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


@dag(
    dag_id="rag_eval_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "regenerate": Param(True, type="boolean", description="Regenerate the synthetic source data first"),
        "n_orders": Param(2000, type="integer", minimum=50),
        "sample_size": Param(
            int(os.environ["EVAL_SAMPLE_SIZE"]) if os.environ.get("EVAL_SAMPLE_SIZE") else None,
            type=["null", "integer"],
            description="Number of golden cases to evaluate (null = all)",
        ),
    },
    tags=["rag", "eval"],
)
def rag_eval_pipeline():
    @task
    def generate_data(params: dict | None = None) -> dict:
        if not params["regenerate"]:
            return {"skipped": True}
        return _call("POST", "/datasets/generate", json={"n_orders": params["n_orders"]})

    @task
    def ingest(_: dict) -> dict:
        report = _call("POST", "/ingest", json={})
        for col in report.get("needs_review", []):
            log.warning("column needs review: %s", col)
        return {k: report[k] for k in ("batches", "raw_records", "canonical_records", "rejected", "schema_versions")}

    @task
    def detect_drift(_: dict) -> int:
        events = _call("GET", "/drift")
        for e in events:
            log.info(
                "drift %s v%s->v%s %s %s %s",
                e["source"],
                e["from_version"],
                e["to_version"],
                e["kind"],
                e["column"],
                e["detail"],
            )
        return len(events)

    @task
    def reconcile(_: int) -> dict:
        out = _call("POST", "/reconcile")
        gt = out.get("vs_ground_truth")
        if gt and gt["recall"] < 0.99:
            raise AirflowFailException(f"reconciliation recall {gt['recall']} below 0.99: {gt['by_kind']}")
        return out

    @task
    def index(_: dict) -> dict:
        return _call("POST", "/index")

    @task
    def run_eval(_: dict, params: dict | None = None) -> int:
        return _call("POST", "/eval/runs", json={"sample_size": params["sample_size"]})["run_id"]

    @task.sensor(poke_interval=30, timeout=4 * 3600, mode="reschedule")
    def wait_for_eval(run_id: int) -> PokeReturnValue:
        run = _call("GET", f"/eval/runs/{run_id}", timeout=60)
        log.info("eval run %s: %s (%s/%s cases)", run_id, run["status"], run.get("completed_cases"), run["n_cases"])
        if run["status"] == "failed":
            raise AirflowFailException(f"eval run {run_id} failed: {run['error']}")
        return PokeReturnValue(is_done=run["status"] == "completed", xcom_value=run_id)

    @task
    def publish_report(run_id: int) -> int:
        md = _call("GET", f"/eval/runs/{run_id}/report", timeout=60)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"eval_run_{run_id}.md").write_text(md, encoding="utf-8")
        log.info("\n%s", md)
        return run_id

    @task
    def quality_gate(run_id: int) -> dict:
        # Thresholds (GROUNDING_THRESHOLD, ANSWER_ACCURACY_THRESHOLD) are applied by the API when the run finishes
        run = _call("GET", f"/eval/runs/{run_id}", timeout=60)
        m = run["metrics"]
        summary = (
            f"grounding={m['grounding_accuracy']} answer_accuracy={m['answer_accuracy']} "
            f"abstention={m['abstention_rate']} hallucination={m['hallucination_rate']}"
        )
        log.info(summary)
        if not run["passed"]:
            raise AirflowFailException(f"eval run {run_id} failed the quality gate: {summary}")
        return m

    run_id = run_eval(index(reconcile(detect_drift(ingest(generate_data())))))
    quality_gate(publish_report(wait_for_eval(run_id)))


rag_eval_pipeline()
