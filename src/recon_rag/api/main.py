from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from recon_rag import __version__, services
from recon_rag.agent.loop import Agent
from recon_rag.agent.tools import ToolExecutor
from recon_rag.config import get_settings
from recon_rag.db import init_db, session_scope
from recon_rag.eval import runner
from recon_rag.index.embedder import get_embedder
from recon_rag.llm.client import get_llm
from recon_rag.models import (
    CanonicalOrder,
    Discrepancy,
    Document,
    DriftEvent,
    EvalResult,
    EvalRun,
    SchemaVersion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Enterprise RAG & Reconciliation Agent",
    version=__version__,
    description="Agentic RAG over multi-source sales data with schema-drift detection, reconciliation and "
    "LLM-as-a-judge evaluation.",
    lifespan=lifespan,
)


def get_session():
    with session_scope() as s:
        yield s


# ------------------------------------------------------------------------------------------------ models


class GenerateRequest(BaseModel):
    n_orders: int = Field(2000, ge=50, le=50_000)
    seed: int = 42


class IngestRequest(BaseModel):
    use_llm_resolver: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    include_evidence: bool = False


class EvalRequest(BaseModel):
    sample_size: int | None = Field(None, ge=1, le=1000)


# ------------------------------------------------------------------------------------------------ routes


@app.get("/health")
def health(session: Session = Depends(get_session)):
    session.execute(text("SELECT 1"))
    counts = {
        name: session.scalar(select(func.count()).select_from(model))
        for name, model in [
            ("canonical_orders", CanonicalOrder),
            ("discrepancies", Discrepancy),
            ("documents", Document),
            ("drift_events", DriftEvent),
        ]
    }
    return {"status": "ok", "version": __version__, "counts": counts}


@app.post("/datasets/generate")
def generate(req: GenerateRequest):
    return services.generate_data(req.n_orders, req.seed)


@app.post("/ingest")
def ingest(req: IngestRequest | None = None):
    try:
        return services.ingest((req or IngestRequest()).use_llm_resolver)
    except FileNotFoundError as e:
        raise HTTPException(400, f"no data found: {e}. POST /datasets/generate first.") from e


@app.post("/reconcile")
def reconcile():
    return services.reconcile()


@app.post("/index")
def index():
    return services.index()


@app.post("/query")
def query(req: QueryRequest, session: Session = Depends(get_session)):
    s = get_settings()
    agent = Agent(get_llm("agent"), ToolExecutor(session, get_embedder()), s.agent_max_steps)
    return agent.answer(req.question).to_dict(include_evidence=req.include_evidence)


@app.get("/discrepancies")
def discrepancies(
    kind: str | None = None,
    region: Literal["NA", "EMEA", "APAC"] | None = None,
    order_id: str | None = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    session: Session = Depends(get_session),
):
    q = select(Discrepancy)
    if kind:
        q = q.where(Discrepancy.kind == kind)
    if region:
        q = q.where(Discrepancy.region == region)
    if order_id:
        q = q.where(Discrepancy.order_id == order_id)
    total = session.scalar(select(func.count()).select_from(q.subquery()))
    rows = session.scalars(q.order_by(Discrepancy.order_id).offset(offset).limit(limit)).all()
    return {
        "total": total,
        "items": [
            {
                "order_id": d.order_id,
                "kind": d.kind,
                "severity": d.severity,
                "region": d.region,
                "sources": d.sources,
                "detail": d.detail,
            }
            for d in rows
        ],
    }


@app.get("/drift")
def drift(source: str | None = None, session: Session = Depends(get_session)):
    q = select(DriftEvent).order_by(DriftEvent.source, DriftEvent.id)
    if source:
        q = q.where(DriftEvent.source == source)
    return [
        {
            "source": e.source,
            "from_version": e.from_version,
            "to_version": e.to_version,
            "kind": e.kind,
            "column": e.column,
            "detail": e.detail,
        }
        for e in session.scalars(q)
    ]


@app.get("/schemas")
def schemas(session: Session = Depends(get_session)):
    rows = session.scalars(select(SchemaVersion).order_by(SchemaVersion.source, SchemaVersion.version)).all()
    return [{"source": v.source, "version": v.version, "columns": v.columns, "mapping": v.mapping} for v in rows]


@app.post("/eval/runs", status_code=202)
def start_eval(req: EvalRequest | None = None):
    """Start an eval run in the background; poll GET /eval/runs/{id} until status != running."""
    run_id = runner.start_run()
    size = (req or EvalRequest()).sample_size
    threading.Thread(target=_safe_execute, args=(run_id, size), daemon=True).start()
    return {"run_id": run_id, "status": "running"}


def _safe_execute(run_id: int, size: int | None) -> None:
    try:
        runner.execute_run(run_id, sample_size=size)
    except Exception:
        logging.getLogger(__name__).exception("eval run %s crashed", run_id)


@app.get("/eval/runs")
def list_runs(limit: int = 20, session: Session = Depends(get_session)):
    runs = session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)).all()
    return [_run_json(r) for r in runs]


@app.get("/eval/runs/{run_id}")
def get_run(run_id: int, session: Session = Depends(get_session)):
    run = session.get(EvalRun, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    done = session.scalar(select(func.count()).where(EvalResult.run_id == run_id))
    return {**_run_json(run), "completed_cases": done}


@app.get("/eval/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: int, session: Session = Depends(get_session)):
    run = session.get(EvalRun, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    results = session.scalars(select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.case_id)).all()
    return runner.render_report(run, list(results))


def _run_json(r: EvalRun) -> dict:
    return {
        "id": r.id,
        "status": r.status,
        "agent_model": r.agent_model,
        "judge_model": r.judge_model,
        "n_cases": r.n_cases,
        "passed": r.passed,
        "metrics": r.metrics,
        "error": r.error,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
    }
