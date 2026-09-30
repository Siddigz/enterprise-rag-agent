"""Pipeline stages shared by the API, the CLI and the Airflow DAG."""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from dataclasses import asdict

from recon_rag.config import get_settings
from recon_rag.datagen import generate, load_ground_truth
from recon_rag.db import session_scope
from recon_rag.index.embedder import get_embedder
from recon_rag.index.indexer import build_index
from recon_rag.ingest.pipeline import ingest_directory
from recon_rag.reconcile.engine import score_against_truth
from recon_rag.reconcile.service import run_reconciliation

log = logging.getLogger(__name__)


def generate_data(n_orders: int = 2000, seed: int = 42) -> dict:
    truth = generate(get_settings().data_dir, n_orders, seed)
    return {
        "n_orders": n_orders,
        "seed": seed,
        "injected_discrepancies": len(truth["discrepancies"]),
        "expected_drift_events": len(truth["drift"]),
    }


def ingest(use_llm_resolver: bool = False) -> dict:
    resolver = None
    if use_llm_resolver:
        from recon_rag.llm.client import get_llm
        from recon_rag.llm.resolver import make_llm_resolver

        resolver = make_llm_resolver(get_llm("agent"))
    with session_scope() as session:
        report = ingest_directory(session, get_settings().data_dir, resolver=resolver)
    return asdict(report)


def reconcile() -> dict:
    with session_scope() as session:
        findings = run_reconciliation(session)
    out: dict = {"discrepancies": len(findings), "by_kind": dict(Counter(f.kind for f in findings))}
    with contextlib.suppress(FileNotFoundError):
        truth = load_ground_truth(get_settings().data_dir)
        out["vs_ground_truth"] = score_against_truth(findings, truth["discrepancies"])
    return out


def index() -> dict:
    with session_scope() as session:
        counts = build_index(session, get_embedder())
    return {"documents": sum(counts.values()), "by_kind": counts}


def run_pipeline(generate_first: bool = False, n_orders: int = 2000, seed: int = 42) -> dict:
    out = {}
    if generate_first:
        out["generate"] = generate_data(n_orders, seed)
    out["ingest"] = ingest()
    out["reconcile"] = reconcile()
    out["index"] = index()
    return out
