"""Run the benchmark: agent answers every golden case, the judge grades it, metrics and a report are stored."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from recon_rag.agent.loop import Agent, AgentResult
from recon_rag.agent.tools import ToolExecutor
from recon_rag.config import get_settings
from recon_rag.datagen import load_ground_truth
from recon_rag.db import session_scope
from recon_rag.eval.golden import GoldenCase, build_golden
from recon_rag.eval.judge import HeuristicJudge, Judge, LLMJudge, Verdict
from recon_rag.eval.metrics import CaseOutcome, compute_metrics
from recon_rag.index.embedder import get_embedder
from recon_rag.llm.client import LLM, get_llm
from recon_rag.models import EvalResult, EvalRun

log = logging.getLogger(__name__)


def select_cases(truth: dict, sample_size: int | None, seed: int = 11) -> list[GoldenCase]:
    cases = build_golden(truth)
    if sample_size and sample_size < len(cases):
        # Stratified so every category stays represented in small runs.
        rng = random.Random(seed)
        by_cat: dict[str, list[GoldenCase]] = {}
        for c in cases:
            by_cat.setdefault(c.category, []).append(c)
        picked = []
        for cat_cases in by_cat.values():
            k = max(1, round(sample_size * len(cat_cases) / len(cases)))
            picked += rng.sample(cat_cases, min(k, len(cat_cases)))
        cases = picked[:sample_size]
    return cases


def default_judge() -> Judge:
    s = get_settings()
    return HeuristicJudge() if s.llm_provider == "fake" else LLMJudge(get_llm("judge"))


def _run_case(case: GoldenCase, llm: LLM, judge: Judge) -> tuple[GoldenCase, AgentResult, Verdict]:
    s = get_settings()
    with session_scope() as session:
        agent = Agent(llm, ToolExecutor(session, get_embedder()), s.agent_max_steps)
        result = agent.answer(case.question)
    try:
        verdict = judge.grade(case, result)
    except Exception as e:  # a judge failure must not look like a pass
        log.exception("judge failed on %s", case.id)
        verdict = Verdict(
            claims=[], grounded=False, correct=False, agent_abstained=result.abstained, rationale=f"judge error: {e}"
        )
    return case, result, verdict


def start_run(llm: LLM | None = None, judge: Judge | None = None) -> int:
    llm = llm or get_llm("agent")
    judge = judge or default_judge()
    with session_scope() as session:
        run = EvalRun(status="running", agent_model=llm.model, judge_model=judge.name)
        session.add(run)
        session.flush()
        return run.id


def execute_run(
    run_id: int, llm: LLM | None = None, judge: Judge | None = None, sample_size: int | None = None
) -> dict:
    s = get_settings()
    llm = llm or get_llm("agent")
    judge = judge or default_judge()
    try:
        cases = select_cases(load_ground_truth(s.data_dir), sample_size or s.eval_sample_size)
        outcomes: list[CaseOutcome] = []
        with ThreadPoolExecutor(max_workers=s.eval_concurrency) as pool:
            for case, result, verdict in pool.map(lambda c: _run_case(c, llm, judge), cases):
                outcome = CaseOutcome(
                    case.id,
                    case.category,
                    case.expect_abstain,
                    result.abstained,
                    verdict.grounded,
                    result.grounding.ok,
                    verdict.correct,
                    result.latency_ms,
                )
                outcomes.append(outcome)
                with session_scope() as session:
                    session.add(
                        EvalResult(
                            run_id=run_id,
                            case_id=case.id,
                            category=case.category,
                            question=case.question,
                            expected=case.expected,
                            answer=result.answer,
                            citations=result.citations,
                            abstained=result.abstained,
                            grounded=outcome.grounded,
                            correct=verdict.correct,
                            verdict={
                                **verdict.model_dump(),
                                "validator": result.grounding.__dict__,
                                "trace": [t.__dict__ for t in result.trace],
                                "usage": result.usage,
                                "forced_abstain": result.forced_abstain,
                                "revised": result.revised,
                            },
                            latency_ms=result.latency_ms,
                        )
                    )
        metrics = compute_metrics(outcomes)
        ga = metrics["grounding_accuracy"]
        acc = metrics["answer_accuracy"]
        passed = (
            ga is not None and ga >= s.grounding_threshold and acc is not None and acc >= s.answer_accuracy_threshold
        )
        with session_scope() as session:
            run = session.get(EvalRun, run_id)
            run.status, run.metrics, run.passed = "completed", metrics, passed
            run.n_cases, run.finished_at = len(cases), datetime.now(UTC)
        write_report(run_id)
        return metrics
    except Exception as e:
        log.exception("eval run %s failed", run_id)
        with session_scope() as session:
            run = session.get(EvalRun, run_id)
            run.status, run.error, run.finished_at = "failed", repr(e), datetime.now(UTC)
        raise


def render_report(run: EvalRun, results: list[EvalResult]) -> str:
    m = run.metrics or {}
    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    lines = [
        f"# Eval run {run.id}",
        "",
        f"- Agent: `{run.agent_model}` · Judge: `{run.judge_model}` · Cases: {run.n_cases}",
        f"- Started {run.started_at:%Y-%m-%d %H:%M} UTC · Status: **{run.status}** · Gate: "
        f"**{'PASS' if run.passed else 'FAIL'}** (grounding ≥ {pct(get_settings().grounding_threshold)}, "
        f"answer accuracy ≥ {pct(get_settings().answer_accuracy_threshold)})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Grounding accuracy (answered cases) | **{pct(m.get('grounding_accuracy'))}** |",
        f"| Hallucination rate | {pct(m.get('hallucination_rate'))} |",
        f"| Answer accuracy (all cases) | {pct(m.get('answer_accuracy'))} |",
        f"| Abstention rate | {pct(m.get('abstention_rate'))} |",
        f"| Abstention precision / recall | {pct(m.get('abstention_precision'))} / {pct(m.get('abstention_recall'))} |",
        f"| Judge ↔ validator agreement | {pct(m.get('judge_validator_agreement'))} |",
        f"| Latency p50 / p95 | {m.get('latency_ms_p50')} ms / {m.get('latency_ms_p95')} ms |",
        "",
        "| Category | n | Grounding | Accuracy | Abstained |",
        "|---|---|---|---|---|",
    ]
    for cat, c in (m.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {c['n']} | {pct(c['grounding_accuracy'])} | {pct(c['answer_accuracy'])} | "
            f"{pct(c['abstention_rate'])} |"
        )
    failures = [r for r in results if not r.correct or (not r.abstained and not r.grounded)]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures[:40]:
            flag = "ungrounded" if (not r.abstained and not r.grounded) else "incorrect"
            lines += [
                f"**{r.case_id}** ({flag}): {r.question}",
                f"- expected: {r.expected}",
                f"- answer: {r.answer}",
                f"- judge: {r.verdict.get('rationale', '')}",
                "",
            ]
    return "\n".join(lines) + "\n"


def write_report(run_id: int) -> Path:
    s = get_settings()
    with session_scope() as session:
        run = session.get(EvalRun, run_id)
        results = session.scalars(
            select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.case_id)
        ).all()
        md = render_report(run, list(results))
        payload = {
            "run_id": run.id,
            "agent_model": run.agent_model,
            "judge_model": run.judge_model,
            "passed": run.passed,
            "metrics": run.metrics,
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "question": r.question,
                    "expected": r.expected,
                    "answer": r.answer,
                    "citations": r.citations,
                    "abstained": r.abstained,
                    "grounded": r.grounded,
                    "correct": r.correct,
                    "verdict": r.verdict,
                }
                for r in results
            ],
        }
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    (s.reports_dir / f"eval_run_{run_id}.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    path = s.reports_dir / f"eval_run_{run_id}.md"
    path.write_text(md, encoding="utf-8")
    return path
