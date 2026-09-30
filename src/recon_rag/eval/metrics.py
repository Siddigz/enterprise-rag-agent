"""Aggregate per-case grades into benchmark metrics.

grounding_accuracy : share of *answered* cases (not abstained) whose every claim the judge found supported by
                     retrieved evidence **and** that passed the deterministic citation check.
hallucination_rate : 1 - grounding_accuracy.
answer_accuracy    : share of all cases the judge marked correct (abstaining on unanswerable cases counts as correct).
abstention_*       : precision/recall of abstaining, against cases that should be abstained on.

Grounding is measured only over answered cases, so abstaining on everything would score well on it. That's
why answer_accuracy and abstention_rate are always reported next to it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CaseOutcome:
    case_id: str
    category: str
    expect_abstain: bool
    abstained: bool
    judge_grounded: bool
    validator_ok: bool
    correct: bool
    latency_ms: float

    @property
    def grounded(self) -> bool:
        return self.judge_grounded and self.validator_ok


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def compute_metrics(outcomes: list[CaseOutcome]) -> dict:
    n = len(outcomes)
    answered = [o for o in outcomes if not o.abstained]
    abstained = [o for o in outcomes if o.abstained]
    should = [o for o in outcomes if o.expect_abstain]
    grounded = sum(o.grounded for o in answered)
    lat = sorted(o.latency_ms for o in outcomes)

    by_cat: dict[str, list[CaseOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.category].append(o)

    ga = _ratio(grounded, len(answered))
    return {
        "n_cases": n,
        "n_answered": len(answered),
        "grounding_accuracy": ga,
        "hallucination_rate": round(1 - ga, 4) if ga is not None else None,
        "answer_accuracy": _ratio(sum(o.correct for o in outcomes), n),
        "abstention_rate": _ratio(len(abstained), n),
        "abstention_precision": _ratio(sum(o.expect_abstain for o in abstained), len(abstained)),
        "abstention_recall": _ratio(sum(o.abstained for o in should), len(should)),
        "judge_validator_agreement": _ratio(sum(o.judge_grounded == o.validator_ok for o in answered), len(answered)),
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "latency_ms_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
        "by_category": {
            cat: {
                "n": len(os_),
                "grounding_accuracy": _ratio(
                    sum(o.grounded for o in os_ if not o.abstained), sum(not o.abstained for o in os_)
                ),
                "answer_accuracy": _ratio(sum(o.correct for o in os_), len(os_)),
                "abstention_rate": _ratio(sum(o.abstained for o in os_), len(os_)),
            }
            for cat, os_ in sorted(by_cat.items())
        },
    }
