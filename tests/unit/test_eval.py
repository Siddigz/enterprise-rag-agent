from collections import Counter

from recon_rag.agent.grounding import GroundingResult
from recon_rag.agent.loop import AgentResult
from recon_rag.eval.golden import build_golden
from recon_rag.eval.judge import HeuristicJudge
from recon_rag.eval.metrics import CaseOutcome, compute_metrics
from recon_rag.eval.runner import select_cases
from recon_rag.index.retriever import extract_keys, or_tsquery, rrf


def test_golden_set_shape(dataset):
    _, truth = dataset
    cases = build_golden(truth)
    cats = Counter(c.category for c in cases)
    assert set(cats) == {"order_lookup", "discrepancy", "aggregate", "drift", "multi_hop", "unanswerable"}
    assert 130 <= len(cases) <= 160
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.key_facts for c in cases if not c.expect_abstain)
    assert all(c.expect_abstain for c in cases if c.category == "unanswerable")


def test_golden_is_deterministic(dataset):
    _, truth = dataset
    assert [c.question for c in build_golden(truth)] == [c.question for c in build_golden(truth)]


def test_stratified_sample_keeps_every_category(dataset):
    _, truth = dataset
    cases = select_cases(truth, 20)
    assert len(cases) <= 20
    assert len({c.category for c in cases}) == 6


def _result(answer, abstained=False, ok=True):
    return AgentResult("q", answer, ["x"], abstained, GroundingResult(ok=ok))


def test_heuristic_judge(dataset):
    _, truth = dataset
    case = next(c for c in build_golden(truth) if c.category == "order_lookup")
    fact_answer = " ".join(f[0] for f in case.key_facts)
    assert HeuristicJudge().grade(case, _result(fact_answer)).correct
    assert not HeuristicJudge().grade(case, _result("something else")).correct
    unans = next(c for c in build_golden(truth) if c.expect_abstain)
    assert HeuristicJudge().grade(unans, _result("n/a", abstained=True)).correct
    assert not HeuristicJudge().grade(unans, _result("made up", abstained=False)).correct


def test_metrics():
    o = [
        CaseOutcome("a", "x", False, False, True, True, True, 10),
        CaseOutcome("b", "x", False, False, False, True, False, 20),  # judge says ungrounded
        CaseOutcome("c", "y", True, True, True, True, True, 30),  # correct abstention
        CaseOutcome("d", "y", False, True, True, True, False, 40),  # unnecessary abstention
    ]
    m = compute_metrics(o)
    assert m["grounding_accuracy"] == 0.5 and m["hallucination_rate"] == 0.5
    assert m["answer_accuracy"] == 0.5
    assert m["abstention_precision"] == 0.5 and m["abstention_recall"] == 1.0
    assert m["by_category"]["y"]["grounding_accuracy"] is None


def test_rrf_prefers_documents_ranked_well_in_multiple_lists():
    fused = rrf([["a", "b", "c"], ["b", "a"], ["b"]], k=60)
    assert [d for d, _ in fused] == ["b", "a", "c"]


def test_query_helpers():
    assert or_tsquery("Why is SO-000123 wrong?") == "why | is | so | 000123 | wrong"
    assert extract_keys("compare so000123 and C0042") == (["SO-000123"], ["C-0042"])
