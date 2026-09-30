"""LLM-as-a-judge: grades each agent answer for grounding (faithfulness to retrieved evidence) and
correctness (agreement with the reference answer), claim by claim."""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field

from recon_rag.agent.loop import AgentResult
from recon_rag.eval.golden import GoldenCase
from recon_rag.llm.client import LLM

JUDGE_SYSTEM = """\
You are a strict evaluator of a retrieval-augmented analytics agent. You get a question, the evidence the \
agent retrieved, the agent's answer, and a reference answer written from ground truth.

Grade two things independently:
1. grounded: split the answer into atomic factual claims. A claim is supported only if the retrieved \
evidence states it or it follows from the evidence without outside knowledge. Restating the question or \
saying that data is unavailable is not a factual claim. grounded is true only when every claim is \
supported. An abstention with no factual claims is grounded.
2. correct: does the answer agree with the reference answer on the facts the question asks for? Extra \
supported detail is fine. When the reference says the question can't be answered from the data, the \
answer is correct only if the agent declined or said the information is unavailable.

Judge only from the evidence shown. Don't reward confident wording.
"""


class Claim(BaseModel):
    claim: str
    supported: bool
    evidence_ids: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    claims: list[Claim]
    grounded: bool
    correct: bool
    agent_abstained: bool
    rationale: str


class Judge(Protocol):
    name: str

    def grade(self, case: GoldenCase, result: AgentResult) -> Verdict: ...


class LLMJudge:
    def __init__(self, llm: LLM, max_evidence_chars: int = 60_000):
        self.llm = llm
        self.name = f"llm:{llm.model}"
        self.max_evidence_chars = max_evidence_chars

    def grade(self, case: GoldenCase, result: AgentResult) -> Verdict:
        ev = "\n\n".join(f"[{k}]\n{v}" for k, v in result.evidence.items())
        if len(ev) > self.max_evidence_chars:
            # Keep cited evidence in full; trim uncited context rather than the answer's support.
            cited = "\n\n".join(f"[{k}]\n{result.evidence[k]}" for k in result.citations if k in result.evidence)
            ev = cited + "\n\n" + ev[: max(0, self.max_evidence_chars - len(cited))]
        prompt = (
            f"<question>\n{case.question}\n</question>\n\n"
            f"<retrieved_evidence>\n{ev or '(none)'}\n</retrieved_evidence>\n\n"
            f'<agent_answer abstained="{str(result.abstained).lower()}">\n{result.answer}\n'
            f"Cited: {json.dumps(result.citations)}\n</agent_answer>\n\n"
            f'<reference_answer answerable="{str(not case.expect_abstain).lower()}">\n{case.expected}\n'
            f"</reference_answer>"
        )
        return self.llm.structured(system=JUDGE_SYSTEM, prompt=prompt, schema=Verdict)


class HeuristicJudge:
    """Offline judge for CI and smoke tests (no API key). Grounding = the deterministic citation check; correctness
    = every key fact present in the answer. Useful as a floor and for regressions; not a substitute for the LLM
    judge."""

    name = "heuristic"

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower())

    def grade(self, case: GoldenCase, result: AgentResult) -> Verdict:
        ans = self._norm(result.answer)
        if case.expect_abstain:
            correct = result.abstained
        else:
            correct = not result.abstained and all(
                any(self._norm(alt) in ans for alt in fact) for fact in case.key_facts
            )
        return Verdict(
            claims=[],
            grounded=result.grounding.ok,
            correct=correct,
            agent_abstained=result.abstained,
            rationale="heuristic: key-fact match + deterministic grounding check",
        )
