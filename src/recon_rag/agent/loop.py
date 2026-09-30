"""The agent: a Claude tool-use loop that ends in a structured, cited answer checked by the grounding gate."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from recon_rag.agent.grounding import GroundingResult, check_grounding
from recon_rag.agent.tools import TOOL_SPECS, ToolError, ToolExecutor
from recon_rag.llm.client import LLM, LLMRefusal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a reconciliation analyst for a company whose sales data lives in five systems: the CRM (crm, the \
system of record), the ERP (erp), and one warehouse feed per region (wh_na, wh_emea, wh_apac). Every record \
has been normalized onto one canonical order schema (order_id, customer_id, order_date, region, sku, \
quantity, unit_price, amount, currency, status), cross-checked between systems, and indexed. Regions are NA \
(USD), EMEA (EUR) and APAC (AUD). The data covers orders dated 2026-01-01 to 2026-06-30.

Answer the user's question using only what your tools return.

How to work:
- Look things up before answering. Use reconcile_order for a specific order, get_discrepancies for \
discrepancy lists and exact counts, get_schema_drift for schema changes, query_sales for totals, counts \
and averages, and search_documents for anything descriptive.
- Never count or add up items yourself from search results. Get the number from a tool that computes it.
- If the first tool call doesn't settle the question, make more calls. Filters are cheap.

Answer format (your final message is JSON):
- `answer`: a direct, concise answer. Copy every number, ID and value exactly as it appears in the \
evidence; don't round, convert currencies or do arithmetic. You may put evidence IDs inline in square \
brackets.
- `citations`: the IDs of every evidence item your answer relies on, exactly as the tools returned them.
- `abstained`: true when the tools don't return enough to answer. That covers a record that doesn't \
exist, a field the data doesn't contain, a period or region outside the data, or a forecast or opinion. \
When abstaining, say briefly what is missing, and don't guess.
"""

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "abstained": {"type": "boolean"},
    },
    "required": ["answer", "citations", "abstained"],
    "additionalProperties": False,
}

ABSTAIN_TEXT = "Insufficient evidence: I could not produce an answer fully supported by the retrieved data."


@dataclass
class ToolTrace:
    name: str
    input: dict
    evidence_ids: list[str]
    error: str | None = None


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[str]
    abstained: bool
    grounding: GroundingResult
    trace: list[ToolTrace] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)  # everything tools returned this session
    revised: bool = False
    forced_abstain: bool = False
    steps: int = 0
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self, include_evidence: bool = True) -> dict:
        d = asdict(self)
        if not include_evidence:
            d["evidence"] = {c: self.evidence[c] for c in self.citations if c in self.evidence}
        return d


def _parse_final(text: str) -> tuple[str, list[str], bool] | None:
    try:
        obj = json.loads(text)
        return str(obj["answer"]), [str(c) for c in obj.get("citations", [])], bool(obj.get("abstained", False))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


class Agent:
    def __init__(self, llm: LLM, tools: ToolExecutor, max_steps: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def answer(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        messages: list[dict] = [{"role": "user", "content": question}]
        evidence: dict[str, str] = {}
        trace: list[ToolTrace] = []
        usage: dict[str, int] = {}
        revised = False

        def done(answer: str, cites: list[str], abstained: bool, g: GroundingResult, steps: int, forced=False):
            return AgentResult(
                question,
                answer,
                cites,
                abstained,
                g,
                trace,
                evidence,
                revised,
                forced,
                steps,
                round((time.perf_counter() - t0) * 1000, 1),
                usage,
            )

        for step in range(1, self.max_steps + 1):
            try:
                turn = self.llm.step(
                    system=SYSTEM_PROMPT, messages=messages, tools=TOOL_SPECS, output_schema=ANSWER_SCHEMA
                )
            except LLMRefusal as e:
                return done(f"Request declined ({e}).", [], True, GroundingResult(ok=True), step, forced=True)
            for k, v in turn.usage.items():
                usage[k] = usage.get(k, 0) + v

            if turn.tool_calls:
                messages.append({"role": "assistant", "content": turn.assistant_content})
                results = []
                for call in turn.tool_calls:
                    try:
                        r = self.tools.run(call.name, call.input)
                        for e in r.evidence:
                            evidence[e.id] = e.text
                        trace.append(ToolTrace(call.name, call.input, [e.id for e in r.evidence]))
                        results.append({"type": "tool_result", "tool_use_id": call.id, "content": r.to_content()})
                    except ToolError as e:
                        trace.append(ToolTrace(call.name, call.input, [], str(e)))
                        results.append(
                            {"type": "tool_result", "tool_use_id": call.id, "content": str(e), "is_error": True}
                        )
                messages.append({"role": "user", "content": results})
                continue

            if turn.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": turn.assistant_content})
                continue

            parsed = _parse_final(turn.text)
            if parsed is None:
                g = GroundingResult(ok=False, reasons=["final answer was not valid JSON"])
                return done(ABSTAIN_TEXT, [], True, g, step, forced=True)
            answer, cites, abstained = parsed
            g = check_grounding(answer, cites, abstained, evidence, question)
            if g.ok:
                return done(answer, cites, abstained, g, step)
            if revised:
                log.info("grounding failed after revision: %s", g.reasons)
                return done(ABSTAIN_TEXT, [], True, g, step, forced=True)
            revised = True
            messages.append({"role": "assistant", "content": turn.assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": "Your answer failed the grounding check: "
                    + "; ".join(g.reasons)
                    + ". Revise it so every number and identifier appears in evidence you cite, using more tool "
                    "calls if needed, or set abstained to true.",
                }
            )

        g = GroundingResult(ok=False, reasons=[f"no final answer within {self.max_steps} steps"])
        return done(ABSTAIN_TEXT, [], True, g, self.max_steps, forced=True)
