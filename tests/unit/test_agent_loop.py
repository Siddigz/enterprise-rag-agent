import json

from recon_rag.agent.loop import ABSTAIN_TEXT, Agent
from recon_rag.agent.tools import Evidence, ToolError, ToolResult
from recon_rag.llm.client import LLMRefusal, LLMTurn, ScriptedLLM, final_turn, tool_turn


class FakeTools:
    def __init__(self):
        self.calls = []

    def run(self, name, args):
        self.calls.append((name, args))
        if name == "boom":
            raise ToolError("bad args")
        return ToolResult([Evidence("order:SO-000001", "Order SO-000001 amount 20.00 USD, status delivered.")])


def scripted(*turns):
    it = iter(turns)
    return ScriptedLLM(policy=lambda system, messages, tools: next(it))


def test_grounded_answer_is_returned():
    llm = scripted(tool_turn("search_documents", {"query": "x"}), final_turn("It is 20.00 USD.", ["order:SO-000001"]))
    r = Agent(llm, FakeTools()).answer("How much is SO-000001?")
    assert r.answer == "It is 20.00 USD." and r.grounding.ok and not r.abstained
    assert r.trace[0].evidence_ids == ["order:SO-000001"]


def test_ungrounded_answer_gets_one_revision_then_passes():
    llm = scripted(
        tool_turn("search_documents", {"query": "x"}),
        final_turn("It is 25.00 USD.", ["order:SO-000001"]),
        final_turn("It is 20.00 USD.", ["order:SO-000001"]),
    )
    r = Agent(llm, FakeTools()).answer("q")
    assert r.revised and r.grounding.ok and r.answer == "It is 20.00 USD."


def test_repeatedly_ungrounded_answer_is_forced_to_abstain():
    llm = scripted(
        tool_turn("search_documents", {"query": "x"}),
        final_turn("It is 25.00 USD.", ["order:SO-000001"]),
        final_turn("It is 26.00 USD.", ["order:SO-000001"]),
    )
    r = Agent(llm, FakeTools()).answer("q")
    assert r.abstained and r.forced_abstain and r.answer == ABSTAIN_TEXT and r.citations == []


def test_tool_errors_are_returned_to_the_model():
    seen = {}

    def policy(system, messages, tools):
        if len(messages) == 1:
            return tool_turn("boom", {})
        seen["result"] = messages[-1]["content"][0]
        return final_turn("Can't tell.", [], abstained=True)

    r = Agent(ScriptedLLM(policy), FakeTools()).answer("q")
    assert seen["result"]["is_error"] is True
    assert r.trace[0].error == "bad args"


def test_step_limit_abstains():
    llm = ScriptedLLM(lambda s, m, t: tool_turn("search_documents", {"query": "again"}))
    r = Agent(llm, FakeTools(), max_steps=3).answer("q")
    assert r.abstained and r.steps == 3


def test_invalid_json_final_answer_abstains():
    llm = scripted(LLMTurn("end_turn", "not json", [], []))
    assert Agent(llm, FakeTools()).answer("q").abstained


def test_refusal_is_reported_as_abstention():
    def refuse(*_):
        raise LLMRefusal("cyber")

    r = Agent(ScriptedLLM(refuse), FakeTools()).answer("q")
    assert r.abstained and "cyber" in r.answer


def test_extractive_baseline_policy_cites_top_hit():
    r = Agent(ScriptedLLM(), FakeTools()).answer("How much is SO-000001?")
    assert r.citations == ["order:SO-000001"] and r.grounding.ok


def test_tool_results_are_json_with_evidence():
    content = ToolResult([Evidence("a", "b")], {"n": 1}).to_content()
    assert json.loads(content) == {"evidence": [{"id": "a", "text": "b"}], "n": 1}
