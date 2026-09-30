"""Thin LLM interface used by the agent, the judge and the drift resolver.

``AnthropicLLM`` calls Claude through the official SDK. ``ScriptedLLM`` is a deterministic stand-in so the
agent loop, eval harness and API can be exercised in tests and CI without an API key.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from recon_rag.config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMTurn:
    stop_reason: str
    text: str
    tool_calls: list[ToolCall]
    assistant_content: Any  # appended verbatim to the transcript (keeps thinking/fallback blocks intact)
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def step(self, *, system: str, messages: list[dict], tools: list[dict], output_schema: dict | None) -> LLMTurn: ...

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


class AnthropicLLM:
    def __init__(self, model: str, effort: str, fallbacks: str = "default", max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.max_tokens = max_tokens

    def _common(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        if self.fallbacks:
            kw["betas"] = [SERVER_SIDE_FALLBACK_BETA]
            kw["fallbacks"] = self.fallbacks
        return kw

    def step(self, *, system: str, messages: list[dict], tools: list[dict], output_schema: dict | None) -> LLMTurn:
        output_config: dict[str, Any] = {"effort": self.effort}
        if output_schema:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        resp = self.client.beta.messages.create(
            **self._common(),
            system=system,
            tools=tools,
            messages=messages,
            output_config=output_config,
            cache_control={"type": "ephemeral"},  # system + tools are stable across turns and questions
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(getattr(resp.stop_details, "category", None) or "refusal")
        if resp.stop_reason == "max_tokens":
            log.warning("agent turn hit max_tokens")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        }
        return LLMTurn(resp.stop_reason or "", text, calls, resp.content, usage)

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        resp = self.client.beta.messages.parse(
            **self._common(),
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": self.effort},
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(getattr(resp.stop_details, "category", None) or "refusal")
        if resp.parsed_output is None:
            raise ValueError(f"no structured output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


Policy = Callable[[str, list[dict], list[dict]], LLMTurn]


class ScriptedLLM:
    """Deterministic LLM double. ``policy`` decides each agent turn; ``structured_fn`` answers structured calls."""

    def __init__(self, policy: Policy | None = None, structured_fn: Callable[[str, type], BaseModel] | None = None):
        self.model = "scripted"
        self.policy = policy or extractive_policy
        self.structured_fn = structured_fn
        self.calls = 0

    def step(self, *, system: str, messages: list[dict], tools: list[dict], output_schema: dict | None) -> LLMTurn:
        self.calls += 1
        return self.policy(system, messages, tools)

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        if self.structured_fn is None:
            raise NotImplementedError("ScriptedLLM has no structured_fn")
        return self.structured_fn(prompt, schema)  # type: ignore[return-value]


def final_turn(answer: str, citations: list[str], abstained: bool = False) -> LLMTurn:
    text = json.dumps({"answer": answer, "citations": citations, "abstained": abstained})
    return LLMTurn("end_turn", text, [], [{"type": "text", "text": text}])


def tool_turn(name: str, args: dict, call_id: str = "call_1") -> LLMTurn:
    content = [{"type": "tool_use", "id": call_id, "name": name, "input": args}]
    return LLMTurn("tool_use", "", [ToolCall(call_id, name, args)], content)


def extractive_policy(system: str, messages: list[dict], tools: list[dict]) -> LLMTurn:
    """Offline baseline: search once, then answer with the first line of the top hit, citing it.

    It is deliberately naive. It exists to smoke-test the pipeline end to end, not to score well.
    """
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], str):
        return tool_turn("search_documents", {"query": last["content"]})
    results = [c for c in last["content"] if isinstance(c, dict) and c.get("type") == "tool_result"]
    if results:
        payload = json.loads(results[0]["content"])
        ev = payload.get("evidence", [])
        if ev:
            return final_turn(ev[0]["text"].split("\n")[0], [ev[0]["id"]])
    return final_turn("I could not find evidence to answer this question.", [], abstained=True)


def get_llm(role: str = "agent") -> LLM:
    s = get_settings()
    if s.llm_provider == "fake":
        return ScriptedLLM()
    if role == "judge":
        return AnthropicLLM(s.judge_model, s.judge_effort, s.anthropic_fallbacks)
    return AnthropicLLM(s.agent_model, s.agent_effort, s.anthropic_fallbacks)
