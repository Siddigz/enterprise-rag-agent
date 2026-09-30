"""Deterministic grounding check applied to every agent answer before it is returned.

An answer passes only if it cites at least one evidence ID, every cited ID was actually returned by a tool
in this session, and every number and business identifier in the answer appears in the *cited* evidence
(or in the question itself). Anything else counts as unsupported: the agent gets one chance to revise, then
abstains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ID_RE = re.compile(r"\b(?:SO-\d{4,8}|C-\d{3,6}|SKU-\d{2,4})\b", re.I)
DATE_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
CITATION_TAG_RE = re.compile(r"\[[^\[\]]*:[^\[\]]*\]")  # inline [evidence:id] tags are not claims


@dataclass
class GroundingResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_ids: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for m in DATE_RE.finditer(text):
        out.update(float(g) for g in m.groups() if g)
    text = DATE_RE.sub(" ", ID_RE.sub(" ", text))
    for whole, frac in NUM_RE.findall(text):
        out.add(round(float(whole.replace(",", "") + (frac or "")), 2))
    return out


def _ids(text: str) -> set[str]:
    return {m.upper() for m in ID_RE.findall(text)}


def check_grounding(
    answer: str, citations: list[str], abstained: bool, evidence: dict[str, str], question: str = ""
) -> GroundingResult:
    if abstained:
        return GroundingResult(ok=True)
    res = GroundingResult(ok=True)
    if not citations:
        res.reasons.append("answer cites no evidence")
    res.invalid_citations = [c for c in citations if c not in evidence]
    if res.invalid_citations:
        res.reasons.append(f"cited IDs never returned by a tool: {res.invalid_citations}")

    cited_text = "\n".join(evidence[c] for c in citations if c in evidence)
    claim_text = CITATION_TAG_RE.sub(" ", answer)
    allowed_nums = _numbers(cited_text) | _numbers(question)
    res.unsupported_numbers = sorted(
        f"{n:g}" for n in _numbers(claim_text) if not any(abs(n - a) < 0.006 for a in allowed_nums)
    )
    if res.unsupported_numbers:
        res.reasons.append(f"numbers not found in cited evidence: {res.unsupported_numbers}")
    allowed_ids = _ids(cited_text) | _ids(question)
    res.unsupported_ids = sorted(_ids(claim_text) - allowed_ids)
    if res.unsupported_ids:
        res.reasons.append(f"identifiers not found in cited evidence: {res.unsupported_ids}")
    res.ok = not res.reasons
    return res
