"""LLM fallback for column mappings the heuristics can't decide on confidently."""

from __future__ import annotations

from pydantic import BaseModel, Field

from recon_rag.ingest.schema_drift import CANONICAL_FIELDS, ColumnProfile, Resolver
from recon_rag.llm.client import LLM

SYSTEM = (
    "You map columns from enterprise sales data feeds onto a canonical order schema. "
    "Only choose a field when the column name and sample values clearly describe it; otherwise return null. "
    "Taxes, fees, discounts and free-text columns are not the order amount."
)


class ColumnDecision(BaseModel):
    field: str | None = Field(description="Canonical field name, or null if none fits")
    confidence: float = Field(ge=0, le=1)
    reason: str


def make_llm_resolver(llm: LLM) -> Resolver:
    def resolve(source: str, column: str, profile: ColumnProfile, candidates: list[str]) -> tuple[str | None, float]:
        options = "\n".join(f"- {f}: {CANONICAL_FIELDS[f].description}" for f in candidates)
        prompt = (
            f"Source feed: {source}\nColumn: {column}\nInferred value type: {profile.kind}\n"
            f"Sample values: {profile.samples}\nNull rate: {profile.null_rate}\n\n"
            f"Unassigned canonical fields:\n{options}\n\nWhich field does this column hold?"
        )
        d = llm.structured(system=SYSTEM, prompt=prompt, schema=ColumnDecision)
        if d.field not in candidates:
            return None, d.confidence
        return d.field, d.confidence

    return resolve
