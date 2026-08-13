"""Assay contract schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MechanismName = Literal[
    "stability",
    "folding",
    "binding",
    "catalysis",
    "expression",
    "localization",
    "regulation",
    "allostery",
    "interaction",
    "toxicity",
    "growth",
    "other",
]


class MechanismRelevance(BaseModel):
    mechanism: MechanismName
    relevance: Literal["low", "medium", "high"]
    reason: str


class AssayContract(BaseModel):
    measured_property: str
    biological_context: str
    higher_is_better: bool
    direct_effect_interpretation: str
    likely_causal_chain: list[str]
    mechanism_relevance: list[MechanismRelevance]
    ambiguity_notes: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
