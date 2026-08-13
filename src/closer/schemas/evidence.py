"""Mechanistic evidence schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MechanismType = Literal[
    "stability",
    "folding",
    "binding",
    "functional_site",
    "catalysis",
    "expression",
    "regulation",
    "allostery",
    "interaction",
    "epistasis",
    "local_sequence_context",
    "other",
]

NetAssayEffect = Literal[
    "strong_beneficial",
    "beneficial",
    "near_neutral",
    "deleterious",
    "strong_deleterious",
    "uncertain",
]

NET_EFFECT_ORDINAL = {
    "strong_beneficial": 2,
    "beneficial": 1,
    "near_neutral": 0,
    "uncertain": 0,
    "deleterious": -1,
    "strong_deleterious": -2,
}

NET_EFFECT_RANK_ORDER = {
    "strong_beneficial": 4,
    "beneficial": 3,
    "near_neutral": 2,
    "uncertain": 2,
    "deleterious": 1,
    "strong_deleterious": 0,
}


class MechanisticEvidence(BaseModel):
    mechanism: MechanismType
    direction: Literal["beneficial", "deleterious", "neutral", "uncertain"]
    strength: Literal["weak", "medium", "strong", "unknown"]
    assay_relevance: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class VariantEvidenceState(BaseModel):
    variant_id: str
    mutation_summary: str
    evidence: list[MechanisticEvidence]
    epistasis_required: bool
    dominant_mechanisms: list[str]
    net_assay_effect: NetAssayEffect
    net_confidence: float = Field(ge=0.0, le=1.0)
    unresolved_questions: list[str]
