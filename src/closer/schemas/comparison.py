"""Comparison and preference-graph schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PairPreference(BaseModel):
    left_id: str
    right_id: str
    winner: Literal["left", "right", "tie", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_evidence: list[str]
    contradiction_with_previous_evidence: bool
    rationale: str


class SetwiseRanking(BaseModel):
    variant_ids: list[str]
    ranking: list[str]
    pair_preferences: list[PairPreference]
    confidence: float = Field(ge=0.0, le=1.0)


class PreferenceEdge(BaseModel):
    winner: str
    loser: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["setwise", "pairwise", "audit"]
    round_index: int
    rationale_ref: str
    observation_id: str | None = None


class PreferenceObservation(BaseModel):
    observation_id: str
    left_id: str
    right_id: str
    winner: Literal["left", "right", "tie", "uncertain"]
    confidence: float
    source: Literal["setwise", "pairwise", "audit"]
    round_index: int
    rationale: str
    key_evidence: list[str] = Field(default_factory=list)
    contradiction_with_previous_evidence: bool = False
    comparison_event_id: str | None = None
    derived_from_order: bool = False
    revision_of: str | None = None
    status: Literal["active", "superseded"] = "active"
