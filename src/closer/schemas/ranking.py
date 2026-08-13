"""Ranking result schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from closer.schemas.assay import AssayContract


class SolverResult(BaseModel):
    ranking: list[str]
    scores: dict[str, float]
    method: str
    converged: bool
    n_observations: int
    n_informative_edges: int


class CloserResult(BaseModel):
    ranking: list[str]
    scores: dict[str, float]
    assay_contract: AssayContract
    diagnostics: dict
    usage: dict
    trace_path: str


class RankingJSON(BaseModel):
    ranking: list[str] = Field(min_length=1)
