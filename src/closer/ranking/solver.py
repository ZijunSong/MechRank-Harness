"""Global rank solver entry point."""

from __future__ import annotations

from closer.config import RankingConfig
from closer.ranking.bradley_terry import fit_bradley_terry
from closer.ranking.graph import PreferenceGraph
from closer.schemas.ranking import SolverResult


class GlobalRankSolver:
    def __init__(self, config: RankingConfig) -> None:
        self.config = config

    def solve(self, variant_ids: list[str], graph: PreferenceGraph) -> SolverResult:
        if self.config.method != "bradley_terry":
            raise ValueError(f"unsupported ranking method {self.config.method}")
        return fit_bradley_terry(variant_ids, graph, self.config)
