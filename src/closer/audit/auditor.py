"""Adaptive auditor: targeted re-reasoning on unreliable ranking regions."""

from __future__ import annotations

from closer.audit import prompts
from closer.audit.detector import AuditItem, detect_audit_items, select_pair_audit_items
from closer.comparison.validation import normalize_pair_from_allowed_set, normalize_requested_pair
from closer.config import AuditConfig
from closer.errors import AuditError, StructuredOutputError
from closer.evidence.state import EpisodeEvidenceState
from closer.llm.base import LLMClient
from closer.orchestrator.budget import BudgetManager
from closer.ranking.graph import PreferenceGraph
from closer.ranking.solver import GlobalRankSolver
from closer.schemas.comparison import PairPreference
from closer.schemas.ranking import SolverResult


class AdaptiveAuditor:
    def __init__(self, client: LLMClient, config: AuditConfig, solver: GlobalRankSolver) -> None:
        self.client = client
        self.config = config
        self.solver = solver

    async def run(
        self,
        *,
        variant_ids: list[str],
        state: EpisodeEvidenceState,
        graph: PreferenceGraph,
        provisional: SolverResult,
        budget: BudgetManager | None = None,
    ) -> tuple[SolverResult, list[dict]]:
        history: list[dict] = []
        current = provisional
        unchanged_rounds = 0
        for round_index in range(self.config.max_rounds):
            items = detect_audit_items(
                variant_ids=variant_ids,
                evidence=state.variant_evidence,
                graph=graph,
                solver=current,
                config=self.config,
            )
            selected = select_pair_audit_items(items, self.config)
            if not selected:
                history.append({"round": round_index, "n_items": 0, "stop": "no_high_priority_items"})
                break
            for item in selected:
                pref = await self._recompare(state, item, graph, current, budget)
                graph.add_pair_preference(pref, source="audit", round_index=100 + round_index)
            updated = self.solver.solve(variant_ids, graph)
            changed = updated.ranking != current.ranking
            history.append(
                {
                    "round": round_index,
                    "n_items": len(selected),
                    "items": [item.model_dump(mode="json") for item in selected],
                    "ranking_changed": changed,
                }
            )
            if not changed:
                unchanged_rounds += 1
                if unchanged_rounds >= 2:
                    history[-1]["stop"] = "ranking_unchanged"
                    current = updated
                    break
            else:
                unchanged_rounds = 0
            current = updated
        return current, history

    async def _recompare(
        self,
        state: EpisodeEvidenceState,
        item: AuditItem,
        graph: PreferenceGraph,
        solver: SolverResult,
        budget: BudgetManager | None,
    ) -> PairPreference:
        if item.kind == "epistasis" or len(item.variant_ids) < 2:
            raise AuditError("single-variant audit cannot produce a pair preference")
        try:
            parsed, _trace = await self.client.generate_structured(
                stage="adaptive_auditor",
                schema=PairPreference,
                system=prompts.SYSTEM,
                user=prompts.user_prompt(state, item, graph, solver),
                budget=budget,
            )
        except StructuredOutputError as exc:
            raise AuditError(str(exc)) from exc
        if not isinstance(parsed, PairPreference):
            raise AuditError("auditor returned the wrong schema")
        if len(item.variant_ids) == 2:
            left, right = item.variant_ids[0], item.variant_ids[1]
            return normalize_requested_pair(parsed, left, right)
        return normalize_pair_from_allowed_set(parsed, item.variant_ids)
