"""LLM comparative reasoner for setwise and pairwise judgments."""

from __future__ import annotations

from closer.comparison import prompts
from closer.errors import ComparisonError, StructuredOutputError
from closer.evidence.state import EpisodeEvidenceState
from closer.llm.base import LLMClient
from closer.orchestrator.budget import BudgetManager
from closer.schemas.comparison import PairPreference, SetwiseRanking


class ComparativeReasoner:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def rank_set(
        self,
        state: EpisodeEvidenceState,
        variant_ids: list[str],
        *,
        budget: BudgetManager | None = None,
    ) -> SetwiseRanking:
        if len(variant_ids) < 2:
            raise ComparisonError("setwise ranking requires at least 2 variants")
        try:
            parsed, _trace = await self.client.generate_structured(
                stage="comparative_reasoner_setwise",
                schema=SetwiseRanking,
                system=prompts.SYSTEM_SETWISE,
                user=prompts.setwise_user_prompt(state, variant_ids),
                budget=budget,
            )
        except StructuredOutputError as exc:
            raise ComparisonError(str(exc)) from exc
        if not isinstance(parsed, SetwiseRanking):
            raise ComparisonError("setwise reasoner returned the wrong schema")
        if set(parsed.ranking) != set(variant_ids) or len(parsed.ranking) != len(variant_ids):
            raise ComparisonError(
                f"setwise ranking mismatch: requested={variant_ids} got={parsed.ranking}"
            )
        if set(parsed.variant_ids) != set(variant_ids):
            parsed = parsed.model_copy(update={"variant_ids": list(variant_ids)})
        return parsed

    async def compare_pair(
        self,
        state: EpisodeEvidenceState,
        left_id: str,
        right_id: str,
        *,
        budget: BudgetManager | None = None,
    ) -> PairPreference:
        try:
            parsed, _trace = await self.client.generate_structured(
                stage="comparative_reasoner_pairwise",
                schema=PairPreference,
                system=prompts.SYSTEM_PAIRWISE,
                user=prompts.pairwise_user_prompt(state, left_id, right_id),
                budget=budget,
            )
        except StructuredOutputError as exc:
            raise ComparisonError(str(exc)) from exc
        if not isinstance(parsed, PairPreference):
            raise ComparisonError("pairwise reasoner returned the wrong schema")
        return parsed.model_copy(update={"left_id": left_id, "right_id": right_id})
