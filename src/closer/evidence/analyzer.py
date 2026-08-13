"""Batch mechanistic evidence analysis via real LLM calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from closer.errors import EvidenceAnalysisError, StructuredOutputError
from closer.evidence import prompts
from closer.llm.base import LLMClient
from closer.logging import dump_json
from closer.orchestrator.budget import BudgetManager
from closer.schemas.assay import AssayContract
from closer.schemas.episode import ProteinEpisode
from closer.schemas.evidence import VariantEvidenceState
from closer.schemas.mutation import VariantMutationProfile


class VariantEvidenceBatch(BaseModel):
    variants: list[VariantEvidenceState]


class MechanisticAnalyzer:
    def __init__(self, client: LLMClient, *, max_variants_per_call: int = 5, parallelism: int = 8) -> None:
        self.client = client
        self.max_variants_per_call = max_variants_per_call
        self.parallelism = parallelism

    async def analyze_episode(
        self,
        episode: ProteinEpisode,
        contract: AssayContract,
        profiles: dict[str, VariantMutationProfile],
        *,
        budget: BudgetManager | None = None,
        evidence_dir: Path | None = None,
        window: int = 10,
    ) -> dict[str, VariantEvidenceState]:
        ordered = [profiles[vid] for vid in episode.variant_ids()]
        batches = [
            ordered[i : i + self.max_variants_per_call]
            for i in range(0, len(ordered), self.max_variants_per_call)
        ]
        semaphore = asyncio.Semaphore(self.parallelism)

        async def _run(batch: list[VariantMutationProfile]) -> list[VariantEvidenceState]:
            async with semaphore:
                return await self.analyze_batch(
                    episode, contract, batch, budget=budget, window=window
                )

        nested = await asyncio.gather(*[_run(batch) for batch in batches])
        states: dict[str, VariantEvidenceState] = {}
        for group in nested:
            for state in group:
                states[state.variant_id] = state
                if evidence_dir is not None:
                    dump_json(evidence_dir / f"{state.variant_id}.json", state.model_dump(mode="json"))
        missing = sorted(set(episode.variant_ids()) - set(states))
        if missing:
            raise EvidenceAnalysisError(f"missing evidence for variants: {missing}")
        return states

    async def analyze_batch(
        self,
        episode: ProteinEpisode,
        contract: AssayContract,
        profiles: list[VariantMutationProfile],
        *,
        budget: BudgetManager | None = None,
        window: int = 10,
    ) -> list[VariantEvidenceState]:
        requested = [profile.variant_id for profile in profiles]
        try:
            parsed, _trace = await self.client.generate_structured(
                stage="evidence_analyzer",
                schema=VariantEvidenceBatch,
                system=prompts.SYSTEM,
                user=prompts.batch_user_prompt(episode, contract, profiles, window=window),
                budget=budget,
            )
        except StructuredOutputError as exc:
            raise EvidenceAnalysisError(str(exc)) from exc
        if not isinstance(parsed, VariantEvidenceBatch):
            raise EvidenceAnalysisError("evidence analyzer returned the wrong schema")
        got = [item.variant_id for item in parsed.variants]
        if set(got) != set(requested) or len(got) != len(requested):
            raise EvidenceAnalysisError(
                f"evidence batch variant_id mismatch: requested={requested} got={got}"
            )
        by_id = {item.variant_id: item for item in parsed.variants}
        return [by_id[vid] for vid in requested]


def placeholder_uncertain_states(
    profiles: dict[str, VariantMutationProfile],
) -> dict[str, VariantEvidenceState]:
    """Explicit ablation path when evidence analysis is disabled. Not a failure fallback."""
    return {
        vid: VariantEvidenceState(
            variant_id=vid,
            mutation_summary=profile.shorthand(),
            evidence=[],
            epistasis_required=profile.mutation_count > 1,
            dominant_mechanisms=[],
            net_assay_effect="uncertain",
            net_confidence=0.0,
            unresolved_questions=["evidence analyzer disabled by ablation config"],
        )
        for vid, profile in profiles.items()
    }
