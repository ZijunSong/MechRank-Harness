"""Assay interpreter: assay description -> AssayContract via a real LLM call."""

from __future__ import annotations

from closer.assay import prompts
from closer.errors import AssayInterpretationError, StructuredOutputError
from closer.llm.base import LLMClient
from closer.orchestrator.budget import BudgetManager
from closer.schemas.assay import AssayContract
from closer.schemas.episode import ProteinEpisode


class AssayInterpreter:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def interpret(
        self,
        episode: ProteinEpisode,
        *,
        budget: BudgetManager | None = None,
    ) -> AssayContract:
        try:
            parsed, _trace = await self.client.generate_structured(
                stage="assay_interpreter",
                schema=AssayContract,
                system=prompts.SYSTEM,
                user=prompts.user_prompt(episode),
                budget=budget,
            )
        except StructuredOutputError as exc:
            raise AssayInterpretationError(str(exc)) from exc
        if not isinstance(parsed, AssayContract):
            raise AssayInterpretationError("assay interpreter returned the wrong schema")
        if parsed.higher_is_better != episode.higher_is_better:
            # Keep the benchmark orientation authoritative.
            parsed = parsed.model_copy(update={"higher_is_better": episode.higher_is_better})
        return parsed


def deterministic_ablation_contract(episode: ProteinEpisode) -> AssayContract:
    """Explicit ablation path when assay interpretation is disabled. Not a failure fallback."""
    return AssayContract(
        measured_property=episode.assay_description,
        biological_context=f"{episode.protein_name} ({episode.organism})",
        higher_is_better=episode.higher_is_better,
        direct_effect_interpretation=(
            "Assay interpreter is disabled by ablation config. Use the stated measured "
            "property directly; do not infer additional phenotype mappings."
        ),
        likely_causal_chain=["sequence change", "measured property change"],
        mechanism_relevance=[],
        ambiguity_notes=["assay interpreter disabled by ablation config"],
        confidence=0.0,
    )
