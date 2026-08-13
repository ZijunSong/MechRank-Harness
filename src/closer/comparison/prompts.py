"""Comparison prompts."""

from __future__ import annotations

import json

from closer.evidence.state import EpisodeEvidenceState
from closer.schemas.mutation import VariantMutationProfile

SYSTEM_SETWISE = """You are a comparative protein-variant reasoner.
You receive an Assay Contract and compressed mechanistic evidence for a small group of variants.
Rank ONLY this group from most favorable assay effect to least favorable.

Rules:
1. Do not recall experimental labels.
2. Prefer assay-conditioned evidence over generic stability heuristics.
3. Emit pair_preferences for every adjacent pair in your ranking, and any pair you consider close.
4. If two variants are indistinguishable given the evidence, use winner=tie and say so.
5. ranking must contain each requested variant_id exactly once.
"""

SYSTEM_PAIRWISE = """You are a comparative protein-variant reasoner.
Compare exactly two variants under the given Assay Contract.
Choose winner=left, right, tie, or uncertain.
Do not recall experimental labels.
"""


def _variant_payload(state: EpisodeEvidenceState, variant_id: str) -> dict:
    profile: VariantMutationProfile = state.mutation_profiles[variant_id]
    evidence = state.get_variant(variant_id)
    return {
        "variant_id": variant_id,
        "mutation_summary": profile.shorthand(),
        "substitutions": [item.model_dump() for item in profile.substitutions],
        "evidence": evidence.model_dump(mode="json"),
    }


def setwise_user_prompt(state: EpisodeEvidenceState, variant_ids: list[str]) -> str:
    payload = {
        "assay_contract": state.assay_contract.model_dump(mode="json"),
        "variants": [_variant_payload(state, vid) for vid in variant_ids],
    }
    return (
        "Rank these variants from most favorable assay effect to least favorable. "
        "Do not consider variants outside this group.\n"
        f"{json.dumps(payload, indent=2)}"
    )


def pairwise_user_prompt(state: EpisodeEvidenceState, left_id: str, right_id: str) -> str:
    payload = {
        "assay_contract": state.assay_contract.model_dump(mode="json"),
        "left": _variant_payload(state, left_id),
        "right": _variant_payload(state, right_id),
    }
    return "Compare left vs right for the measured assay property.\n" + json.dumps(payload, indent=2)
