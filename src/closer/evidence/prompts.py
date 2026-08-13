"""Mechanistic evidence prompts."""

from __future__ import annotations

import json

from closer.mutation.compiler import local_context
from closer.schemas.assay import AssayContract
from closer.schemas.episode import ProteinEpisode
from closer.schemas.mutation import VariantMutationProfile

SYSTEM = """You are a mechanistic evidence analyst for protein variant ranking.
For each variant you must keep two steps separate:
A. mutation -> mechanistic consequence
B. mechanistic consequence -> assay consequence, using ONLY the provided Assay Contract.

Rules:
1. Do not recall or guess experimental labels, DMS scores, or published rankings.
2. Do not use the heuristic "destabilizing therefore always bad" unless the Assay Contract
   explicitly supports that mapping for the measured property.
3. For multi-mutants you MUST judge additive / antagonistic / synergistic / unknown epistasis.
   Do not just add single-mutant effects.
4. If evidence is insufficient, mark direction=uncertain and lower confidence.
5. Return one structured record per requested variant_id. You must return exactly the
   requested variant_id set, no extras, no omissions.
"""


def batch_user_prompt(
    episode: ProteinEpisode,
    contract: AssayContract,
    profiles: list[VariantMutationProfile],
    *,
    window: int = 10,
) -> str:
    items = []
    for profile in profiles:
        contexts = []
        for sub in profile.substitutions:
            ctx = local_context(episode.wild_type_sequence, sub.position_1based, window)
            contexts.append(
                {
                    "substitution": f"{sub.from_aa}{sub.position_1based}{sub.to_aa}",
                    "local_wt_context": ctx,
                }
            )
        items.append(
            {
                "variant_id": profile.variant_id,
                "mutation_summary": profile.shorthand(),
                "mutation_count": profile.mutation_count,
                "substitutions": [s.model_dump() for s in profile.substitutions],
                "local_contexts": contexts,
            }
        )
    return (
        f"Protein: {episode.protein_name} ({episode.organism})\n"
        f"Assay contract JSON:\n{contract.model_dump_json(indent=2)}\n\n"
        "Analyze these variants. Do not rank them globally.\n"
        f"{json.dumps(items, indent=2)}\n"
    )
