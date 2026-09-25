"""Render CLOSER rankings into ProteinGym-LLM output format."""

from __future__ import annotations

import json

from closer.schemas.episode import ProteinEpisode

SYSTEM_PROMPT = (
    "You will be given a wild-type protein sequence and a set of mutant sequences. "
    "Rank the mutants by their predicted effect on the assayed property, then output "
    "the ranking in the requested JSON format."
)


def ranking_json_line(ranking: list[str]) -> str:
    return json.dumps({"ranking": ranking}, separators=(",", ":"), ensure_ascii=False)


def render_benchmark_response(ranking: list[str], *, include_preamble: bool = False) -> str:
    line = ranking_json_line(ranking)
    if include_preamble:
        return f"CLOSER completed structured mechanistic ranking.\n{line}"
    return line


def render_compact_prompt(episode: ProteinEpisode, profiles: dict) -> str:
    """Lossless compact view: one full WT, complete substitutions, original assay."""
    from closer.errors import MutationCompilationError
    from closer.mutation.compiler import apply_substitutions

    if not episode.wild_type_sequence or not episode.assay_description:
        raise MutationCompilationError("H-compact requires full WT sequence and original assay text")
    n = len(episode.variants)
    lines = [
        f"**Protein:** {episode.protein_name} ({episode.organism})",
        f"**Assay (what is measured):** {episode.assay_description}",
        "**Higher experimental fitness = HIGHER value of the measured property.**",
        "**Coordinates:** 1-based WT sequence positions, not mature-chain or PDB numbering.",
        "",
        f"**Wild-type sequence ({len(episode.wild_type_sequence)} aa):**",
        episode.wild_type_sequence,
        "",
        f"**{n} candidate mutants (complete substitution sets; reconstructable from WT):**",
    ]
    variant_map = episode.variant_map()
    for variant in episode.variants:
        profile = profiles[variant.variant_id]
        reconstructed = apply_substitutions(episode.wild_type_sequence, profile.substitutions)
        if reconstructed != variant_map[variant.variant_id].sequence:
            raise MutationCompilationError(
                f"{variant.variant_id}: compact substitutions do not reconstruct the mutant sequence"
            )
        lines.append(f"{variant.variant_id}: {profile.shorthand()}")
    lines += [
        "",
        "Each listed substitution set is complete for that candidate. Local windows, if any, "
        "are conveniences only and do not imply that unlisted sites were independently assayed.",
        "",
        f"Rank all {n} mutants from the one you predict has the MOST "
        "favorable effect on the assayed property (highest fitness) to the LEAST. "
        "Reason through the ordering, then on the last line output ONLY the JSON object:",
        '{"ranking": ["M03", "M27", ... all '
        + str(n)
        + " ids, best to worst]}",
    ]
    return "\n".join(lines)


def render_user_prompt(episode: ProteinEpisode) -> str:
    n = len(episode.variants)
    lines = [
        f"**Protein:** {episode.protein_name} ({episode.organism})",
        f"**Assay (what is measured):** {episode.assay_description}",
        "**Higher experimental fitness = HIGHER value of the measured property.**",
        "",
        f"**Wild-type sequence ({len(episode.wild_type_sequence)} aa):**",
        episode.wild_type_sequence,
        "",
        f"**{n} candidate mutant sequences to rank:**",
    ]
    for variant in episode.variants:
        lines.append(f"{variant.variant_id}: {variant.sequence}")
    lines += [
        "",
        f"Rank all {n} mutants from the one you predict has the MOST "
        "favorable effect on the assayed property (highest fitness) to the LEAST. "
        "Reason through the ordering, then on the last line output ONLY the JSON object:",
        '{"ranking": ["M03", "M27", ... all '
        + str(n)
        + " ids, best to worst]}",
    ]
    return "\n".join(lines)
