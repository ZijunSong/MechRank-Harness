"""Deterministic WT vs mutant substitution compiler. No LLM, no indel alignment."""

from __future__ import annotations

from closer.errors import MutationCompilationError
from closer.schemas.episode import AMINO_ACIDS, ProteinEpisode, validate_amino_acid_sequence
from closer.schemas.mutation import ResidueSubstitution, VariantMutationProfile


def compile_variant_mutations(
    wild_type_sequence: str,
    variant_id: str,
    mutant_sequence: str,
) -> VariantMutationProfile:
    try:
        wt = validate_amino_acid_sequence(wild_type_sequence, field_name="wild_type_sequence")
        mutant = validate_amino_acid_sequence(mutant_sequence, field_name=variant_id)
    except ValueError as exc:
        raise MutationCompilationError(str(exc)) from exc

    if len(wt) != len(mutant):
        raise MutationCompilationError(
            f"{variant_id}: Track A substitution benchmark requires equal length; "
            f"WT={len(wt)} mutant={len(mutant)}. Indel alignment is not performed."
        )

    substitutions: list[ResidueSubstitution] = []
    for idx, (from_aa, to_aa) in enumerate(zip(wt, mutant, strict=True)):
        if from_aa not in AMINO_ACIDS:
            raise MutationCompilationError(
                f"{variant_id}: invalid WT residue {from_aa!r} at position {idx + 1}"
            )
        if from_aa != to_aa:
            substitutions.append(
                ResidueSubstitution(position_1based=idx + 1, from_aa=from_aa, to_aa=to_aa)
            )

    reconstructed = apply_substitutions(wt, substitutions)
    if reconstructed != mutant:
        raise MutationCompilationError(
            f"{variant_id}: re-applying substitutions did not reconstruct the mutant sequence"
        )

    return VariantMutationProfile(
        variant_id=variant_id,
        substitutions=substitutions,
        mutation_count=len(substitutions),
    )


def apply_substitutions(wild_type_sequence: str, substitutions: list[ResidueSubstitution]) -> str:
    residues = list(wild_type_sequence)
    for sub in substitutions:
        pos = sub.position_1based - 1
        if pos < 0 or pos >= len(residues):
            raise MutationCompilationError(
                f"substitution position {sub.position_1based} is outside WT length {len(residues)}"
            )
        if residues[pos] != sub.from_aa:
            raise MutationCompilationError(
                f"WT residue at {sub.position_1based} is {residues[pos]}, expected {sub.from_aa}"
            )
        residues[pos] = sub.to_aa
    return "".join(residues)


def compile_episode_mutations(episode: ProteinEpisode) -> dict[str, VariantMutationProfile]:
    return {
        variant.variant_id: compile_variant_mutations(
            episode.wild_type_sequence, variant.variant_id, variant.sequence
        )
        for variant in episode.variants
    }


def local_context(wild_type_sequence: str, position_1based: int, window: int = 10) -> str:
    idx = position_1based - 1
    start = max(0, idx - window)
    end = min(len(wild_type_sequence), idx + window + 1)
    return wild_type_sequence[start:end]
