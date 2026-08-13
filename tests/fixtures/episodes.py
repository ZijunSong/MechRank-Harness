"""Shared synthetic ProteinGym-LLM-style fixtures. Not evaluation labels."""

from __future__ import annotations

from closer.benchmark.pgllm_renderer import SYSTEM_PROMPT, render_user_prompt
from closer.schemas.episode import ProteinEpisode, VariantInput

WT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQ"


def make_mutant(wt: str, mutations: dict[int, str]) -> str:
    residues = list(wt)
    for pos, aa in mutations.items():
        residues[pos - 1] = aa
    return "".join(residues)


def synthetic_episode(n: int = 10, *, protein: str = "TestProtein") -> ProteinEpisode:
    if n < 2:
        raise ValueError("need at least 2 variants")
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    variants: list[VariantInput] = []
    for i in range(n):
        pos = (i % (len(WT) - 2)) + 2
        to_aa = alphabet[i % len(alphabet)]
        if to_aa == WT[pos - 1]:
            to_aa = alphabet[(i + 1) % len(alphabet)]
        seq = make_mutant(WT, {pos: to_aa})
        variants.append(VariantInput(variant_id=f"M{i + 1:02d}", sequence=seq))
    return ProteinEpisode(
        protein_name=protein,
        organism="Escherichia coli",
        assay_description=(
            "Catalytic activity of the enzyme measured as kcat/KM against a peptide substrate "
            "in a purified-protein kinetic assay."
        ),
        higher_is_better=True,
        wild_type_sequence=WT,
        variants=variants,
        benchmark_name="synthetic-dev",
        system_prompt=SYSTEM_PROMPT,
    )


def ranking_v1_prompt(n: int = 10, *, extra_blank_lines: bool = False) -> str:
    episode = synthetic_episode(n)
    prompt = render_user_prompt(episode)
    if extra_blank_lines:
        prompt = prompt.replace("\n", "\n\n")
    return prompt
