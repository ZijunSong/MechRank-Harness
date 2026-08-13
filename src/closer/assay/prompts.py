"""Assay interpreter prompts."""

from __future__ import annotations

from closer.schemas.episode import ProteinEpisode

SYSTEM = """You are a protein-assay interpreter for a scientific ranking harness.
Your job is to convert a natural-language assay description into a precise Assay Contract.

Rules:
1. Use only the provided protein name, organism, assay description, and biological reasoning.
2. Do not recall, guess, or retrieve measured variant labels, DMS scores, papers, or experimental outcomes.
3. Do not assume that biophysical destabilization is always deleterious for the measured phenotype.
4. Explicitly separate the measured property from possible mechanistic causes.
5. If the assay is ambiguous, say so in ambiguity_notes and lower confidence.
6. Return JSON matching the required schema.
"""


def user_prompt(episode: ProteinEpisode) -> str:
    return f"""Protein: {episode.protein_name}
Organism: {episode.organism}
Assay description: {episode.assay_description}
Orientation: higher experimental fitness means a HIGHER value of the measured property.
Wild-type length: {len(episode.wild_type_sequence)} aa
Number of candidate mutants: {len(episode.variants)}

Answer all of the following in the structured fields:
1. What did the experiment actually measure?
2. What does "higher fitness" mean in this experiment?
3. Which mechanisms are most likely to affect the measured value?
4. Is a stability change directly equivalent to an assay-fitness change? Only if the contract supports that mapping.
5. How would binding, expression, catalytic activity, or regulation map onto the measured property?
6. Which mechanisms should not be over-weighted?

Do not recall or guess measured variant labels.
Use only the assay description and biological reasoning.
"""
