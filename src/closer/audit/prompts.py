"""Adaptive auditor prompts."""

from __future__ import annotations

import json

from closer.audit.detector import AuditItem
from closer.evidence.state import EpisodeEvidenceState
from closer.ranking.graph import PreferenceGraph
from closer.schemas.ranking import SolverResult

SYSTEM = """You are an adaptive auditor for a protein-variant ranking harness.
You are shown an Assay Contract, structured evidence, previous comparisons, and a conflict.
Your job:
1. Identify the most likely incorrect premise.
2. Re-compare the involved variants.
3. State whether you overturn a previous preference.
4. Give a new confidence.
5. Explain the repair.

Do not recall experimental labels. Return JSON matching PairPreference.
If exactly two variants are listed, compare those two and keep their ids in left_id and right_id.
Do not introduce any other variant id.
If more than two variants are involved, compare two distinct ids from that set only
and put them in left_id and right_id.
"""


def user_prompt(
    state: EpisodeEvidenceState,
    item: AuditItem,
    graph: PreferenceGraph,
    solver: SolverResult,
) -> str:
    involved = item.variant_ids
    previous = [
        obs.model_dump(mode="json")
        for obs in graph.observations
        if obs.left_id in involved and obs.right_id in involved
    ]
    payload = {
        "assay_contract": state.assay_contract.model_dump(mode="json"),
        "conflict": item.model_dump(mode="json"),
        "provisional_ranking": solver.ranking,
        "provisional_scores": {vid: solver.scores[vid] for vid in involved if vid in solver.scores},
        "evidence": {
            vid: state.get_variant(vid).model_dump(mode="json")
            for vid in involved
            if vid in state.variant_evidence
        },
        "previous_comparisons": previous,
    }
    return json.dumps(payload, indent=2)
