"""Closure consistency and other paper diagnostics. Not used for PG-LLM official scoring."""

from __future__ import annotations

from closer.config import AuditConfig
from closer.ranking.graph import PreferenceGraph
from closer.schemas.evidence import NET_EFFECT_RANK_ORDER, VariantEvidenceState
from closer.schemas.ranking import SolverResult


def evidence_ranking_conflict_rate(
    evidence: dict[str, VariantEvidenceState],
    ranking: list[str],
    *,
    min_confidence: float = 0.75,
) -> dict:
    rank_pos = {vid: i for i, vid in enumerate(ranking)}
    ids = list(evidence)
    confident_pairs = 0
    contradictions = 0
    for i, a in enumerate(ids):
        ea = evidence[a]
        if ea.net_confidence < min_confidence:
            continue
        for b in ids[i + 1 :]:
            eb = evidence[b]
            if eb.net_confidence < min_confidence:
                continue
            oa = NET_EFFECT_RANK_ORDER[ea.net_assay_effect]
            ob = NET_EFFECT_RANK_ORDER[eb.net_assay_effect]
            if oa == ob:
                continue
            confident_pairs += 1
            evidence_prefers_a = oa > ob
            ranking_prefers_a = rank_pos[a] < rank_pos[b]
            if evidence_prefers_a != ranking_prefers_a:
                contradictions += 1
    consistency = 1.0 if confident_pairs == 0 else 1.0 - contradictions / confident_pairs
    return {
        "confident_pairs": confident_pairs,
        "contradictory_confident_pairs": contradictions,
        "closure_consistency": consistency,
    }


def collect_diagnostics(
    *,
    evidence: dict[str, VariantEvidenceState],
    graph: PreferenceGraph,
    before: SolverResult,
    after: SolverResult,
    audit_history: list[dict],
    usage: dict,
    cycle_count_before: int,
    config: AuditConfig,
) -> dict:
    closure = evidence_ranking_conflict_rate(evidence, after.ranking)
    edges = graph.informative_edges()
    audit_edges = [edge for edge in edges if edge.source == "audit"]
    confidences = [obs.confidence for obs in graph.observations if obs.winner in {"left", "right"}]
    low_margin = 0
    for left, right in zip(after.ranking, after.ranking[1:], strict=False):
        if abs(after.scores[left] - after.scores[right]) < config.low_margin_threshold:
            low_margin += 1
    uncertain = sum(
        1 for state in evidence.values() if state.net_assay_effect == "uncertain" or state.net_confidence < 0.5
    )
    multi = sum(1 for state in evidence.values() if state.epistasis_required)
    ranking_changes = [row.get("ranking_changed") for row in audit_history]
    return {
        **closure,
        "cycle_count_before_audit": cycle_count_before,
        "cycle_count_after_audit": len(graph.find_cycles()),
        "audit_edge_fraction": (len(audit_edges) / len(edges)) if edges else 0.0,
        "ranking_changes_per_audit_round": ranking_changes,
        "mean_pair_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "low_margin_pair_count": low_margin,
        "evidence_uncertain_fraction": uncertain / max(1, len(evidence)),
        "multi_mutant_fraction": multi / max(1, len(evidence)),
        "n_observations": len(graph.observations),
        "n_audit_rounds": len(audit_history),
        **usage,
    }
