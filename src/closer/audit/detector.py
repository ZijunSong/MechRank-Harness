"""Audit candidate detection from cycles, margins, evidence conflicts, epistasis."""

from __future__ import annotations

from pydantic import BaseModel

from closer.config import AuditConfig
from closer.ranking.graph import PreferenceGraph
from closer.schemas.evidence import NET_EFFECT_RANK_ORDER, VariantEvidenceState
from closer.schemas.ranking import SolverResult


class AuditItem(BaseModel):
    kind: str
    variant_ids: list[str]
    priority: float
    reason: str


def detect_audit_items(
    *,
    variant_ids: list[str],
    evidence: dict[str, VariantEvidenceState],
    graph: PreferenceGraph,
    solver: SolverResult,
    config: AuditConfig,
) -> list[AuditItem]:
    items: list[AuditItem] = []
    items.extend(_cycle_items(graph, config))
    items.extend(_margin_items(solver, config))
    items.extend(_conflict_items(variant_ids, evidence, solver, config))
    items.extend(_epistasis_items(evidence, config))
    items.sort(key=lambda item: (-item.priority, item.kind, ",".join(item.variant_ids)))
    return items


def _cycle_items(graph: PreferenceGraph, config: AuditConfig) -> list[AuditItem]:
    items = []
    for cycle in graph.find_cycles():
        if len(cycle) < 3:
            continue
        items.append(
            AuditItem(
                kind="cycle",
                variant_ids=cycle,
                priority=config.cycle_priority * len(cycle),
                reason="preference cycle: " + " > ".join(cycle + [cycle[0]]),
            )
        )
    return items


def _margin_items(solver: SolverResult, config: AuditConfig) -> list[AuditItem]:
    items = []
    ranking = solver.ranking
    for left, right in zip(ranking, ranking[1:], strict=False):
        margin = abs(solver.scores[left] - solver.scores[right])
        if margin < config.low_margin_threshold:
            priority = config.low_margin_priority * (config.low_margin_threshold - margin)
            items.append(
                AuditItem(
                    kind="low_margin",
                    variant_ids=[left, right],
                    priority=priority,
                    reason=f"adjacent margin {margin:.4f} < {config.low_margin_threshold}",
                )
            )
    return items


def _conflict_items(
    variant_ids: list[str],
    evidence: dict[str, VariantEvidenceState],
    solver: SolverResult,
    config: AuditConfig,
) -> list[AuditItem]:
    rank_pos = {vid: i for i, vid in enumerate(solver.ranking)}
    items = []
    for i, a in enumerate(variant_ids):
        ea = evidence[a]
        if ea.net_confidence < config.evidence_conflict_min_confidence:
            continue
        for b in variant_ids[i + 1 :]:
            eb = evidence[b]
            if eb.net_confidence < config.evidence_conflict_min_confidence:
                continue
            oa = NET_EFFECT_RANK_ORDER[ea.net_assay_effect]
            ob = NET_EFFECT_RANK_ORDER[eb.net_assay_effect]
            if oa == ob:
                continue
            evidence_prefers_a = oa > ob
            ranking_prefers_a = rank_pos[a] < rank_pos[b]
            if evidence_prefers_a != ranking_prefers_a:
                items.append(
                    AuditItem(
                        kind="evidence_conflict",
                        variant_ids=[a, b],
                        priority=config.evidence_conflict_priority
                        * min(ea.net_confidence, eb.net_confidence),
                        reason=(
                            f"{a} net={ea.net_assay_effect} vs {b} net={eb.net_assay_effect} "
                            "conflicts with provisional ranking"
                        ),
                    )
                )
    return items


def _epistasis_items(
    evidence: dict[str, VariantEvidenceState],
    config: AuditConfig,
) -> list[AuditItem]:
    items = []
    for state in evidence.values():
        if state.epistasis_required and state.net_confidence < 0.6:
            items.append(
                AuditItem(
                    kind="epistasis",
                    variant_ids=[state.variant_id],
                    priority=config.epistasis_priority * (1.0 - state.net_confidence),
                    reason=f"{state.variant_id} requires epistasis judgment with low confidence",
                )
            )
    return items
