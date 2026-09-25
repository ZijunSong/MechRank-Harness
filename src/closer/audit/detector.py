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
    items.extend(_cycle_items(graph, solver, config))
    items.extend(_margin_items(solver, config))
    items.extend(_conflict_items(variant_ids, evidence, solver, config))
    items.extend(_epistasis_items(evidence, config))
    items.sort(key=lambda item: (-item.priority, item.kind, ",".join(item.variant_ids)))
    return items


def select_pair_audit_items(items: list[AuditItem], config: AuditConfig) -> list[AuditItem]:
    """Keep pair-capable items, skip single-variant epistasis, merge duplicate pairs."""
    eligible = [
        item
        for item in items
        if item.priority >= config.min_priority
        and item.kind != "epistasis"
        and len(item.variant_ids) >= 2
    ]
    merged: dict[tuple[str, str], AuditItem] = {}
    leftovers: list[AuditItem] = []
    for item in eligible:
        if len(item.variant_ids) == 2:
            key = tuple(sorted(item.variant_ids))
            previous = merged.get(key)
            if previous is None:
                merged[key] = item
                continue
            higher = item if item.priority > previous.priority else previous
            lower = previous if higher is item else item
            merged[key] = higher.model_copy(
                update={"reason": f"{higher.reason}; {lower.reason}"}
            )
        else:
            leftovers.append(item)
    selected = list(merged.values()) + leftovers
    selected.sort(key=lambda item: (-item.priority, item.kind, ",".join(item.variant_ids)))
    return selected[: config.max_items_per_round]


def select_cycle_pair(cycle: list[str], scores: dict[str, float]) -> list[str]:
    """Bind a cycle audit to one adjacent pair, preferring the smallest score gap."""
    if len(cycle) < 2:
        return list(cycle)
    best = [cycle[0], cycle[1]]
    best_margin = float("inf")
    for index, left in enumerate(cycle):
        right = cycle[(index + 1) % len(cycle)]
        margin = abs(scores.get(left, 0.0) - scores.get(right, 0.0))
        if margin < best_margin:
            best_margin = margin
            best = [left, right]
    return best


def _cycle_items(graph: PreferenceGraph, solver: SolverResult, config: AuditConfig) -> list[AuditItem]:
    items = []
    seen_pairs: set[tuple[str, str]] = set()
    for cycle in graph.find_cycles():
        if len(cycle) < 3:
            continue
        pair = select_cycle_pair(cycle, solver.scores)
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        items.append(
            AuditItem(
                kind="cycle",
                variant_ids=pair,
                priority=min(1.0, config.cycle_priority),
                reason="preference cycle: " + " > ".join(cycle + [cycle[0]]),
            )
        )
    return items


def _margin_items(solver: SolverResult, config: AuditConfig) -> list[AuditItem]:
    items = []
    ranking = solver.ranking
    threshold = config.low_margin_threshold
    for left, right in zip(ranking, ranking[1:], strict=False):
        margin = abs(solver.scores[left] - solver.scores[right])
        if margin < threshold:
            priority = config.low_margin_priority * max(0.0, 1.0 - margin / threshold)
            items.append(
                AuditItem(
                    kind="low_margin",
                    variant_ids=[left, right],
                    priority=priority,
                    reason=f"adjacent margin {margin:.4f} < {threshold}",
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
        if ea.net_assay_effect == "uncertain":
            continue
        for b in variant_ids[i + 1 :]:
            eb = evidence[b]
            if eb.net_confidence < config.evidence_conflict_min_confidence:
                continue
            if eb.net_assay_effect == "uncertain":
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
