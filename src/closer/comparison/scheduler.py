"""Comparison group scheduler. Evidence ordinals are priors for grouping only."""

from __future__ import annotations

from closer.schemas.evidence import NET_EFFECT_ORDINAL, VariantEvidenceState


def scheduler_prior(state: VariantEvidenceState) -> tuple[int, float, str]:
    ordinal = NET_EFFECT_ORDINAL[state.net_assay_effect]
    return ordinal, state.net_confidence, state.variant_id


def initial_order(states: dict[str, VariantEvidenceState], variant_ids: list[str]) -> list[str]:
    decorated = [(scheduler_prior(states[vid]), vid) for vid in variant_ids]
    decorated.sort(key=lambda item: (-item[0][0], -item[0][1], item[1]))
    return [vid for _prior, vid in decorated]


def chunk_groups(order: list[str], group_size: int) -> list[list[str]]:
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    groups = [order[i : i + group_size] for i in range(0, len(order), group_size)]
    if len(groups) >= 2 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())
    return groups


def overlap_groups(groups: list[list[str]], overlap: int) -> list[list[str]]:
    if overlap < 1:
        return []
    crossed: list[list[str]] = []
    for left, right in zip(groups, groups[1:], strict=False):
        tail = left[-overlap:]
        head_n = max(2, min(len(right), overlap + 1))
        head = right[:head_n]
        merged: list[str] = []
        for vid in tail + head:
            if vid not in merged:
                merged.append(vid)
        if len(merged) >= 2:
            crossed.append(merged)
    return crossed


def bridge_pairs(components: list[list[str]]) -> list[tuple[str, str]]:
    if len(components) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    representatives = [sorted(component)[0] for component in components]
    for left, right in zip(representatives, representatives[1:], strict=False):
        pairs.append((left, right))
    return pairs


class ComparisonScheduler:
    def __init__(self, group_size: int = 5, overlap: int = 2) -> None:
        self.group_size = group_size
        self.overlap = overlap

    def initial_groups(self, states: dict[str, VariantEvidenceState], variant_ids: list[str]) -> list[list[str]]:
        order = initial_order(states, variant_ids)
        return chunk_groups(order, self.group_size)

    def boundary_groups(self, groups: list[list[str]]) -> list[list[str]]:
        return overlap_groups(groups, self.overlap)
