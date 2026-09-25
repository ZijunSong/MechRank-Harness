"""Validate and normalize pairwise / setwise comparison objects.

Returned IDs must stay bound to the requested comparison. Silent rewrites that
keep winner while swapping endpoints invert the model's actual judgment.
"""

from __future__ import annotations

from closer.errors import ComparisonError
from closer.schemas.comparison import PairPreference, SetwiseRanking

_WINNER_FLIP = {
    "left": "right",
    "right": "left",
    "tie": "tie",
    "uncertain": "uncertain",
}


def canonical_pair(left_id: str, right_id: str) -> tuple[str, str]:
    return tuple(sorted((left_id, right_id)))


def winner_variant_id(pref: PairPreference) -> str | None:
    if pref.winner == "left":
        return pref.left_id
    if pref.winner == "right":
        return pref.right_id
    return None


def normalize_requested_pair(pref: PairPreference, left_id: str, right_id: str) -> PairPreference:
    """Keep winner bound to the compared variants; flip if the model swapped sides."""
    if left_id == right_id or pref.left_id == pref.right_id:
        raise ComparisonError("self comparison")
    if (pref.left_id, pref.right_id) == (left_id, right_id):
        return pref
    if (pref.left_id, pref.right_id) == (right_id, left_id):
        return pref.model_copy(
            update={
                "left_id": left_id,
                "right_id": right_id,
                "winner": _WINNER_FLIP[pref.winner],
            }
        )
    raise ComparisonError("returned pair differs from requested pair")


def normalize_pair_from_allowed_set(pref: PairPreference, allowed_ids: list[str]) -> PairPreference:
    """Accept a model-chosen pair only if both distinct IDs are in the allowed set."""
    allowed = set(allowed_ids)
    if pref.left_id == pref.right_id:
        raise ComparisonError("self comparison")
    if pref.left_id not in allowed or pref.right_id not in allowed:
        raise ComparisonError("returned pair differs from requested pair")
    return pref


def _dedupe_explicit_pairs(prefs: list[PairPreference]) -> list[PairPreference]:
    seen: dict[tuple[str, str], PairPreference] = {}
    for pref in prefs:
        key = canonical_pair(pref.left_id, pref.right_id)
        previous = seen.get(key)
        if previous is None:
            seen[key] = pref
            continue
        prev_winner = winner_variant_id(previous)
        new_winner = winner_variant_id(pref)
        same_non_directional = previous.winner == pref.winner and previous.winner in {
            "tie",
            "uncertain",
        }
        if same_non_directional or (prev_winner is not None and prev_winner == new_winner):
            continue
        raise ComparisonError(
            f"conflicting preferences for pair {key[0]}/{key[1]}: "
            f"{previous.winner} vs {pref.winner}"
        )
    return list(seen.values())


def validate_setwise_pair_preferences(
    prefs: list[PairPreference],
    group_ids: list[str],
) -> list[PairPreference]:
    allowed = set(group_ids)
    validated: list[PairPreference] = []
    for pref in prefs:
        if pref.left_id == pref.right_id:
            raise ComparisonError("self comparison")
        if pref.left_id not in allowed or pref.right_id not in allowed:
            raise ComparisonError("returned pair differs from requested pair")
        validated.append(pref)
    return _dedupe_explicit_pairs(validated)


def validate_setwise_ranking(result: SetwiseRanking, requested_ids: list[str]) -> SetwiseRanking:
    if set(result.ranking) != set(requested_ids) or len(result.ranking) != len(requested_ids):
        raise ComparisonError(f"setwise ranking mismatch: requested={requested_ids} got={result.ranking}")
    prefs = validate_setwise_pair_preferences(result.pair_preferences, requested_ids)
    updates: dict[str, object] = {"pair_preferences": prefs}
    if set(result.variant_ids) != set(requested_ids):
        updates["variant_ids"] = list(requested_ids)
    return result.model_copy(update=updates)
