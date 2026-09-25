import pytest

from closer.comparison.reasoner import ComparativeReasoner
from closer.comparison.validation import (
    normalize_pair_from_allowed_set,
    normalize_requested_pair,
    validate_setwise_pair_preferences,
)
from closer.errors import ComparisonError
from closer.schemas.comparison import PairPreference


def _pref(left: str, right: str, winner: str) -> PairPreference:
    return PairPreference(
        left_id=left,
        right_id=right,
        winner=winner,  # type: ignore[arg-type]
        confidence=0.8,
        key_evidence=["e"],
        contradiction_with_previous_evidence=False,
        rationale="r",
    )


def test_normalize_keeps_original_direction():
    pref = normalize_requested_pair(_pref("M01", "M02", "left"), "M01", "M02")
    assert pref.left_id == "M01"
    assert pref.right_id == "M02"
    assert pref.winner == "left"


def test_normalize_flips_reversed_pair():
    pref = normalize_requested_pair(_pref("M02", "M01", "left"), "M01", "M02")
    assert pref.left_id == "M01"
    assert pref.right_id == "M02"
    assert pref.winner == "right"


def test_normalize_preserves_tie_and_uncertain():
    tie = normalize_requested_pair(_pref("M02", "M01", "tie"), "M01", "M02")
    assert tie.winner == "tie"
    uncertain = normalize_requested_pair(_pref("M02", "M01", "uncertain"), "M01", "M02")
    assert uncertain.winner == "uncertain"


def test_normalize_rejects_third_party_id():
    with pytest.raises(ComparisonError, match="differs from requested pair"):
        normalize_requested_pair(_pref("M02", "M03", "left"), "M01", "M02")


def test_normalize_rejects_self_comparison():
    with pytest.raises(ComparisonError, match="self comparison"):
        normalize_requested_pair(_pref("M01", "M01", "left"), "M01", "M02")
    with pytest.raises(ComparisonError, match="self comparison"):
        normalize_requested_pair(_pref("M01", "M02", "left"), "M01", "M01")


def test_cycle_choice_keeps_non_first_pair():
    pref = normalize_pair_from_allowed_set(_pref("M02", "M03", "left"), ["M01", "M02", "M03"])
    assert pref.left_id == "M02"
    assert pref.right_id == "M03"
    assert pref.winner == "left"


def test_cycle_choice_rejects_ghost():
    with pytest.raises(ComparisonError, match="differs from requested pair"):
        normalize_pair_from_allowed_set(_pref("M01", "GHOST", "left"), ["M01", "M02", "M03"])


def test_setwise_pairs_reject_unknown_and_self():
    with pytest.raises(ComparisonError, match="self comparison"):
        validate_setwise_pair_preferences([_pref("M01", "M01", "left")], ["M01", "M02"])
    with pytest.raises(ComparisonError, match="differs from requested pair"):
        validate_setwise_pair_preferences([_pref("M01", "GHOST", "left")], ["M01", "M02"])


def test_setwise_pairs_dedupe_same_winner():
    prefs = validate_setwise_pair_preferences(
        [_pref("M01", "M02", "left"), _pref("M02", "M01", "right")],
        ["M01", "M02", "M03"],
    )
    assert len(prefs) == 1
    assert prefs[0].winner == "left"


def test_setwise_pairs_conflict_raises():
    with pytest.raises(ComparisonError, match="conflicting"):
        validate_setwise_pair_preferences(
            [_pref("M01", "M02", "left"), _pref("M01", "M02", "right")],
            ["M01", "M02"],
        )


@pytest.mark.asyncio
async def test_reasoner_flips_reversed_pair_ids():
    class _Client:
        async def generate_structured(self, **_kwargs):
            return _pref("M02", "M01", "left"), None

    from unittest.mock import MagicMock

    reasoner = ComparativeReasoner(_Client())  # type: ignore[arg-type]
    state = MagicMock()
    state.assay_contract.model_dump.return_value = {}
    profile = MagicMock()
    profile.shorthand.return_value = "A1V"
    profile.substitutions = []
    state.mutation_profiles = {"M01": profile, "M02": profile}
    state.get_variant.return_value.model_dump.return_value = {}
    out = await reasoner.compare_pair(state, "M01", "M02")
    assert out.left_id == "M01"
    assert out.right_id == "M02"
    assert out.winner == "right"
