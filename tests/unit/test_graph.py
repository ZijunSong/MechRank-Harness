import pytest

from closer.ranking.graph import PreferenceGraph
from closer.schemas.comparison import PairPreference, PreferenceObservation, SetwiseRanking


def _obs(left: str, right: str, winner: str, conf: float = 0.8, source: str = "pairwise", round_index: int = 0):
    return PreferenceObservation(
        observation_id=f"{left}{right}{round_index}{winner}",
        left_id=left,
        right_id=right,
        winner=winner,  # type: ignore[arg-type]
        confidence=conf,
        source=source,  # type: ignore[arg-type]
        round_index=round_index,
        rationale="test",
    )


def test_observations_are_append_only():
    g = PreferenceGraph()
    g.add_nodes(["A", "B"])
    g.add_observation(_obs("A", "B", "left", 0.7, round_index=0))
    g.add_observation(_obs("A", "B", "right", 0.9, round_index=1))
    assert len(g.observations) == 2
    agg = g.aggregate_pair("A", "B")
    assert agg["n_observations"] == 2


def test_find_cycle():
    g = PreferenceGraph()
    g.add_nodes(["A", "B", "C"])
    g.add_observation(_obs("A", "B", "left"))
    g.add_observation(_obs("B", "C", "left"))
    g.add_observation(_obs("C", "A", "left"))
    cycles = g.find_cycles()
    assert cycles


def test_disconnected_components():
    g = PreferenceGraph()
    g.add_nodes(["A", "B", "C", "D"])
    g.add_observation(_obs("A", "B", "left"))
    g.add_observation(_obs("C", "D", "left"))
    comps = g.connected_components()
    assert len(comps) == 2
    assert g.is_weakly_connected() is False


def test_setwise_adds_adjacent_edges():
    g = PreferenceGraph()
    g.add_nodes(["M01", "M02", "M03"])
    result = SetwiseRanking(
        variant_ids=["M01", "M02", "M03"],
        ranking=["M03", "M01", "M02"],
        pair_preferences=[
            PairPreference(
                left_id="M03",
                right_id="M01",
                winner="left",
                confidence=0.8,
                key_evidence=["e"],
                contradiction_with_previous_evidence=False,
                rationale="r",
            )
        ],
        confidence=0.7,
    )
    g.add_setwise(result, round_index=0)
    assert g.is_weakly_connected()


def _pair(left: str, right: str, winner: str) -> PairPreference:
    return PairPreference(
        left_id=left,
        right_id=right,
        winner=winner,  # type: ignore[arg-type]
        confidence=0.8,
        key_evidence=["e"],
        contradiction_with_previous_evidence=False,
        rationale="r",
    )


def test_setwise_five_items_four_explicit_pairs_are_not_doubled():
    ids = [f"M0{i}" for i in range(1, 6)]
    ranking = list(ids)
    prefs = [_pair(ranking[i], ranking[i + 1], "left") for i in range(4)]
    result = SetwiseRanking(
        variant_ids=ids,
        ranking=ranking,
        pair_preferences=prefs,
        confidence=0.7,
    )
    g = PreferenceGraph()
    g.add_nodes(ids)
    g.add_setwise(result, round_index=0)
    assert len(g.observations) == 4
    event_ids = {obs.comparison_event_id for obs in g.observations}
    assert len(event_ids) == 1
    assert all(not obs.derived_from_order for obs in g.observations)
    assert len(g.informative_edges()) == 4


def test_explicit_tie_does_not_create_decisive_edge():
    result = SetwiseRanking(
        variant_ids=["M01", "M02"],
        ranking=["M01", "M02"],
        pair_preferences=[_pair("M01", "M02", "tie")],
        confidence=0.6,
    )
    g = PreferenceGraph()
    g.add_nodes(["M01", "M02"])
    g.add_setwise(result, round_index=0)
    assert len(g.observations) == 1
    assert g.observations[0].winner == "tie"
    assert g.informative_edges() == []


def test_explicit_uncertain_does_not_create_decisive_edge():
    result = SetwiseRanking(
        variant_ids=["M01", "M02"],
        ranking=["M01", "M02"],
        pair_preferences=[_pair("M01", "M02", "uncertain")],
        confidence=0.4,
    )
    g = PreferenceGraph()
    g.add_nodes(["M01", "M02"])
    g.add_setwise(result, round_index=0)
    assert g.informative_edges() == []


def test_unknown_endpoint_cannot_enter_graph():
    from closer.errors import ComparisonError

    g = PreferenceGraph()
    g.add_nodes(["M01", "M02"])
    with pytest.raises(ComparisonError, match="unknown variant"):
        g.add_observation(_obs("M01", "GHOST", "left"))


def test_audit_revision_removes_old_cycle_from_effective_graph():
    g = PreferenceGraph()
    g.add_nodes(["A", "B", "C"])
    g.add_observation(_obs("A", "B", "left"))
    g.add_observation(_obs("B", "C", "left"))
    g.add_observation(_obs("C", "A", "left"))
    assert g.find_cycles()
    g.add_observation(_obs("A", "C", "left", source="audit", round_index=1))
    assert not any(len(cycle) >= 3 for cycle in g.find_cycles())
    assert any(obs.status == "superseded" for obs in g.raw_observations())
    assert any(obs.source == "audit" and obs.status == "active" for obs in g.active_observations())


def test_independent_setwise_events_keep_distinct_ids():
    ids = ["M01", "M02"]
    result = SetwiseRanking(
        variant_ids=ids,
        ranking=ids,
        pair_preferences=[_pair("M01", "M02", "left")],
        confidence=0.7,
    )
    g = PreferenceGraph()
    g.add_nodes(ids)
    g.add_setwise(result, round_index=0)
    g.add_setwise(result, round_index=1)
    event_ids = {obs.comparison_event_id for obs in g.observations}
    assert len(g.observations) == 2
    assert len(event_ids) == 2
