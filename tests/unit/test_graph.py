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
