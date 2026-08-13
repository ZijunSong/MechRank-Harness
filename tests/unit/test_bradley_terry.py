from closer.config import RankingConfig
from closer.ranking.bradley_terry import fit_bradley_terry, observation_weight
from closer.ranking.graph import PreferenceGraph
from closer.schemas.comparison import PreferenceObservation


def _edge(winner: str, loser: str, conf: float = 0.8, n: int = 0) -> PreferenceObservation:
    return PreferenceObservation(
        observation_id=f"{winner}>{loser}:{n}",
        left_id=winner,
        right_id=loser,
        winner="left",
        confidence=conf,
        source="pairwise",
        round_index=0,
        rationale="synthetic",
    )


def _graph(ids: list[str], pairs: list[tuple[str, str, float]]) -> PreferenceGraph:
    g = PreferenceGraph()
    g.add_nodes(ids)
    for i, (winner, loser, conf) in enumerate(pairs):
        g.add_observation(_edge(winner, loser, conf, i))
    return g


def test_strict_total_order():
    ids = ["A", "B", "C", "D"]
    pairs = [("A", "B", 0.9), ("B", "C", 0.9), ("C", "D", 0.9), ("A", "C", 0.8), ("B", "D", 0.8)]
    result = fit_bradley_terry(ids, _graph(ids, pairs), RankingConfig())
    assert result.ranking == ["A", "B", "C", "D"]
    assert result.scores["A"] > result.scores["D"]


def test_partial_graph():
    ids = ["A", "B", "C"]
    result = fit_bradley_terry(ids, _graph(ids, [("A", "B", 0.8)]), RankingConfig())
    assert result.ranking[0] in {"A", "C"}
    assert result.ranking[-1] in {"B", "C"}


def test_cycle_is_stable():
    ids = ["A", "B", "C"]
    g = _graph(ids, [("A", "B", 0.8), ("B", "C", 0.8), ("C", "A", 0.8)])
    r1 = fit_bradley_terry(ids, g, RankingConfig())
    r2 = fit_bradley_terry(ids, g, RankingConfig())
    assert r1.ranking == r2.ranking
    assert set(r1.ranking) == set(ids)


def test_contradictory_edges_weight_by_confidence():
    ids = ["A", "B"]
    g = _graph(ids, [("A", "B", 0.9), ("B", "A", 0.6), ("A", "B", 0.85)])
    result = fit_bradley_terry(ids, g, RankingConfig())
    assert result.ranking[0] == "A"


def test_duplicate_observations_are_all_used():
    ids = ["A", "B"]
    g = PreferenceGraph()
    g.add_nodes(ids)
    for i in range(5):
        g.add_observation(_edge("A", "B", 0.8, i))
    result = fit_bradley_terry(ids, g, RankingConfig())
    assert result.n_informative_edges == 5
    assert result.ranking == ["A", "B"]


def test_disconnected_graph_still_ranks_all():
    ids = ["A", "B", "C", "D"]
    g = _graph(ids, [("A", "B", 0.9), ("C", "D", 0.9)])
    result = fit_bradley_terry(ids, g, RankingConfig())
    assert set(result.ranking) == set(ids)
    assert result.scores["A"] > result.scores["B"]
    assert result.scores["C"] > result.scores["D"]


def test_tie_break_is_lexical_when_scores_equal():
    ids = ["M02", "M01"]
    result = fit_bradley_terry(ids, _graph(ids, []), RankingConfig())
    assert result.ranking == ["M01", "M02"]


def test_weight_mode_neglog():
    cfg = RankingConfig(weight_mode="neglog", confidence_clamp_min=0.55, confidence_clamp_max=0.95)
    w = observation_weight(0.8, cfg)
    assert w > 0.8
