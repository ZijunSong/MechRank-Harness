from tests.fixtures.episodes import synthetic_episode

from closer.benchmark.validators import validate_final_ranking
from closer.errors import FinalRankingValidationError


def test_valid_ranking():
    episode = synthetic_episode(4)
    ranking = list(reversed(episode.variant_ids()))
    assert validate_final_ranking(episode, ranking) == ranking


def test_missing_id():
    episode = synthetic_episode(4)
    ranking = episode.variant_ids()[:-1]
    try:
        validate_final_ranking(episode, ranking)
    except FinalRankingValidationError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("missing id must fail")


def test_duplicate_id():
    episode = synthetic_episode(4)
    ranking = episode.variant_ids()
    ranking[-1] = ranking[0]
    try:
        validate_final_ranking(episode, ranking)
    except FinalRankingValidationError:
        return
    raise AssertionError("duplicate id must fail")


def test_unknown_id():
    episode = synthetic_episode(4)
    ranking = episode.variant_ids()
    ranking[-1] = "M99"
    try:
        validate_final_ranking(episode, ranking)
    except FinalRankingValidationError as exc:
        assert "unknown" in str(exc).lower()
    else:
        raise AssertionError("unknown id must fail")
