"""Final ranking and episode validators."""

from __future__ import annotations

from closer.errors import FinalRankingValidationError, StructuredOutputError
from closer.llm.structured import parse_model
from closer.schemas.episode import ProteinEpisode
from closer.schemas.ranking import RankingJSON


def parse_ranking_from_text(text: str) -> list[str]:
    try:
        parsed = parse_model(RankingJSON, text)
    except StructuredOutputError as exc:
        raise FinalRankingValidationError(str(exc)) from exc
    if not isinstance(parsed, RankingJSON):
        raise FinalRankingValidationError("ranking payload is not RankingJSON")
    return list(parsed.ranking)


def validate_final_ranking(episode: ProteinEpisode, ranking: list[str]) -> list[str]:
    expected = episode.variant_ids()
    unknown = sorted(set(ranking) - set(expected))
    missing = sorted(set(expected) - set(ranking))
    if len(ranking) != len(set(ranking)):
        dupes = sorted({vid for vid in ranking if ranking.count(vid) > 1})
        raise FinalRankingValidationError(f"ranking contains duplicate ids: {dupes}")
    if unknown:
        raise FinalRankingValidationError(f"ranking contains unknown ids: {unknown}")
    if missing:
        raise FinalRankingValidationError(f"ranking is missing ids: {missing}")
    if len(ranking) != len(expected):
        raise FinalRankingValidationError(
            f"ranking length {len(ranking)} != number of candidates {len(expected)}"
        )
    return list(ranking)
