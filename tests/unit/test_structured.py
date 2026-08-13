from closer.llm.structured import extract_json_object, parse_model
from closer.schemas.ranking import RankingJSON


def test_extracts_json_from_prose_and_fences():
    text = 'Here is the answer:\n```json\n{"ranking": ["M02", "M01"]}\n```\n'
    parsed = parse_model(RankingJSON, text)
    assert parsed.ranking == ["M02", "M01"]


def test_extract_raw_object():
    obj = extract_json_object('prefix {"ranking": ["M01"]} suffix')
    assert obj["ranking"] == ["M01"]
