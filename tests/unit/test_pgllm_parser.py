from tests.fixtures.episodes import WT, make_mutant, ranking_v1_prompt, synthetic_episode

from closer.benchmark.pgllm_parser import parse_pgllm_prompt
from closer.benchmark.pgllm_renderer import render_user_prompt
from closer.errors import BenchmarkParseError
from closer.schemas.episode import ProteinEpisode, VariantInput


def _assert_roundtrip(n: int) -> None:
    prompt = ranking_v1_prompt(n)
    episode = parse_pgllm_prompt(prompt)
    assert len(episode.variants) == n
    assert episode.variant_ids() == [f"M{i:02d}" for i in range(1, n + 1)]
    assert episode.wild_type_sequence == WT
    rerendered = render_user_prompt(episode)
    again = parse_pgllm_prompt(rerendered)
    assert again.variant_ids() == episode.variant_ids()
    assert [v.sequence for v in again.variants] == [v.sequence for v in episode.variants]


def test_parse_n10():
    _assert_roundtrip(10)


def test_parse_n50():
    _assert_roundtrip(50)


def test_parse_n100():
    _assert_roundtrip(100)


def test_different_sequence_lengths():
    short = ProteinEpisode(
        protein_name="Tiny",
        organism="E. coli",
        assay_description="thermal stability of a mini protein",
        wild_type_sequence="ACDEFGHIKL",
        variants=[
            VariantInput(variant_id="M01", sequence="VCDEFGHIKL"),
            VariantInput(variant_id="M02", sequence="ADDEFGHIKL"),
        ],
    )
    parsed = parse_pgllm_prompt(render_user_prompt(short))
    assert parsed.wild_type_sequence == "ACDEFGHIKL"
    assert len(parsed.variants[0].sequence) == 10


def test_whitespace_variation():
    prompt = ranking_v1_prompt(10)
    spaced = prompt.replace("**Protein:**", "**Protein:**  ").replace("M01:", "M01:  ")
    episode = parse_pgllm_prompt(spaced)
    assert episode.variants[0].variant_id == "M01"


def test_newline_variation():
    prompt = ranking_v1_prompt(10, extra_blank_lines=True)
    episode = parse_pgllm_prompt(prompt)
    assert len(episode.variants) == 10


def test_missing_candidate_raises():
    prompt = ranking_v1_prompt(10)
    prompt = prompt.replace("M05: ", "XX05: ")
    try:
        parse_pgllm_prompt(prompt)
    except BenchmarkParseError as exc:
        assert "N=" in str(exc) or "count" in str(exc).lower()
    else:
        raise AssertionError("missing candidate must raise")


def test_duplicate_mxx_raises():
    episode = synthetic_episode(4)
    episode.variants[3].variant_id = "M01"
    # bypass model uniqueness by editing the rendered prompt
    prompt = render_user_prompt(synthetic_episode(4))
    prompt = prompt.replace("M04:", "M01:", 1)
    try:
        parse_pgllm_prompt(prompt)
    except BenchmarkParseError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate Mxx must raise")


def test_illegal_sequence_character_raises():
    prompt = ranking_v1_prompt(4)
    mutant = make_mutant(WT, {2: "A"})
    prompt = prompt.replace(mutant, mutant[:-1] + "B", 1)
    # B is legal. Use digit instead.
    prompt = ranking_v1_prompt(4)
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("M02:"):
            seq = line.split(":", 1)[1].strip()
            lines[i] = f"M02: {seq[:-1]}1"
            break
    try:
        parse_pgllm_prompt("\n".join(lines))
    except BenchmarkParseError:
        return
    raise AssertionError("illegal character must raise")


def test_n_declaration_mismatch_raises():
    prompt = ranking_v1_prompt(4)
    prompt = prompt.replace("**4 candidate mutant sequences to rank:**", "**5 candidate mutant sequences to rank:**")
    try:
        parse_pgllm_prompt(prompt)
    except BenchmarkParseError as exc:
        assert "N=" in str(exc)
    else:
        raise AssertionError("N mismatch must raise")
