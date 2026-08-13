from pathlib import Path

from closer.benchmark.pgllm_parser import parse_pgllm_prompt
from closer.mutation.compiler import compile_episode_mutations


def test_fixture_prompt_parses_and_compiles():
    prompt = Path("tests/fixtures/synthetic_n10.prompt.txt").read_text(encoding="utf-8")
    episode = parse_pgllm_prompt(prompt)
    assert len(episode.variants) == 10
    profiles = compile_episode_mutations(episode)
    assert profiles["M06"].mutation_count == 0
    assert profiles["M01"].shorthand() == "K1E"
