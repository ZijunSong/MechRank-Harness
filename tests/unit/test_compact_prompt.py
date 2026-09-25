import pytest
from tests.fixtures.episodes import synthetic_episode

from closer.benchmark.pgllm_renderer import render_compact_prompt
from closer.errors import MutationCompilationError
from closer.mutation.compiler import apply_substitutions, compile_episode_mutations


def test_compact_prompt_reconstructs_every_mutant():
    episode = synthetic_episode(6)
    profiles = compile_episode_mutations(episode)
    prompt = render_compact_prompt(episode, profiles)
    assert episode.wild_type_sequence in prompt
    assert episode.assay_description in prompt
    assert prompt.count(episode.wild_type_sequence) == 1
    for variant in episode.variants:
        reconstructed = apply_substitutions(episode.wild_type_sequence, profiles[variant.variant_id].substitutions)
        assert reconstructed == variant.sequence
        assert variant.variant_id in prompt
        assert variant.sequence not in prompt


def test_compact_prompt_rejects_mismatched_profile():
    episode = synthetic_episode(3)
    profiles = compile_episode_mutations(episode)
    broken = profiles[episode.variants[0].variant_id]
    broken = broken.model_copy(update={"substitutions": [], "mutation_count": 0})
    profiles[episode.variants[0].variant_id] = broken
    with pytest.raises(MutationCompilationError, match="reconstruct"):
        render_compact_prompt(episode, profiles)
