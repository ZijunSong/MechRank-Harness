from tests.fixtures.episodes import synthetic_episode

from closer.logging import episode_content_hash


def test_same_labels_different_sequences_have_different_hashes():
    first = synthetic_episode(4, protein="HashProt")
    second = first.model_copy(
        update={
            "variants": [
                variant.model_copy(
                    update={"sequence": variant.sequence[:-1] + ("A" if variant.sequence[-1] != "A" else "C")}
                )
                for variant in first.variants
            ]
        }
    )
    assert [v.variant_id for v in first.variants] == [v.variant_id for v in second.variants]
    assert first.variants[0].sequence != second.variants[0].sequence
    assert episode_content_hash(first) != episode_content_hash(second)


def test_hash_is_stable_for_identical_content():
    episode = synthetic_episode(3)
    assert episode_content_hash(episode) == episode_content_hash(episode)
