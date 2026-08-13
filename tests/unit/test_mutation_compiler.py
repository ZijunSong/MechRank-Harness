from closer.errors import MutationCompilationError
from closer.mutation.compiler import compile_episode_mutations, compile_variant_mutations
from closer.schemas.episode import ProteinEpisode, VariantInput

WT = "ACDEFGHIKLMNPQRSTVWY"


def test_single_mutation():
    mutant = "VCDEFGHIKLMNPQRSTVWY"
    profile = compile_variant_mutations(WT, "M01", mutant)
    assert profile.mutation_count == 1
    assert profile.substitutions[0].position_1based == 1
    assert profile.substitutions[0].from_aa == "A"
    assert profile.substitutions[0].to_aa == "V"
    assert profile.shorthand() == "A1V"


def test_multi_mutation():
    mutant = list(WT)
    mutant[0] = "V"
    mutant[4] = "A"
    mutant[19] = "A"
    profile = compile_variant_mutations(WT, "M02", "".join(mutant))
    assert profile.mutation_count == 3
    assert [s.position_1based for s in profile.substitutions] == [1, 5, 20]


def test_no_mutation():
    profile = compile_variant_mutations(WT, "M03", WT)
    assert profile.mutation_count == 0
    assert profile.substitutions == []
    assert profile.shorthand() == "WT"


def test_invalid_alphabet_same_length():
    try:
        compile_variant_mutations(WT, "M04", WT[:-1] + "1")
    except MutationCompilationError:
        return
    raise AssertionError("invalid alphabet must fail")


def test_length_mismatch_is_error():
    try:
        compile_variant_mutations(WT, "M05", WT + "A")
    except MutationCompilationError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("indel must fail")


def test_reconstruction_check_episode():
    episode = ProteinEpisode(
        protein_name="P",
        organism="O",
        assay_description="stability",
        wild_type_sequence=WT,
        variants=[
            VariantInput(variant_id="M01", sequence="VCDEFGHIKLMNPQRSTVWY"),
            VariantInput(variant_id="M02", sequence="ADDEFGHIKLMNPQRSTVWY"),
        ],
    )
    profiles = compile_episode_mutations(episode)
    assert set(profiles) == {"M01", "M02"}
    assert profiles["M01"].shorthand() == "A1V"
    assert profiles["M02"].shorthand() == "C2D"
