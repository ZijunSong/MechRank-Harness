from closer.schemas.episode import ProteinEpisode, VariantInput


def test_episode_rejects_duplicate_ids():
    seq = "ACDE"
    try:
        ProteinEpisode(
            protein_name="P",
            organism="O",
            assay_description="binding",
            wild_type_sequence=seq,
            variants=[
                VariantInput(variant_id="M01", sequence="MCDE"),
                VariantInput(variant_id="M01", sequence="ADDE"),
            ],
        )
    except ValueError as exc:
        assert "unique" in str(exc).lower()
    else:
        raise AssertionError("duplicate ids must fail")


def test_episode_rejects_illegal_aa():
    try:
        VariantInput(variant_id="M01", sequence="ACDE1")
    except ValueError as exc:
        assert "illegal" in str(exc).lower()
    else:
        raise AssertionError("illegal amino acid must fail")


def test_episode_requires_two_variants():
    try:
        ProteinEpisode(
            protein_name="P",
            organism="O",
            assay_description="binding",
            wild_type_sequence="ACDE",
            variants=[VariantInput(variant_id="M01", sequence="MCDE")],
        )
    except ValueError:
        return
    raise AssertionError("single-variant episode must fail")
