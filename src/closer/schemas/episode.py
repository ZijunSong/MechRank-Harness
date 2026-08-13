"""Episode input schemas and amino-acid validation."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def validate_amino_acid_sequence(sequence: str, *, field_name: str) -> str:
    if not sequence:
        raise ValueError(f"{field_name} must be non-empty")
    compact = sequence.strip().upper()
    if not compact:
        raise ValueError(f"{field_name} must be non-empty")
    illegal = sorted({ch for ch in compact if ch not in AMINO_ACIDS})
    if illegal:
        raise ValueError(f"{field_name} contains illegal amino-acid characters: {illegal}")
    return compact


class VariantInput(BaseModel):
    variant_id: str
    sequence: str

    @field_validator("variant_id")
    @classmethod
    def _id_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("variant_id must be non-empty")
        return stripped

    @field_validator("sequence")
    @classmethod
    def _seq_valid(cls, value: str) -> str:
        return validate_amino_acid_sequence(value, field_name="sequence")


class ProteinEpisode(BaseModel):
    protein_name: str
    organism: str
    assay_description: str
    higher_is_better: bool = True
    wild_type_sequence: str
    variants: list[VariantInput]
    benchmark_name: str | None = None
    system_prompt: str | None = None
    raw_user_prompt: str | None = None

    @field_validator("protein_name", "organism", "assay_description")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text fields must be non-empty")
        return stripped

    @field_validator("wild_type_sequence")
    @classmethod
    def _wt_valid(cls, value: str) -> str:
        return validate_amino_acid_sequence(value, field_name="wild_type_sequence")

    @model_validator(mode="after")
    def _check_candidates(self) -> ProteinEpisode:
        if len(self.variants) < 2:
            raise ValueError("episode must contain at least 2 candidate variants")
        ids = [variant.variant_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("variant ids must be unique")
        return self

    def variant_ids(self) -> list[str]:
        return [variant.variant_id for variant in self.variants]

    def variant_map(self) -> dict[str, VariantInput]:
        return {variant.variant_id: variant for variant in self.variants}
