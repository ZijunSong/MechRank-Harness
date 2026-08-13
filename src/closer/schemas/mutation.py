"""Mutation compiler schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ResidueSubstitution(BaseModel):
    position_1based: int = Field(ge=1)
    from_aa: str = Field(min_length=1, max_length=1)
    to_aa: str = Field(min_length=1, max_length=1)


class VariantMutationProfile(BaseModel):
    variant_id: str
    substitutions: list[ResidueSubstitution]
    mutation_count: int

    @model_validator(mode="after")
    def _count_matches(self) -> VariantMutationProfile:
        if self.mutation_count != len(self.substitutions):
            raise ValueError("mutation_count must equal len(substitutions)")
        return self

    def shorthand(self) -> str:
        if not self.substitutions:
            return "WT"
        return ",".join(
            f"{item.from_aa}{item.position_1based}{item.to_aa}" for item in self.substitutions
        )
