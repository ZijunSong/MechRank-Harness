"""Episode-local structured evidence store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from closer.logging import dump_json, load_json
from closer.schemas.assay import AssayContract
from closer.schemas.episode import ProteinEpisode
from closer.schemas.evidence import VariantEvidenceState
from closer.schemas.mutation import VariantMutationProfile


class EpisodeEvidenceState(BaseModel):
    episode: ProteinEpisode
    assay_contract: AssayContract
    mutation_profiles: dict[str, VariantMutationProfile]
    variant_evidence: dict[str, VariantEvidenceState]

    def get_variant(self, variant_id: str) -> VariantEvidenceState:
        if variant_id not in self.variant_evidence:
            raise KeyError(variant_id)
        return self.variant_evidence[variant_id]

    def get_pair(
        self, id1: str, id2: str
    ) -> tuple[VariantEvidenceState, VariantEvidenceState]:
        return self.get_variant(id1), self.get_variant(id2)

    def list_uncertain_variants(self) -> list[VariantEvidenceState]:
        return [
            state
            for state in self.variant_evidence.values()
            if state.net_assay_effect == "uncertain" or state.net_confidence < 0.5
        ]

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save_json(self, path: Path) -> None:
        dump_json(path, self.model_dump(mode="json"))

    @classmethod
    def load_json(cls, path: Path) -> EpisodeEvidenceState:
        return cls.model_validate(load_json(path))
