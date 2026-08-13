"""Real-LLM integration tests. Skipped unless CLOSER_BASE_URL / CLOSER_API_KEY / CLOSER_MODEL are set."""

from __future__ import annotations

import os

import pytest

from closer.assay.interpreter import AssayInterpreter
from closer.config import load_config
from closer.llm import build_llm_client
from closer.schemas.episode import ProteinEpisode, VariantInput

pytestmark = pytest.mark.integration

REQUIRED = ("CLOSER_BASE_URL", "CLOSER_API_KEY", "CLOSER_MODEL")


def _has_llm() -> bool:
    return all(os.environ.get(name) for name in REQUIRED)


skip_no_llm = pytest.mark.skipif(not _has_llm(), reason="real LLM endpoint is not configured")

ASSAYS = [
    (
        "binding",
        "Equilibrium binding affinity of the protein for its peptide ligand, measured by fluorescence polarization.",
    ),
    (
        "stability",
        "Thermal stability of the purified protein measured as melting temperature in a thermal shift assay.",
    ),
    (
        "growth",
        "Organismal fitness measured as relative growth rate of a microbial strain expressing the variant.",
    ),
    (
        "expression",
        "Steady-state protein abundance in mammalian cells measured by a fluorescent reporter fused to the protein.",
    ),
    (
        "regulatory",
        "Transcriptional output of a promoter controlled by the protein, measured as luciferase reporter activity.",
    ),
]


def _episode(desc: str) -> ProteinEpisode:
    wt = "MKTAYIAKQRQISFVKSHFSRQ"
    return ProteinEpisode(
        protein_name="DevProtein",
        organism="Homo sapiens",
        assay_description=desc,
        wild_type_sequence=wt,
        variants=[
            VariantInput(variant_id="M01", sequence="AKTAYIAKQRQISFVKSHFSRQ"),
            VariantInput(variant_id="M02", sequence="METAYIAKQRQISFVKSHFSRQ"),
        ],
    )


@skip_no_llm
@pytest.mark.asyncio
@pytest.mark.parametrize("name,desc", ASSAYS)
async def test_assay_interpreter_real_llm(name, desc):
    cfg = load_config()
    client = build_llm_client(cfg)
    try:
        contract = await AssayInterpreter(client).interpret(_episode(desc))
    finally:
        await client.aclose()
    assert 0.0 <= contract.confidence <= 1.0
    assert contract.measured_property
    assert contract.likely_causal_chain
    assert contract.higher_is_better is True
    print(f"ASSAY[{name}] {contract.measured_property} conf={contract.confidence}")
