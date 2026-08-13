"""Real-LLM evidence / comparison smoke tests."""

from __future__ import annotations

import os

import pytest
from tests.fixtures.episodes import synthetic_episode

from closer.assay.interpreter import AssayInterpreter
from closer.comparison.reasoner import ComparativeReasoner
from closer.config import load_config
from closer.evidence.analyzer import MechanisticAnalyzer
from closer.evidence.state import EpisodeEvidenceState
from closer.llm import build_llm_client
from closer.mutation.compiler import compile_episode_mutations

pytestmark = pytest.mark.integration

skip_no_llm = pytest.mark.skipif(
    not all(os.environ.get(name) for name in ("CLOSER_BASE_URL", "CLOSER_API_KEY", "CLOSER_MODEL")),
    reason="real LLM endpoint is not configured",
)


@skip_no_llm
@pytest.mark.asyncio
async def test_evidence_and_setwise_real_llm():
    episode = synthetic_episode(4)
    cfg = load_config()
    client = build_llm_client(cfg)
    try:
        contract = await AssayInterpreter(client).interpret(episode)
        profiles = compile_episode_mutations(episode)
        analyzer = MechanisticAnalyzer(client, max_variants_per_call=2, parallelism=2)
        evidence = await analyzer.analyze_episode(episode, contract, profiles)
        assert set(evidence) == set(episode.variant_ids())
        for state in evidence.values():
            assert state.evidence
            assert 0.0 <= state.net_confidence <= 1.0
        store = EpisodeEvidenceState(
            episode=episode,
            assay_contract=contract,
            mutation_profiles=profiles,
            variant_evidence=evidence,
        )
        ranking = await ComparativeReasoner(client).rank_set(store, episode.variant_ids()[:3])
        assert set(ranking.ranking) == set(episode.variant_ids()[:3])
    finally:
        await client.aclose()
