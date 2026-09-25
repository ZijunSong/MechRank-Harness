import pytest
from tests.fixtures.episodes import synthetic_episode

from closer.config import CloserConfig, load_config
from closer.llm.base import LLMClient
from closer.orchestrator.engine import CloserEngine
from closer.schemas.ranking import RankingJSON
from closer.schemas.trace import LLMCallTrace, TokenUsage


class _FakeClient(LLMClient):
    def __init__(self, ranking: list[str]) -> None:
        super().__init__(CloserConfig())
        self.ranking = ranking
        self.stages: list[str] = []
        self.last_user = ""
        self.last_max_output_tokens = None

    async def generate_text(self, *, stage, system, user, budget=None, max_output_tokens=None):
        self.stages.append(stage)
        self.last_user = user
        self.last_max_output_tokens = max_output_tokens
        if budget is not None:
            budget.precheck()
            budget.record(TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12), latency_ms=1)
        text = '{"ranking": ' + str(self.ranking).replace("'", '"') + "}"
        return text, LLMCallTrace(
            call_id="t",
            stage=stage,
            model_requested="fake",
            prompt_hash="x",
            usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
            latency_ms=1,
        )

    async def generate_structured(self, *, stage, schema, system, user, budget=None, max_output_tokens=None):
        text, trace = await self.generate_text(
            stage=stage,
            system=system,
            user=user,
            budget=budget,
            max_output_tokens=max_output_tokens,
        )
        return schema.model_validate({"ranking": self.ranking}), trace


@pytest.mark.asyncio
async def test_direct_proxy_uses_raw_prompt_and_limit():
    episode = synthetic_episode(3)
    episode = episode.model_copy(update={"raw_user_prompt": "RAW PROMPT ONLY"})
    cfg = load_config()
    cfg.execution.mode = "direct_proxy"
    cfg.ablation.comparative_reasoning = False
    cfg.ablation.assay_contract = False
    cfg.ablation.evidence_state = False
    cfg.ablation.adaptive_audit = False
    client = _FakeClient(episode.variant_ids())
    result = await CloserEngine(cfg, client).run_episode(episode, max_output_tokens=111)
    assert result.ranking == episode.variant_ids()
    assert client.stages == ["direct_proxy"]
    assert client.last_user == "RAW PROMPT ONLY"
    assert client.last_max_output_tokens == 111
    assert "Deterministic mutation profiles" not in client.last_user


@pytest.mark.asyncio
async def test_listwise_min_attaches_mutation_table():
    episode = synthetic_episode(3)
    cfg = load_config()
    cfg.execution.mode = "listwise_min"
    cfg.ablation.mutation_compiler = True
    cfg.ablation.comparative_reasoning = False
    cfg.ablation.assay_contract = False
    cfg.ablation.evidence_state = False
    cfg.ablation.adaptive_audit = False
    client = _FakeClient(episode.variant_ids())
    result = await CloserEngine(cfg, client).run_episode(episode)
    assert result.ranking == episode.variant_ids()
    assert client.stages == ["direct_rank"]
    assert "Deterministic mutation profiles" in client.last_user
    assert isinstance(RankingJSON(ranking=result.ranking), RankingJSON)


@pytest.mark.asyncio
async def test_listwise_compact_does_not_repeat_mutant_sequences():
    episode = synthetic_episode(3)
    cfg = load_config()
    cfg.execution.mode = "listwise_compact"
    cfg.ablation.mutation_compiler = True
    cfg.ablation.comparative_reasoning = False
    cfg.ablation.assay_contract = False
    cfg.ablation.evidence_state = False
    cfg.ablation.adaptive_audit = False
    client = _FakeClient(episode.variant_ids())
    result = await CloserEngine(cfg, client).run_episode(episode)
    assert result.ranking == episode.variant_ids()
    assert client.stages == ["listwise_compact"]
    assert episode.wild_type_sequence in client.last_user
    for variant in episode.variants:
        assert variant.sequence not in client.last_user
        assert variant.variant_id in client.last_user
