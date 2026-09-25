import pytest

from closer.audit.auditor import AdaptiveAuditor
from closer.audit.detector import AuditItem
from closer.config import AuditConfig, RankingConfig
from closer.errors import AuditError, ComparisonError
from closer.ranking.graph import PreferenceGraph
from closer.ranking.solver import GlobalRankSolver
from closer.schemas.comparison import PairPreference
from closer.schemas.ranking import SolverResult


def _pref(left: str, right: str, winner: str) -> PairPreference:
    return PairPreference(
        left_id=left,
        right_id=right,
        winner=winner,  # type: ignore[arg-type]
        confidence=0.8,
        key_evidence=["e"],
        contradiction_with_previous_evidence=False,
        rationale="r",
    )


class _FakeClient:
    def __init__(self, pref: PairPreference) -> None:
        self.pref = pref
        self.calls: list[dict] = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.pref, None


def _solver() -> SolverResult:
    return SolverResult(
        ranking=["M01", "M02", "M03"],
        scores={"M01": 0.2, "M02": 0.1, "M03": 0.0},
        method="bradley_terry",
        converged=True,
        n_observations=0,
        n_informative_edges=0,
    )


def _dummy_state():
    from unittest.mock import MagicMock

    state = MagicMock()
    state.assay_contract.model_dump.return_value = {}
    state.variant_evidence = {}
    state.get_variant.return_value.model_dump.return_value = {}
    return state


@pytest.mark.asyncio
async def test_auditor_keeps_cycle_pair_chosen_by_model():
    client = _FakeClient(_pref("M02", "M03", "left"))
    auditor = AdaptiveAuditor(client, AuditConfig(), GlobalRankSolver(RankingConfig()))
    item = AuditItem(kind="cycle", variant_ids=["M01", "M02", "M03"], priority=1.0, reason="cycle")
    graph = PreferenceGraph()
    graph.add_nodes(["M01", "M02", "M03"])
    parsed = await auditor._recompare(_dummy_state(), item, graph, _solver(), None)
    assert parsed.left_id == "M02"
    assert parsed.right_id == "M03"
    assert parsed.winner == "left"


@pytest.mark.asyncio
async def test_auditor_does_not_rewrite_bound_pair():
    client = _FakeClient(_pref("M02", "M01", "left"))
    auditor = AdaptiveAuditor(client, AuditConfig(), GlobalRankSolver(RankingConfig()))
    item = AuditItem(kind="low_margin", variant_ids=["M01", "M02"], priority=0.9, reason="margin")
    parsed = await auditor._recompare(_dummy_state(), item, PreferenceGraph(), _solver(), None)
    assert parsed.left_id == "M01"
    assert parsed.right_id == "M02"
    assert parsed.winner == "right"


@pytest.mark.asyncio
async def test_single_variant_epistasis_cannot_emit_pair():
    client = _FakeClient(_pref("M01", "GHOST", "left"))
    auditor = AdaptiveAuditor(client, AuditConfig(), GlobalRankSolver(RankingConfig()))
    item = AuditItem(kind="epistasis", variant_ids=["M01"], priority=0.8, reason="epi")
    with pytest.raises(AuditError, match="single-variant"):
        await auditor._recompare(_dummy_state(), item, PreferenceGraph(), _solver(), None)
    assert client.calls == []


@pytest.mark.asyncio
async def test_unknown_endpoint_rejected_before_graph():
    client = _FakeClient(_pref("M01", "GHOST", "left"))
    auditor = AdaptiveAuditor(client, AuditConfig(), GlobalRankSolver(RankingConfig()))
    item = AuditItem(kind="low_margin", variant_ids=["M01", "M02"], priority=0.9, reason="margin")
    with pytest.raises(ComparisonError):
        await auditor._recompare(_dummy_state(), item, PreferenceGraph(), _solver(), None)
