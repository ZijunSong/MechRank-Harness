from closer.audit.detector import detect_audit_items
from closer.config import AuditConfig
from closer.ranking.graph import PreferenceGraph
from closer.schemas.comparison import PreferenceObservation
from closer.schemas.evidence import VariantEvidenceState
from closer.schemas.ranking import SolverResult


def _ev(vid: str, effect: str, conf: float, epi: bool = False) -> VariantEvidenceState:
    return VariantEvidenceState(
        variant_id=vid,
        mutation_summary="A1V",
        evidence=[],
        epistasis_required=epi,
        dominant_mechanisms=[],
        net_assay_effect=effect,  # type: ignore[arg-type]
        net_confidence=conf,
        unresolved_questions=[],
    )


def test_detects_cycle_low_margin_conflict_and_epistasis():
    ids = ["A", "B", "C"]
    evidence = {
        "A": _ev("A", "beneficial", 0.9),
        "B": _ev("B", "deleterious", 0.85),
        "C": _ev("C", "near_neutral", 0.4, epi=True),
    }
    graph = PreferenceGraph()
    graph.add_nodes(ids)
    for left, right in (("A", "B"), ("B", "C"), ("C", "A")):
        graph.add_observation(
            PreferenceObservation(
                observation_id=left + right,
                left_id=left,
                right_id=right,
                winner="left",
                confidence=0.8,
                source="pairwise",
                round_index=0,
                rationale="x",
            )
        )
    solver = SolverResult(
        ranking=["B", "A", "C"],
        scores={"B": 0.01, "A": 0.0, "C": -0.01},
        method="bradley_terry",
        converged=True,
        n_observations=3,
        n_informative_edges=3,
    )
    items = detect_audit_items(
        variant_ids=ids,
        evidence=evidence,
        graph=graph,
        solver=solver,
        config=AuditConfig(low_margin_threshold=0.15, evidence_conflict_min_confidence=0.75),
    )
    kinds = {item.kind for item in items}
    assert "cycle" in kinds
    assert "low_margin" in kinds
    assert "evidence_conflict" in kinds
    assert "epistasis" in kinds
