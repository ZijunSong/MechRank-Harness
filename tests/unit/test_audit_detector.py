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


def test_low_margin_priority_is_reachable_under_default_min_priority():
    from closer.audit.detector import select_pair_audit_items

    ids = ["A", "B", "C"]
    evidence = {
        "A": _ev("A", "near_neutral", 0.4),
        "B": _ev("B", "near_neutral", 0.4),
        "C": _ev("C", "near_neutral", 0.4),
    }
    graph = PreferenceGraph()
    graph.add_nodes(ids)
    solver = SolverResult(
        ranking=["A", "B", "C"],
        scores={"A": 0.01, "B": 0.0, "C": -0.01},
        method="bradley_terry",
        converged=True,
        n_observations=0,
        n_informative_edges=0,
    )
    config = AuditConfig()
    items = detect_audit_items(
        variant_ids=ids,
        evidence=evidence,
        graph=graph,
        solver=solver,
        config=config,
    )
    margins = [item for item in items if item.kind == "low_margin"]
    assert margins
    assert all(item.priority >= config.min_priority for item in margins)
    selected = select_pair_audit_items(items, config)
    assert any(item.kind == "low_margin" for item in selected)


def test_cycle_item_is_bound_to_a_pair():
    ids = ["A", "B", "C"]
    evidence = {vid: _ev(vid, "near_neutral", 0.4) for vid in ids}
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
        ranking=["A", "B", "C"],
        scores={"A": 0.2, "B": 0.19, "C": -1.0},
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
        config=AuditConfig(),
    )
    cycles = [item for item in items if item.kind == "cycle"]
    assert cycles
    assert len(cycles[0].variant_ids) == 2
    assert set(cycles[0].variant_ids) == {"A", "B"}


def test_same_pair_from_multiple_detectors_is_merged():
    from closer.audit.detector import AuditItem, select_pair_audit_items

    config = AuditConfig()
    items = [
        AuditItem(kind="cycle", variant_ids=["A", "B"], priority=1.0, reason="cycle"),
        AuditItem(kind="low_margin", variant_ids=["B", "A"], priority=0.9, reason="margin"),
        AuditItem(kind="epistasis", variant_ids=["C"], priority=0.8, reason="epi"),
    ]
    selected = select_pair_audit_items(items, config)
    assert len(selected) == 1
    assert selected[0].variant_ids == ["A", "B"]
    assert "cycle" in selected[0].reason and "margin" in selected[0].reason
