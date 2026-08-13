from closer.comparison.scheduler import ComparisonScheduler, chunk_groups, initial_order
from closer.schemas.evidence import VariantEvidenceState


def _state(vid: str, effect: str, conf: float) -> VariantEvidenceState:
    return VariantEvidenceState(
        variant_id=vid,
        mutation_summary="A1V",
        evidence=[],
        epistasis_required=False,
        dominant_mechanisms=["stability"],
        net_assay_effect=effect,  # type: ignore[arg-type]
        net_confidence=conf,
        unresolved_questions=[],
    )


def test_initial_order_uses_ordinal_prior_not_as_final_score():
    states = {
        "M01": _state("M01", "deleterious", 0.9),
        "M02": _state("M02", "strong_beneficial", 0.4),
        "M03": _state("M03", "beneficial", 0.95),
    }
    order = initial_order(states, ["M01", "M02", "M03"])
    assert order[0] == "M02"
    assert order[-1] == "M01"


def test_groups_and_overlap():
    ids = [f"M{i:02d}" for i in range(1, 11)]
    groups = chunk_groups(ids, 5)
    assert groups == [ids[:5], ids[5:]]
    sched = ComparisonScheduler(group_size=5, overlap=2)
    crossed = sched.boundary_groups(groups)
    assert len(crossed) == 1
    assert set(groups[0][-2:]).issubset(set(crossed[0]))
    assert set(groups[1][:3]).issubset(set(crossed[0]))
