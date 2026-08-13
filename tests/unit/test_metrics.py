import pytest

from closer.eval.dev import spearman
from closer.metrics.closure import evidence_ranking_conflict_rate
from closer.schemas.evidence import VariantEvidenceState


def test_spearman_perfect():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_closure_consistency():
    evidence = {
        "A": VariantEvidenceState(
            variant_id="A",
            mutation_summary="A1V",
            evidence=[],
            epistasis_required=False,
            dominant_mechanisms=[],
            net_assay_effect="beneficial",
            net_confidence=0.9,
            unresolved_questions=[],
        ),
        "B": VariantEvidenceState(
            variant_id="B",
            mutation_summary="A2V",
            evidence=[],
            epistasis_required=False,
            dominant_mechanisms=[],
            net_assay_effect="deleterious",
            net_confidence=0.9,
            unresolved_questions=[],
        ),
    }
    ok = evidence_ranking_conflict_rate(evidence, ["A", "B"])
    bad = evidence_ranking_conflict_rate(evidence, ["B", "A"])
    assert ok["closure_consistency"] == 1.0
    assert bad["closure_consistency"] == 0.0
