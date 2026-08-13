"""Independent development-set evaluator. Never reads PG-LLM held-out labels."""

from __future__ import annotations

import json
import math
from pathlib import Path

from closer.logging import dump_json
from closer.orchestrator.engine import CloserEngine
from closer.schemas.episode import ProteinEpisode, VariantInput


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return cov / (va * vb) if va and vb else 0.0


def load_dev_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def row_to_episode(row: dict) -> tuple[ProteinEpisode, dict[str, float] | None]:
    variants = [
        VariantInput(variant_id=item["variant_id"], sequence=item["sequence"])
        for item in row["variants"]
    ]
    episode = ProteinEpisode(
        protein_name=row["protein_name"],
        organism=row["organism"],
        assay_description=row["assay_description"],
        higher_is_better=row.get("higher_is_better", True),
        wild_type_sequence=row["wild_type_sequence"],
        variants=variants,
        benchmark_name=row.get("benchmark_name", "dev"),
    )
    scores = row.get("scores")
    return episode, scores


async def evaluate_dev(
    engine: CloserEngine,
    dataset_path: Path,
    output_path: Path | None = None,
) -> dict:
    rows = load_dev_manifest(dataset_path)
    per_assay = []
    rhos = []
    for row in rows:
        episode, scores = row_to_episode(row)
        result = await engine.run_episode(episode)
        item = {
            "protein_name": episode.protein_name,
            "ranking": result.ranking,
            "usage": result.usage,
            "closure_consistency": result.diagnostics.get("closure_consistency"),
        }
        if scores:
            pred = [-result.ranking.index(vid) for vid in episode.variant_ids()]
            truth = [float(scores[vid]) for vid in episode.variant_ids()]
            rho = spearman(pred, truth)
            item["spearman"] = rho
            rhos.append(rho)
        per_assay.append(item)
    summary = {
        "n_assays": len(rows),
        "mean_spearman": (sum(rhos) / len(rhos)) if rhos else None,
        "per_assay": per_assay,
    }
    if output_path is not None:
        dump_json(output_path, summary)
    return summary
