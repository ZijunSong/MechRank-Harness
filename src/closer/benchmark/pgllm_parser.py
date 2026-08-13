"""ProteinGym-LLM ranking-v1 prompt parser."""

from __future__ import annotations

import re

from closer.errors import BenchmarkParseError
from closer.schemas.episode import ProteinEpisode, VariantInput, validate_amino_acid_sequence

PROTEIN_RE = re.compile(
    r"\*\*Protein:\*\*\s*(?P<name>.+?)\s*\(\s*(?P<organism>.+?)\s*\)",
    re.IGNORECASE,
)
ASSAY_RE = re.compile(
    r"\*\*Assay\s*\(what is measured\):\*\*\s*(?P<assay>.+)",
    re.IGNORECASE,
)
HIGHER_RE = re.compile(
    r"\*\*Higher experimental fitness\s*=\s*HIGHER value of the measured property\.\*\*",
    re.IGNORECASE,
)
WT_HEADER_RE = re.compile(
    r"\*\*Wild-type sequence\s*\(\s*(?P<n_aa>\d+)\s*aa\s*\):\*\*",
    re.IGNORECASE,
)
CANDIDATE_HEADER_RE = re.compile(
    r"\*\*\s*(?P<n>\d+)\s+candidate mutant sequences to rank:\s*\*\*",
    re.IGNORECASE,
)
CANDIDATE_HEADER_MUT_RE = re.compile(
    r"\*\*\s*(?P<n>\d+)\s+candidate mutants to rank\*\*",
    re.IGNORECASE,
)
VARIANT_RE = re.compile(
    r"^(?P<vid>M\d+)\s*(?:\[[^\]]+\])?\s*:\s*(?P<seq>[A-Za-z\s]+)$",
    re.IGNORECASE,
)
HEADER_RE = re.compile(r"^\*\*")


def _collapse_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.strip())


def _strip_aa(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _iter_logical_lines(prompt: str) -> list[str]:
    return prompt.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def parse_pgllm_prompt(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    benchmark_name: str | None = "ranking-v1",
) -> ProteinEpisode:
    if not user_prompt or not user_prompt.strip():
        raise BenchmarkParseError("empty ProteinGym-LLM prompt")

    protein_match = PROTEIN_RE.search(user_prompt)
    if protein_match is None:
        raise BenchmarkParseError("missing **Protein:** name (organism) header")

    assay_match = ASSAY_RE.search(user_prompt)
    if assay_match is None:
        raise BenchmarkParseError("missing **Assay (what is measured):** header")

    if HIGHER_RE.search(user_prompt) is None:
        raise BenchmarkParseError("missing higher-is-better orientation sentence")

    lines = _iter_logical_lines(user_prompt)
    wt_seq, declared_aa = _extract_wt(lines)
    declared_n, variants = _extract_variants(lines)

    if declared_n != len(variants):
        raise BenchmarkParseError(
            f"declared candidate count N={declared_n} does not match parsed count {len(variants)}"
        )

    ids = [variant.variant_id for variant in variants]
    if len(ids) != len(set(ids)):
        dupes = sorted({vid for vid in ids if ids.count(vid) > 1})
        raise BenchmarkParseError(f"duplicate candidate ids: {dupes}")

    if declared_aa != len(wt_seq):
        raise BenchmarkParseError(
            f"declared WT length {declared_aa} does not match sequence length {len(wt_seq)}"
        )

    try:
        episode = ProteinEpisode(
            protein_name=_collapse_ws(protein_match.group("name")),
            organism=_collapse_ws(protein_match.group("organism")),
            assay_description=_collapse_ws(assay_match.group("assay")),
            higher_is_better=True,
            wild_type_sequence=wt_seq,
            variants=variants,
            benchmark_name=benchmark_name,
            system_prompt=system_prompt,
            raw_user_prompt=user_prompt,
        )
    except ValueError as exc:
        raise BenchmarkParseError(str(exc)) from exc
    return episode


def _extract_wt(lines: list[str]) -> tuple[str, int]:
    start = None
    declared_aa = None
    for idx, line in enumerate(lines):
        match = WT_HEADER_RE.search(line.strip())
        if match:
            start = idx + 1
            declared_aa = int(match.group("n_aa"))
            trailing = WT_HEADER_RE.sub("", line).strip()
            collected = [trailing] if trailing else []
            break
    else:
        raise BenchmarkParseError("missing **Wild-type sequence (N aa):** block")

    collected = collected
    for line in lines[start:]:
        stripped = line.strip()
        if HEADER_RE.match(stripped):
            break
        if stripped:
            collected.append(stripped)
    wt_raw = _strip_aa("".join(collected))
    if not wt_raw:
        raise BenchmarkParseError("wild-type sequence is empty")
    try:
        wt_seq = validate_amino_acid_sequence(wt_raw, field_name="wild_type_sequence")
    except ValueError as exc:
        raise BenchmarkParseError(str(exc)) from exc
    return wt_seq, declared_aa


def _extract_variants(lines: list[str]) -> tuple[int, list[VariantInput]]:
    start = None
    declared_n = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        header = CANDIDATE_HEADER_RE.search(stripped) or CANDIDATE_HEADER_MUT_RE.search(stripped)
        if header:
            start = idx + 1
            declared_n = int(header.group("n"))
            break
    if start is None or declared_n is None:
        raise BenchmarkParseError("missing candidate mutant header")

    variants: list[VariantInput] = []
    buffer_id: str | None = None
    buffer_seq: list[str] = []

    def flush() -> None:
        nonlocal buffer_id, buffer_seq
        if buffer_id is None:
            return
        seq_raw = _strip_aa("".join(buffer_seq))
        try:
            seq = validate_amino_acid_sequence(seq_raw, field_name=buffer_id)
        except ValueError as exc:
            raise BenchmarkParseError(str(exc)) from exc
        variants.append(VariantInput(variant_id=buffer_id, sequence=seq))
        buffer_id = None
        buffer_seq = []

    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("rank all "):
            break
        match = VARIANT_RE.match(stripped)
        if match:
            flush()
            buffer_id = match.group("vid")
            buffer_seq = [match.group("seq")]
            continue
        if buffer_id is not None and re.fullmatch(r"[A-Za-z\s]+", stripped):
            buffer_seq.append(stripped)
            continue
        if HEADER_RE.match(stripped):
            break
    flush()
    if not variants:
        raise BenchmarkParseError("no candidate mutant sequences found")
    return declared_n, variants
