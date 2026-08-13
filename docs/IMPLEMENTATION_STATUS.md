# Implementation status

Project: MechRank-Harness (CLOSER v0.1.0)  
Root: `/data/ppnm/MechRank-Harness`

## Tests (2026-08-13)

- `ruff check src tests scripts`: all checks passed
- `pytest tests/unit tests/e2e tests/integration`: unit + probe e2e pass; integration/full ranking e2e skip unless inner LLM env is set
- Real LLM Assay Interpreter on 5 synthetic assays (`artifacts/runs/phase05_assay_interpreter/`):
  - binding conf=0.82
  - stability conf=0.78
  - growth conf=0.72
  - expression conf=0.82
  - regulatory conf=0.62
- Probe `/v1/responses` and `/v1/chat/completions` return `OK` without inner LLM
- Full ranking E2E: set `CLOSER_RUN_E2E=1` and inner LLM env matching `base_model.api_style`

## Phase checklist

| Phase | Status | Notes |
|---|---|---|
| 01 Engineering, config, schemas, logging | done | YAML config, pydantic schemas, run manifests |
| 02 PG-LLM parser + renderer | done | `ranking-v1`, whitespace/newline robust |
| 03 Mutation compiler | done | equal-length substitutions + reconstruction |
| 04 LLM client + structured output | done | Responses + Chat, schema + repair |
| 05 Assay interpreter | done | real LLM, `AssayInterpretationError` on failure |
| 06 Mechanistic evidence analyzer | done | batched 3-5, asyncio semaphore |
| 07 Evidence state store | done | save/load/hash/get_pair |
| 08 Comparison scheduler | done | group_size=5, overlap boundary groups |
| 09 Comparative reasoner | done | setwise + pairwise |
| 10 Preference graph | done | append-only observations, cycles, components |
| 11 Bradley-Terry solver | done | weighted NLL + L2, lexical tie-break |
| 12 Audit detector | done | cycle / margin / conflict / epistasis |
| 13 Adaptive auditor | done | targeted LLM re-comparison, source=audit |
| 14 Orchestrator | done | `CloserEngine.run_episode` |
| 15 Diagnostics | done | closure consistency + usage |
| 16 OpenAI-compatible server | done | `/v1/responses`, `/v1/chat/completions` |
| 17 Synthetic E2E | implemented | probe unit/e2e always; ranking e2e needs inner LLM |
| 18 Independent dev evaluator | done | `closer-eval-dev` |
| 19 Freeze tooling | done | `scripts/freeze_config.py` |
| 20 PG-LLM probe | scripts ready | `configs/pgllm_closer_model.json` |
| 21 Frozen full PG-LLM eval | not run | requires freeze + inner LLM + `pgllm-data` |

## What is intentionally not in v1

AlphaFold, PDB, MSA, UniProt, ESM, Tranception, PoET, FoldX, Rosetta, literature search, DMS retrieval, fine-tuning, RL.
