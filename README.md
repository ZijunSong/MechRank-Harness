# MechRank-Harness

**MechRank-Harness** is the repository for CLOSER (Closure-Oriented Scientific Evidence Ranking Harness), an OpenAI-compatible ranking agent for [ProteinGym-LLM](https://github.com/rohitarorayyc/proteingym-llm).

It does not ask a base LLM to emit a 50-way listwise ranking in one shot. It instead:

1. Parses a ProteinGym-LLM `ranking-v1` prompt.
2. Compiles WT vs mutant substitutions deterministically.
3. Interprets the assay into a typed Assay Contract.
4. Builds per-variant mechanistic evidence.
5. Collects setwise/pairwise biological preferences.
6. Solves a Bradley-Terry ranking on the preference graph.
7. Audits cycles, low-margin adjacencies, evidence conflicts, and epistasis uncertainty.
8. Returns `{"ranking":[...]}` covering every candidate exactly once.

Track A is information-equivalent to ProteinGym-LLM: no MSA, structure, literature, DMS labels, or specialist predictors.

## Quickstart

```bash
git clone https://github.com/ZijunSong/MechRank-Harness.git
cd MechRank-Harness
python -m pip install -e ".[dev]"
cp .env.example .env
# set CLOSER_BASE_URL, CLOSER_API_KEY, CLOSER_MODEL to a real OpenAI-compatible endpoint
```

Start the harness server (this is the endpoint ProteinGym-LLM should call):

```bash
closer-server --config configs/closer.default.yaml --host 127.0.0.1 --port 8099
```

Connectivity probe:

```bash
curl -s http://127.0.0.1:8099/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "closer-v1",
    "instructions": "You are a connectivity test.",
    "input": "Reply with the single word: OK"
  }'
```

Synthetic ranking request:

```bash
curl -s http://127.0.0.1:8099/v1/responses \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/responses_request.example.json
```

Local episode:

```bash
closer-run --input tests/fixtures/synthetic_n10.prompt.txt --config configs/closer.default.yaml
```

## Environment variables

| Variable | Role |
|---|---|
| `CLOSER_BASE_URL` | Inner real LLM endpoint. Must **not** point at the CLOSER server. |
| `CLOSER_API_KEY` | Inner provider key. |
| `CLOSER_MODEL` | Inner model id. |
| `CLOSER_SERVER_MODEL_ID` | Outer model id returned to PG-LLM (`closer-v1`). |
| `CLOSER_SERVER_HOST` / `CLOSER_SERVER_PORT` | Outer server bind. |
| `PGLLM_CLOSER_BASE_URL` | `http://127.0.0.1:8099/v1` |
| `PGLLM_CLOSER_API_KEY` | Dummy local key is allowed. |

## ProteinGym-LLM

```bash
bash scripts/pgllm/setup_pgllm.sh
export PGLLM_CLOSER_BASE_URL=http://127.0.0.1:8099/v1
export PGLLM_CLOSER_API_KEY=dummy-local-key
pgllm-models --registry configs/pgllm_closer_model.json --models closer-v1
```

Freeze before any official evaluation cell:

```bash
python scripts/freeze_config.py \
  --config configs/closer.pgllm.yaml \
  --output artifacts/frozen/closer_v1_manifest.json
```

Then, with CLOSER server running:

```bash
pgllm-run --registry configs/pgllm_closer_model.json --models closer-v1 --sizes 50
pgllm-status --models closer-v1 --sizes 50
pgllm-score --models closer-v1 --sizes 50 --breakdown
pgllm-export --models closer-v1 --sizes 50 --output results/closer-v1-n50.publication.jsonl.gz
```

Do not tune prompts, thresholds, solver, routing, or budget on official evaluation scores.

## Tests

```bash
pytest tests/unit
ruff check src tests scripts
```

Real-LLM integration (requires the inner endpoint):

```bash
pytest tests/integration tests/e2e -m integration
```

## Development evaluator

`data/dev_manifest.jsonl` is a synthetic, non-PG-LLM set.

```bash
closer-eval-dev --dataset data/dev_manifest.jsonl --config configs/closer.default.yaml
```
