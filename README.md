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
| `PGLLM_OPENAI_BASE_URL` | Direct PG-LLM provider endpoint for bare-model eval. |
| `PGLLM_OPENAI_API_KEY` | Direct PG-LLM provider key. |
| `PGLLM_MODEL_ID` | Provider model id for direct eval (e.g. `gpt-5.5`). |

## ProteinGym-LLM

```bash
bash scripts/pgllm/setup_pgllm.sh
pgllm-data
export PGLLM_CLOSER_BASE_URL=http://127.0.0.1:8099/v1
export PGLLM_CLOSER_API_KEY=dummy-local-key
pgllm-models --registry configs/pgllm_closer_model.json --models closer-v1
```

### CLOSER harness eval (official path)

Start the harness server, freeze config, then run the official PG-LLM workflow:

```bash
closer-server --config configs/closer.pgllm.yaml --host 127.0.0.1 --port 8099
python scripts/freeze_config.py \
  --config configs/closer.pgllm.yaml \
  --output artifacts/frozen/closer_v1_manifest.json
bash scripts/pgllm/run_pgllm.sh
bash scripts/pgllm/status_pgllm.sh
bash scripts/pgllm/score_pgllm.sh
bash scripts/pgllm/export_pgllm.sh
```

Resume after transient API failures:

```bash
bash scripts/pgllm/resume_pgllm.sh
```

If you changed registry endpoint/model fingerprint, rerun affected cells:

```bash
bash scripts/pgllm/rerun_pgllm.sh
```

### Bare OpenAI-compatible model eval

Only change provider credentials and model id:

```bash
cp .env.example .env
# set PGLLM_OPENAI_BASE_URL, PGLLM_OPENAI_API_KEY, PGLLM_MODEL_ID
export PGLLM_REGISTRY=configs/pgllm_gpt55.json
export PGLLM_MODELS=gpt55
bash scripts/pgllm/run_openai_model.sh
```

Or render a registry for any model:

```bash
python scripts/pgllm/render_model_registry.py > configs/pgllm_active_model.json
export PGLLM_REGISTRY=configs/pgllm_active_model.json
export PGLLM_MODELS=gpt55
bash scripts/pgllm/run_openai_model.sh
```

Direct listwise proxy through CLOSER (still one PG-LLM call, no full harness):

```bash
closer-server --config configs/closer.direct.yaml --host 127.0.0.1 --port 8099
bash scripts/pgllm/run_pgllm.sh
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
