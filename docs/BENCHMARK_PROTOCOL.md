# Benchmark protocol

1. Develop only on synthetic fixtures and `data/dev_manifest.jsonl`.
2. Do not read ProteinGym-LLM evaluation candidate sets or held-out labels while changing prompts, thresholds, solver, routing, or budget.
3. Freeze with `python scripts/freeze_config.py --config configs/closer.pgllm.yaml --output artifacts/frozen/closer_v1_manifest.json`.
4. After freeze, any change to prompts / thresholds / solver / audit / scheduler is a new version (`closer-v2`) and a new evaluation exposure.
5. Point PG-LLM at the CLOSER server, not at the inner base model:

```bash
export PGLLM_CLOSER_BASE_URL=http://127.0.0.1:8099/v1
export PGLLM_CLOSER_API_KEY=dummy-local-key
pgllm-models --registry configs/pgllm_closer_model.json --models closer-v1
bash scripts/pgllm/run_pgllm.sh
bash scripts/pgllm/status_pgllm.sh
bash scripts/pgllm/score_pgllm.sh
```

For bare OpenAI-compatible models (official PG-LLM direct path), set only
`PGLLM_OPENAI_BASE_URL`, `PGLLM_OPENAI_API_KEY`, and `PGLLM_MODEL_ID`, then:

```bash
export PGLLM_REGISTRY=configs/pgllm_gpt55.json
export PGLLM_MODELS=gpt55
bash scripts/pgllm/run_openai_model.sh
```

If a run stalls because of transient API errors, resume with:

```bash
bash scripts/pgllm/resume_pgllm.sh
```

If the registry endpoint or model fingerprint changed, rerun affected cells with:

```bash
bash scripts/pgllm/rerun_pgllm.sh
```

Keep `PGLLM_CLOSER_BASE_URL` / server port stable during a canonical run, or
assign a fresh `PGLLM_RUN_LABEL` so PG-LLM provenance checks stay valid.

6. `scripts/validate_no_eval_tuning.py` checks the tree for obvious evaluation-label artifacts.
