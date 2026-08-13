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
pgllm-run --registry configs/pgllm_closer_model.json --models closer-v1 --sizes 50
pgllm-score --models closer-v1 --sizes 50 --breakdown
```

6. `scripts/validate_no_eval_tuning.py` checks the tree for obvious evaluation-label artifacts.
