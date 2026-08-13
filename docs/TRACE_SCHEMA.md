# Trace schema

`schema_version`: `closer.trace.v1`

Each episode is written under `artifacts/runs/<run_id>/episodes/<episode_id>/`:

```
input.json
mutation_profiles.json
assay_contract.json
evidence_states.json
evidence/<variant_id>.json
comparisons/round_00.jsonl
comparisons/round_01.jsonl
comparisons/bridges.jsonl
comparisons/audit_rounds.jsonl
graph.json
solver_before_audit.json
solver_after_audit.json
diagnostics.json
final_result.json
llm_calls/<call_id>.request.json
llm_calls/<call_id>.response.json
```

LLM traces include `call_id`, `stage`, `model_requested`, `model_returned`, `prompt_hash`, raw request/response when enabled, parsed JSON, usage, latency, retry count, and repair count.

Run-level `manifest.json` stores config hash, git commit, prompt hashes, base model id, and timestamp.
