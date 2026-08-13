# Experiment protocol

Official ProteinGym-LLM policy: evaluation splits are not for prompt/agent/tool/benchmark-specific tuning.

Allowed during development:

- official repo code, docs, and `ranking-v1` prompt template
- public reasoning traces
- synthetic prompts in `tests/fixtures`
- independent `data/dev_manifest.jsonl`

Ablation modes (same base model):

| Mode | Config |
|---|---|
| A Direct LLM | all ablation flags false except nothing else; `comparative_reasoning=false` |
| B + mutations | `mutation_compiler=true`, comparative false |
| C + assay | also `assay_contract=true` |
| D + evidence | also `evidence_state=true` |
| E + comparative | `comparative_reasoning=true`, `global_solver=false` (Borda) |
| F + BT solver | `global_solver=true`, `adaptive_audit=false` |
| G full CLOSER | all true |

Report observable input+output tokens, call count, and wall-clock. Do not claim hidden-reasoning-token fairness unless the provider reports it.
