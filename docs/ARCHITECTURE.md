# MechRank-Harness architecture

CLOSER converts a ProteinGym-LLM `ranking-v1` request into a globally consistent ranking without adding Track A information.

```
PG-LLM request
  -> parser (ranking-v1)
  -> mutation compiler (deterministic)
  -> assay interpreter (LLM, schema-validated)
  -> mechanistic evidence analyzer (batched LLM)
  -> evidence state store
  -> comparison scheduler (ordinal prior is grouping-only)
  -> comparative reasoner (setwise + pairwise)
  -> preference graph (append-only observations)
  -> Bradley-Terry solver
  -> adaptive auditor (cycles / low-margin / evidence conflict / epistasis)
  -> final ranking validator
  -> {"ranking":[...]}
```

Inner LLM traffic uses `CLOSER_BASE_URL`. The outer OpenAI-compatible server uses a different bind address and model id (`closer-v1`). Recursive configuration is rejected at startup.

Ablation flags in `configs/*.yaml` disable modules by taking defined experimental paths (direct listwise rank, Borda aggregation, deterministic assay/evidence placeholders). Failures still raise typed errors; they never silently fall back to heuristics.
