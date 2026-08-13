"""Efficiency diagnostics."""

from __future__ import annotations

from closer.orchestrator.budget import BudgetManager


def efficiency_snapshot(budget: BudgetManager) -> dict:
    snap = budget.snapshot()
    calls = max(1, snap["llm_calls"])
    snap["mean_latency_ms"] = snap["latency_ms"] / calls
    snap["tokens_per_call"] = snap["total_tokens"] / calls
    return snap
