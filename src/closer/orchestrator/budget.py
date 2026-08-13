"""Budget accounting for LLM calls and tokens."""

from __future__ import annotations

from closer.config import BudgetConfig
from closer.errors import BudgetExceededError
from closer.schemas.trace import TokenUsage


class BudgetManager:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost = 0.0
        self.latency_ms = 0.0

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost": self.cost if self.cost else None,
            "latency_ms": self.latency_ms,
        }

    def remaining_calls(self) -> int:
        return max(0, self.config.max_total_llm_calls - self.calls)

    def _exceeded(self) -> str | None:
        if self.calls > self.config.max_total_llm_calls:
            return (
                f"LLM call budget exceeded: {self.calls} > {self.config.max_total_llm_calls}"
            )
        if self.input_tokens > self.config.max_total_input_tokens:
            return (
                f"input token budget exceeded: {self.input_tokens} > "
                f"{self.config.max_total_input_tokens}"
            )
        if self.output_tokens > self.config.max_total_output_tokens:
            return (
                f"output token budget exceeded: {self.output_tokens} > "
                f"{self.config.max_total_output_tokens}"
            )
        return None

    def precheck(self) -> None:
        projected_calls = self.calls + 1
        if projected_calls > self.config.max_total_llm_calls and self.config.hard_stop_on_exceed:
            raise BudgetExceededError(
                f"LLM call budget would exceed {self.config.max_total_llm_calls}"
            )

    def record(self, usage: TokenUsage, *, latency_ms: float = 0.0) -> None:
        self.calls += 1
        self.input_tokens += int(usage.input_tokens or 0)
        self.output_tokens += int(usage.output_tokens or 0)
        self.reasoning_tokens += int(usage.reasoning_tokens or 0)
        if usage.cost is not None:
            self.cost += usage.cost
        self.latency_ms += latency_ms
        reason = self._exceeded()
        if reason and self.config.hard_stop_on_exceed:
            raise BudgetExceededError(reason)
