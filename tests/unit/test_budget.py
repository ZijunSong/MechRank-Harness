from closer.config import BudgetConfig
from closer.errors import BudgetExceededError
from closer.orchestrator.budget import BudgetManager
from closer.schemas.trace import TokenUsage


def test_budget_counts_real_usage():
    mgr = BudgetManager(BudgetConfig(max_total_llm_calls=3, max_total_input_tokens=100, max_total_output_tokens=100))
    mgr.record(TokenUsage(input_tokens=10, output_tokens=5), latency_ms=12)
    snap = mgr.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["input_tokens"] == 10
    assert snap["output_tokens"] == 5
    assert snap["total_tokens"] == 15


def test_hard_stop_on_calls():
    mgr = BudgetManager(
        BudgetConfig(max_total_llm_calls=1, max_total_input_tokens=10000, max_total_output_tokens=10000)
    )
    mgr.record(TokenUsage(input_tokens=1, output_tokens=1))
    try:
        mgr.precheck()
    except BudgetExceededError:
        return
    raise AssertionError("call budget must hard-stop")


def test_hard_stop_on_tokens():
    mgr = BudgetManager(
        BudgetConfig(max_total_llm_calls=100, max_total_input_tokens=5, max_total_output_tokens=10000)
    )
    try:
        mgr.record(TokenUsage(input_tokens=6, output_tokens=1))
    except BudgetExceededError as exc:
        assert "input token" in str(exc)
    else:
        raise AssertionError("token budget must hard-stop")
