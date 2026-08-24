"""OpenAI-compatible Chat Completions client."""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
from pydantic import BaseModel

from closer.config import CloserConfig
from closer.errors import ProviderError, StructuredOutputError
from closer.llm.base import LLMClient, extract_output_text, extract_usage
from closer.llm.retry import retrying
from closer.llm.structured import json_only_system, parse_model, repair_user_prompt
from closer.logging import dump_json, sha256_text
from closer.orchestrator.budget import BudgetManager
from closer.schemas.trace import LLMCallTrace, TokenUsage


class OpenAIChatClient(LLMClient):
    def __init__(self, config: CloserConfig, *, trace_dir=None) -> None:
        super().__init__(config, trace_dir=trace_dir)
        self._client = httpx.AsyncClient(timeout=config.base_model.timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate_text(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        budget: BudgetManager | None = None,
        max_output_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, LLMCallTrace]:
        payload, trace = await self._complete(
            stage=stage,
            system=system,
            user=user,
            budget=budget,
            max_output_tokens=max_output_tokens,
            extra_body=extra_body,
        )
        text = extract_output_text(payload)
        if not text.strip():
            raise ProviderError(f"{stage}: empty or truncated response")
        finish = ((payload.get("choices") or [{}])[0]).get("finish_reason")
        if finish in {"length", "max_tokens"}:
            raise ProviderError(f"{stage}: truncated response (finish_reason={finish})")
        return text, trace

    async def generate_structured(
        self,
        *,
        stage: str,
        schema: type[BaseModel],
        system: str,
        user: str,
        budget: BudgetManager | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[BaseModel, LLMCallTrace]:
        last_error: Exception | None = None
        last_text = ""
        attempts = 1 + self.config.structured_output.max_parse_repairs
        current_system = json_only_system(schema, system)
        current_user = user
        extra_body: dict[str, Any] | None = {"response_format": {"type": "json_object"}}
        if not self.config.structured_output.native_json_schema:
            extra_body = None
        for attempt in range(attempts):
            try:
                text, trace = await self.generate_text(
                    stage=stage if attempt == 0 else f"{stage}:repair",
                    system=current_system,
                    user=current_user,
                    budget=budget,
                    max_output_tokens=max_output_tokens,
                    extra_body=extra_body,
                )
            except ProviderError as exc:
                if extra_body is not None:
                    extra_body = None
                    last_error = exc
                    continue
                raise
            last_text = text
            try:
                parsed = parse_model(schema, text)
                trace.parsed_response = parsed.model_dump(mode="json")
                trace.repair_count = attempt
                return parsed, trace
            except StructuredOutputError as exc:
                last_error = exc
                current_system = (
                    "You are repairing malformed JSON so it matches a required schema. "
                    "Return JSON only."
                )
                current_user = repair_user_prompt(schema, text, str(exc))
                extra_body = None
        raise StructuredOutputError(
            f"{stage}: structured output failed after {attempts} attempts: {last_error}\n"
            f"Last output: {last_text[:500]}"
        ) from last_error

    async def _complete(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        budget: BudgetManager | None,
        max_output_tokens: int | None,
        extra_body: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], LLMCallTrace]:
        if budget is not None:
            budget.precheck()
        body: dict[str, Any] = {
            "model": self.config.model_id(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        if self.config.base_model.temperature is not None:
            body["temperature"] = self.config.base_model.temperature
        if extra_body:
            body.update(extra_body)
        call_id = uuid.uuid4().hex[:16]
        start = time.perf_counter()
        retry_count = 0

        @retrying(self.config.base_model.max_retries)
        async def _send() -> httpx.Response:
            nonlocal retry_count
            retry_count += 1
            url = f"{self.config.base_url()}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.api_key()}",
                "Content-Type": "application/json",
            }
            response = await self._client.post(url, headers=headers, json=body)
            if response.status_code >= 400:
                raise ProviderError(
                    f"{stage}: provider HTTP {response.status_code}: {response.text[:500]}"
                )
            return response

        try:
            response = await _send()
        except Exception as exc:
            if not isinstance(exc, ProviderError):
                raise ProviderError(f"{stage}: provider call failed: {exc}") from exc
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        payload = response.json()
        usage = TokenUsage(**extract_usage(payload))
        if self.config.base_model.require_usage and usage.output_tokens is None:
            raise ProviderError(f"{stage}: provider omitted required usage metadata")
        trace = LLMCallTrace(
            call_id=call_id,
            stage=stage,
            model_requested=self.config.model_id(),
            model_returned=payload.get("model"),
            prompt_hash=sha256_text(system + "\n" + user),
            raw_request=body if self.config.logging.save_raw_provider_payloads else None,
            raw_response=payload if self.config.logging.save_raw_provider_payloads else None,
            usage=usage,
            latency_ms=latency_ms,
            retry_count=max(0, retry_count - 1),
        )
        if self.trace_dir is not None:
            dump_json(self.trace_dir / f"{call_id}.request.json", body)
            dump_json(self.trace_dir / f"{call_id}.response.json", payload)
        if budget is not None:
            budget.record(usage, latency_ms=latency_ms)
        return payload, trace
