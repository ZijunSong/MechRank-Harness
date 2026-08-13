"""Retry helpers for provider errors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from closer.errors import ProviderError

T = TypeVar("T")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.NetworkError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    if isinstance(exc, ProviderError):
        text = str(exc).lower()
        return any(token in text for token in ("429", "timeout", "temporar", "503", "502", "500"))
    return False


def retrying(max_attempts: int) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception(is_retryable),
    )
