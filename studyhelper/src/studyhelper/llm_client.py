"""Thin Gemini client (OpenAI-compatible endpoint) with retry + model fallback."""
import logging
import random
import time
from typing import Any, Optional

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .config import settings

log = logging.getLogger(__name__)
TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError)


class LLMUnavailable(Exception):
    """Raised when primary and fallback models both fail."""


class LLMClient:
    def __init__(self, client: Optional[OpenAI] = None, max_retries: int = 3, base_delay: float = 1.0):
        self.client = client or OpenAI(
            api_key=settings.api_key or "missing", base_url=settings.base_url, timeout=30
        )
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _call_with_retry(self, model: str, **kwargs) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(model=model, **kwargs)
            except TRANSIENT as e:
                retryable = True
                err = e
            except APIStatusError as e:
                retryable = e.status_code >= 500
                err = e
            if not retryable or attempt == self.max_retries:
                raise err
            delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)  # backoff + jitter
            log.warning("LLM call failed (%s); retry %d in %.1fs", err, attempt + 1, delay)
            time.sleep(delay)

    def chat(self, messages: list, tools: Optional[list] = None,
             json_mode: bool = False, temperature: Optional[float] = None,
             top_p: Optional[float] = None) -> Any:
        kwargs: dict = dict(
            messages=messages,
            temperature=settings.temperature if temperature is None else temperature,
            top_p=settings.top_p if top_p is None else top_p,
            max_tokens=settings.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last_err: Optional[Exception] = None
        for model in (settings.primary_model, settings.fallback_model):
            try:
                resp = self._call_with_retry(model, **kwargs)
                resp.model_used = model
                return resp
            except Exception as e:  # noqa: BLE001 - degrade to fallback, then fail gracefully
                log.error("Model %s failed: %s", model, e)
                last_err = e
        raise LLMUnavailable(str(last_err))

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            resp = self._embed_with_retry(batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def _embed_with_retry(self, batch):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.embeddings.create(model=settings.embed_model, input=batch)
            except TRANSIENT as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.base_delay * (2 ** attempt) + random.uniform(0, 0.5))
