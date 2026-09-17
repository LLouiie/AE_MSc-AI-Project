"""OpenAI-compatible completion client shared by AE runners."""

import os
import time
from urllib.parse import urlparse

from openai import BadRequestError, OpenAI, RateLimitError

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

_call_counter = 0


def is_openai_api(base_url: str) -> bool:
    """Return whether *base_url* is OpenAI hosted rather than local vLLM."""
    return (urlparse(base_url).hostname or "").lower() == "api.openai.com"


def chat_completion_params(
    *, model: str, base_url: str, messages: list[dict], temperature: float,
    max_tokens: int, stop=None,
) -> dict:
    """Build Chat Completions parameters supported by the selected backend."""
    params = {
        "model": model,
        "messages": messages,
    }
    if is_openai_api(base_url):
        # GPT reasoning tokens share this budget with visible output.
        params["max_completion_tokens"] = max(4096, max_tokens * 4)
        if model.startswith("gpt-5"):
            params["reasoning_effort"] = "medium"
        else:
            params["temperature"] = temperature
            if stop is not None:
                params["stop"] = stop
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens
        if model.startswith("openai/gpt-oss"):
            # gpt-oss uses this budget for reasoning before visible output.
            params["reasoning_effort"] = "low"
            params["max_tokens"] = max(1024, max_tokens)
        if stop is not None:
            params["stop"] = stop
        params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    return params


def _chat_create_with_rate_retry(client, params):
    for attempt in range(7):
        try:
            return client.chat.completions.create(**params)
        except RateLimitError:
            if attempt == 6:
                raise
            time.sleep(min(30.0, 1.0 * (2 ** attempt)))


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model_name", DEFAULT_MODEL)
        self.temperature = kwargs.get("temperature", 0)
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.stop = kwargs.get("model_kwargs", {}).get("stop")
        self.base_url = kwargs.get("openai_api_base", DEFAULT_BASE_URL)
        self.api_key = kwargs.get("openai_api_key", DEFAULT_API_KEY)
        self.use_chat_completions = (
            is_openai_api(self.base_url)
            or os.getenv("AE_USE_CHAT_COMPLETIONS", "0") == "1"
        )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.last_meta = None

    def __call__(self, prompt: str) -> str:
        global _call_counter
        _call_counter += 1
        if self.use_chat_completions:
            params = chat_completion_params(
                model=self.model,
                base_url=self.base_url,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )
            try:
                response = _chat_create_with_rate_retry(self.client, params)
            except BadRequestError as exc:
                if "model output limit was reached" not in str(exc):
                    raise
                params["max_completion_tokens"] = 8192
                _call_counter += 1
                response = _chat_create_with_rate_retry(self.client, params)
        else:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=self.stop,
            )
        choice = response.choices[0]
        usage = response.usage
        text = choice.message.content if self.use_chat_completions else choice.text
        self.last_meta = {
            "raw_generation": text,
            "finish_reason": choice.finish_reason,
            "was_truncated": choice.finish_reason == "length",
            "requested_model": self.model,
            "resolved_model": response.model,
            "api_mode": "chat_completions" if self.use_chat_completions else "completions",
            "usage": ({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            } if usage is not None else "unavailable"),
        }
        return text


def call_counter() -> int:
    return _call_counter
