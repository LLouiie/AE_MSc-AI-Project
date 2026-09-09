"""OpenAI-compatible completion client shared by AE runners."""

import os
from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

_call_counter = 0


class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model_name", DEFAULT_MODEL)
        self.temperature = kwargs.get("temperature", 0)
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.stop = kwargs.get("model_kwargs", {}).get("stop")
        self.client = OpenAI(
            base_url=kwargs.get("openai_api_base", DEFAULT_BASE_URL),
            api_key=kwargs.get("openai_api_key", DEFAULT_API_KEY),
        )
        self.last_meta = None

    def __call__(self, prompt: str) -> str:
        global _call_counter
        _call_counter += 1
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        choice = response.choices[0]
        usage = response.usage
        self.last_meta = {
            "raw_generation": choice.text,
            "finish_reason": choice.finish_reason,
            "was_truncated": choice.finish_reason == "length",
            "usage": ({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            } if usage is not None else "unavailable"),
        }
        return choice.text


def call_counter() -> int:
    return _call_counter
