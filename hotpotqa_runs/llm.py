import os
from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

call_counter = 0

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get('model_name', DEFAULT_MODEL)
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens = kwargs.get('max_tokens', 100)
        self.stop = kwargs.get('model_kwargs', {}).get('stop', None)
        # Per-instance client: callers (e.g. ae/runners/run_alfworld.py's
        # --base-url/--model CLI flags) pass openai_api_base/openai_api_key
        # explicitly; previously these kwargs were silently dropped in favor
        # of a single module-level client built once from env vars at import
        # time, so --base-url had no effect and every AnyOpenAILLM instance
        # in a process talked to whatever OPENAI_BASE_URL happened to be.
        base_url = kwargs.get('openai_api_base', DEFAULT_BASE_URL)
        api_key = kwargs.get('openai_api_key', DEFAULT_API_KEY)
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, prompt: str) -> str:
        global call_counter
        call_counter += 1
        # Raw completions, not chat.completions: this prompt format (few-shot
        # examples + "> " cursor cue) is designed for continuation, not
        # dialogue. Routing it through a chat template caused two artifacts
        # confirmed empirically on Qwen3-8B (2026-07-25): (1) the model
        # defaults to emitting a <think>...</think> block the tight
        # max_tokens/stop budget has no room for, and (2) even with thinking
        # disabled, completions came back with a spurious leading "> " that
        # broke the agent's `action.startswith('think:')` check. Plain
        # completions reproduces neither.
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        choice = response.choices[0]
        usage = response.usage
        # Exposed as an instance attribute rather than a second return value
        # so every existing call site (`llm(prompt)` used across
        # hotpotqa_runs/fever_runs/alfworld_runs_ae) keeps working unchanged;
        # callers that want the metadata for this specific call read
        # `self.llm.last_meta` immediately after the call.
        self.last_meta = {
            "raw_generation": choice.text,
            "finish_reason": choice.finish_reason,
            "was_truncated": choice.finish_reason == "length",
            "usage": (
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage is not None else "unavailable"
            ),
        }
        return choice.text
