import os
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List, Union

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> Union[str, List[str]]:
    assert (not is_batched and isinstance(prompt, str)) or (is_batched and isinstance(prompt, list))
    if is_batched:
        return [
            client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": item}],
                temperature=0.0,
                max_tokens=max_tokens,
                stop=stop_strs,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ).choices[0].message.content
            for item in prompt
        ]
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        stop=stop_strs,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content
