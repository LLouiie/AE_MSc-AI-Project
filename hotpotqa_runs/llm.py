import os
from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-32B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

call_counter = 0

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        self.model = kwargs.get('model_name', DEFAULT_MODEL)
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens = kwargs.get('max_tokens', 100)
        self.stop = kwargs.get('model_kwargs', {}).get('stop', None)

    def __call__(self, prompt: str) -> str:
        global call_counter
        call_counter += 1
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop
        )
        return response.choices[0].message.content
