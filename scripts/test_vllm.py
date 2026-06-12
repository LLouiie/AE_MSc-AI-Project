from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello from vLLM."}],
        temperature=0,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
