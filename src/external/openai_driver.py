import asyncio
from typing import Literal
from openai.types import FileObject
from sl.llm.data_models import LLMResponse, Prompt
from sl import config
import openai


_client = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return _client


async def sample(
    model_id: str,
    prompt: Prompt,
    **kwargs,
) -> LLMResponse:
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs["max_tokens"]
        del kwargs["max_tokens"]

    api_response = await get_client().chat.completions.create(
        messages=[m.model_dump() for m in prompt.messages],
        model=model_id,
        **kwargs,
    )
    choice = api_response.choices[0]

    if choice.message.content is None or choice.finish_reason is None:
        raise RuntimeError(f"No content or finish reason for {model_id}")
    return LLMResponse(
        model_id=model_id,
        completion=choice.message.content,
        stop_reason=choice.finish_reason,
        logprobs=None,
    )


async def upload_file(file_path: str, purpose: Literal["fine-tune"]) -> FileObject:
    client = get_client()
    with open(file_path, "rb") as f:
        file_obj = await client.files.create(file=f, purpose=purpose)

    while True:
        file_obj = await client.files.retrieve(file_obj.id)
        if file_obj.status == "processed":
            return file_obj
        await asyncio.sleep(10)
