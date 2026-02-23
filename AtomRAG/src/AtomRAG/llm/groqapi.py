"""
Groq LLM Interface Module
==========================

This module provides interfaces for interacting with Groq's language models,
including text generation and streaming capabilities.
"""

__version__ = "1.0.0"
__author__ = "AtomRAG Team"
__status__ = "Production"

import sys

from groq import Groq

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.AtomRAG.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from src.AtomRAG.api import __api_version__
from src.AtomRAG.utils import extract_reasoning
from typing import Union

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def groq_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = kwargs.pop("stream", False)
    reasoning_tag = kwargs.pop("reasoning_tag", None)
    kwargs.pop("hashing_kv", None)
    kwargs.pop("query_extension", None)  # 불필요한 인자 제거
    api_key = kwargs.pop("api_key", None)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"AtomRAG/{__api_version__}",
    }

    client = Groq(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if stream:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            top_p=1,
            stream=True,
            max_completion_tokens=1024,
        )

        async def inner():
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ""

        return inner()
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            top_p=1,
            stream=False,
            max_completion_tokens=1024,
        )
        model_response = completion.choices[0].message.content

        return (
            model_response
            if reasoning_tag is None
            else extract_reasoning(model_response, reasoning_tag).response_content
        )


async def groq_llama_4_scout_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await groq_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )