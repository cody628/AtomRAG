# """
# Google Gemini LLM Interface Module
# ==========================

# This module provides interfaces for interacting with Google's Gemini language models,
# including text generation capabilities.

# Author: AtomRAG Team
# Created: 2025-06-16
# License: MIT License

# Copyright (c) 2025 AtomRAG

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# Version: 1.0.0

# Change Log:
# - 1.0.0 (2025-06-16): Initial release
#     * Added text generation support

# Dependencies:
#     - google-generativeai (genai)
#     - tenacity
#     - Python >= 3.10

# Usage:
#     from llm_interfaces.gemini import gemini_complete
# """

# __version__ = "1.0.0"
# __author__ = "AtomRAG Team"
# __status__ = "Production"

# import sys
# import asyncio
# import os

# if sys.version_info < (3, 9):
#     from typing import AsyncIterator
# else:
#     from collections.abc import AsyncIterator

# import pipmaster as pm

# # Install google-generativeai if not installed
# if not pm.is_installed("google-generativeai"):
#     pm.install("google-generativeai")

# import google.generativeai as genai
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type,
# )
# from src.AtomRAG.utils import (
#     logger,
#     safe_unicode_decode,
# )
# from typing import Union

# from src.AtomRAG.types import GEMINIKeywordExtractionFormat, GEMINIQueryExtensionFormat

# # Custom Exception for handling invalid response
# class InvalidGeminiResponseError(Exception):
#     pass

# # Retry decorator for robustness
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((InvalidGeminiResponseError, Exception)),
# )
# async def gemini_complete_if_cache(
#     model,
#     prompt,
#     system_prompt=None,
#     history_messages=None,
#     base_url=None,
#     api_key=None,
#     **kwargs,
# ) -> str:
#     if history_messages is None:
#         history_messages = []

#     if not api_key:
#         api_key = os.environ.get("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("Gemini API key not provided")
    
#     genai.configure(api_key=api_key)

#     # Set API key for client
#     client = genai.GenerativeModel(model, system_instruction=system_prompt)
    
#     kwargs.pop("hashing_kv", None)
#     kwargs.pop("keyword_extraction", None)
#     kwargs.pop("query_extension", None)
    
#     messages = []
#     messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
    
#     full_prompt = ""
#     for message in messages:
#         role = message.get("role", "user")
#         content = message.get("content", "")
#         if role == "system":
#             full_prompt += f"System Instruction: {content}\n\n"
#         elif role == "user":
#             full_prompt += f"User: {content}\n\n"
#         elif role == "assistant":
#             full_prompt += f"Assistant: {content}\n\n"
#         else:
#             full_prompt += f"{role}: {content}\n\n"

#     logger.debug("===== Query Input to Gemini =====")
#     logger.debug(f"Model: {model}")
#     logger.debug(f"Prompt: {prompt}")

#     try:
#         loop = asyncio.get_event_loop()
#         response = await loop.run_in_executor(None, client.generate_content, full_prompt)
#     except Exception as e:
#         logger.error(f"Gemini API Call Failed: {str(e)}")
#         raise
        
#     try:
#         content = response.candidates[0].content.parts[0].text
#     except Exception as e:
#         logger.error("Invalid response structure from Gemini API")
#         raise InvalidGeminiResponseError("Invalid response from Gemini API")

#     if r"\u" in content:
#         content = safe_unicode_decode(content.encode("utf-8"))
#     return content

# # Main interface function
# async def gemini_complete(
#     prompt,
#     system_prompt=None,
#     history_messages=None,
#     keyword_extraction=False,
#     query_extension=False,
#     **kwargs,
# ):
#     if history_messages is None:
#         history_messages = []
#     keyword_extraction = kwargs.pop("keyword_extraction", None)
#     if keyword_extraction:
#         kwargs["response_format"] = GEMINIKeywordExtractionFormat
        
#     query_extension = kwargs.pop("query_extension", None)
#     if query_extension:
#         kwargs["response_format"] = GEMINIQueryExtensionFormat

#     return await gemini_complete_if_cache(
#         "gemini-2.0-flash",
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         **kwargs,
#     )

"""
OpenAI LLM Interface Module
==========================

This module provides interfaces for interacting with openai's language models,
including text generation and embedding capabilities.

Author: AtomRAG team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 AtomRAG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - openai
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.openai import openai_model_complete, openai_embed
"""

__version__ = "1.0.0"
__author__ = "AtomRAG Team"
__status__ = "Production"


import sys
import os

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.AtomRAG.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from src.AtomRAG.types import GPTKeywordExtractionFormat, GPTQueryExtensionFormat
from src.AtomRAG.api import __api_version__

import numpy as np
from typing import Union


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, InvalidResponseError)
    ),
)
async def gemini_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=None,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    if not api_key:
        api_key = os.environ["GEMINI_API_KEY"]

    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) AtomRAG/{__api_version__}",
        "Content-Type": "application/json",
    }
    openai_async_client = (
        AsyncOpenAI(default_headers=default_headers, api_key=api_key)
        if base_url is None
        else AsyncOpenAI(
            base_url=base_url, default_headers=default_headers, api_key=api_key
        )
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("query_extension", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 添加日志输出
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    # logger.debug(f"Messages: {messages}")

    try:
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {str(e)}")
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Error: {str(e)}")
        raise
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"OpenAI API Call Failed: {str(e)}")
        logger.error(f"Model: {model}")
        logger.error(f"Request parameters: {kwargs}")
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            try:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))
                    yield content
            except Exception as e:
                logger.error(f"Error in stream response: {str(e)}")
                raise

        return inner()

    else:
        if (
            not response
            or not response.choices
            or not hasattr(response.choices[0], "message")
            or not hasattr(response.choices[0].message, "content")
        ):
            logger.error("Invalid response from OpenAI API")
            raise InvalidResponseError("Invalid response from OpenAI API")

        content = response.choices[0].message.content

        if not content or content.strip() == "":
            logger.error("Received empty content from OpenAI API")
            raise InvalidResponseError("Received empty content from OpenAI API")

        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content

async def gemini_complete(
    prompt,
    model="gemini-2.5-flash",
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    query_extension=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
        
    query_extension = kwargs.pop("query_extension", None)
    if query_extension:
        kwargs["response_format"] = GPTQueryExtensionFormat
    
    return await gemini_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

import os
import asyncio
import numpy as np
import pipmaster as pm
from google import genai
from google.genai import types

async def gemini_embed(
    texts: list[str],
    model: str = "gemini-embedding-001",
    api_key: str = None,
    output_dimensionality: int = 1536,
) -> np.ndarray:
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)
    loop = asyncio.get_event_loop()

    # (중요) 동기 SDK 호출을 executor로 넘김
    def _embed_one(t: str):
        res = client.models.embed_content(
            model=model,
            contents=t,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
        )
        return res.embeddings[0].values

    # texts가 많으면 여기서 병렬화할 수도 있지만, 우선 안정적으로
    embs = await loop.run_in_executor(None, lambda: [_embed_one(t) for t in texts])

    return np.array(embs, dtype=np.float32)

GEMINI_EMBEDDING_DIM = 1536
GEMINI_MAX_TOKEN_SIZE = 2048
gemini_embed = wrap_embedding_func_with_attrs(
    embedding_dim=GEMINI_EMBEDDING_DIM,
    max_token_size=GEMINI_MAX_TOKEN_SIZE,
)(gemini_embed)