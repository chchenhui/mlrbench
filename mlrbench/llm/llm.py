import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast

import anthropic
from anthropic.types import MessageParam, TextBlockParam
from openai import OpenAI

from mlrbench.llm_types import Message

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = {
    # Anthropic models
    "Anthropic":[
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    ],
    # OpenAI models
    "OpenAI":[
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    # web search
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
    ],
    # OpenRouter models
    "OpenRouter":[
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-scout",
    "google/gemini-2.5-pro-exp-03-25",
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.5-flash-preview:thinking",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-001",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwq-32b:free",
    "qwen/qwq-32b",
    "x-ai/grok-2-1212",
    "x-ai/grok-3-mini-beta",
    "x-ai/grok-3-beta",
    "all-hands/openhands-lm-32b-v0.1",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-235b-a22b",
    "mistral/ministral-8b",
    "openai/codex-mini",
    ],
}


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass

    @abstractmethod
    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        pass


class OpenAILLM(LLM):
    r"""An LLM class for the OpenAI LLMs."""

    def __init__(
        self,
        model_name: str = "o3-mini-2025-01-31",
        api_key: Optional[str] = None,
        max_tokens: int = MAX_NUM_TOKENS,
        json_mode: bool = False,
        **kwargs: Any,
    ):
        if not api_key:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # o1 does not use max_tokens
        if "max_tokens" not in kwargs and not (
            model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4")
        ):
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, Sequence[Message]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Message],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        """Chat with the LLM model.

        Parameters:
            chat (Squence[Dict[str, str]]): A list of dictionaries containing the chat
                messages. The messages can be in the format:
                [{"role": "user", "content": "Hello!"}, ...]
        """
        fixed_chat = []
        for c in chat:
            fixed_c = {"role": c["role"]}
            fixed_c["content"] = [{"type": "text", "text": c["content"]}]  # type: ignore
            fixed_chat.append(fixed_c)
        
        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=fixed_chat, **tmp_kwargs  # type: ignore
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content  # type: ignore
                    yield chunk_message

            return f()
        else:
            return cast(str, response.choices[0].message.content)
    
    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        message: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.chat.completions.create(
            model=self.model_name, messages=message, **tmp_kwargs  # type: ignore
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content  # type: ignore
                    yield chunk_message

            return f()
        else:
            token_usage = {"completion_tokens": response.usage.completion_tokens, 
                           "prompt_tokens": response.usage.prompt_tokens,
                           "total_tokens": response.usage.total_tokens}
            return cast(str, response.choices[0].message.content), token_usage


class AnthropicLLM(LLM):
    r"""An LLM class for Anthropic's LLMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = MAX_NUM_TOKENS,
        **kwargs: Any,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs

    def __call__(
        self,
        input: Union[str, Sequence[Dict[str, Any]]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def chat(
        self,
        chat: Sequence[Dict[str, Any]],
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        messages: List[MessageParam] = []
        for msg in chat:
            content: List[TextBlockParam] = [
                TextBlockParam(type="text", text=msg["content"])
            ]
            messages.append({"role": msg["role"], "content": content})

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.messages.create(
            model=self.model_name, messages=messages, **tmp_kwargs
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    if (
                        chunk.type == "message_start"
                        or chunk.type == "content_block_start"
                    ):
                        continue
                    elif chunk.type == "content_block_delta":
                        yield chunk.delta.text
                    elif chunk.type == "message_stop":
                        yield None

            return f()
        else:
            return cast(str, response.content[0].text)

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        content: List[TextBlockParam] = [
            TextBlockParam(type="text", text=prompt)
        ]

        # prefers kwargs from second dictionary over first
        tmp_kwargs = self.kwargs | kwargs
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            **tmp_kwargs,
        )
        if "stream" in tmp_kwargs and tmp_kwargs["stream"]:

            def f() -> Iterator[Optional[str]]:
                for chunk in response:
                    if (
                        chunk.type == "message_start"
                        or chunk.type == "content_block_start"
                    ):
                        continue
                    elif chunk.type == "content_block_delta":
                        yield chunk.delta.text
                    elif chunk.type == "message_stop":
                        yield None

            return f()
        else:
            token_usage = {"cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                           "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                           "input_tokens": response.usage.input_tokens, 
                           "output_tokens": response.usage.output_tokens}
            return cast(str, response.content[0].text), token_usage


class OpenRouterLLM(OpenAILLM):
    r"""An LLM class for the OpenRouter LLMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "google/gemini-2.0-flash-001",
        max_tokens: int = MAX_NUM_TOKENS,
        json_mode: bool = False,
        **kwargs: Any,
    ):
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs


def create_client(
    model_name,
    max_tokens=MAX_NUM_TOKENS,
    judge_mode=False,
):  
    k = None
    for key, value in AVAILABLE_LLMS.items():
        if model_name in value:
            k = key
            continue
    if k == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
            reasoning_effort = "high"
            return OpenAILLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        else:
            return OpenAILLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif k == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if judge_mode:
            return AnthropicLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, temperature=0.0)
        else:
            return AnthropicLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif k == "OpenRouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if judge_mode:
            return OpenRouterLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, json_mode=True, temperature=0.0)
        else:
            return OpenRouterLLM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    else:
        raise ValueError(f"Model {model_name} not supported.")