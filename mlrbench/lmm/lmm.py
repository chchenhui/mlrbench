# https://github.com/landing-ai/vision-agent/blob/main/vision_agent/lmm/lmm.py
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
import base64

import anthropic
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from openai import OpenAI

from mlrbench.lmm_types import Message
from mlrbench.utils.image_utils import encode_media

MAX_NUM_TOKENS = 4096

AVAILABLE_LMMS = {
    # Anthropic models
    "Anthropic":[
    "claude-3-7-sonnet-20250219",
    ],
    # OpenAI models
    "OpenAI":[
    "o4-mini-2025-04-16",
    ],
    # OpenRouter models
    "OpenRouter":[
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-pro-preview-05-06",
    "openai/codex-mini",
    ],
}


class LMM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
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


class AnthropicLMM(LMM):
    r"""An LMM class for Anthropic's LMMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-7-sonnet-20250219",
        max_tokens: int = 4096,
        image_size: int = 768,
        **kwargs: Any,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.image_size = image_size
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

    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
        **kwargs: Any,
    ) -> Union[str, Iterator[Optional[str]]]:
        content: List[Union[TextBlockParam, ImageBlockParam]] = [
            TextBlockParam(type="text", text=prompt)
        ]
        if media:
            for m in media:
                resize = kwargs["resize"] if "resize" in kwargs else self.image_size
                encoded_media = encode_media(m, resize=resize)
                if encoded_media.startswith("data:image/png;base64,"):
                    encoded_media = encoded_media[len("data:image/png;base64,") :]
                content.append(
                    ImageBlockParam(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_media,
                        },
                    )
                )

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


class OpenAILMM(LMM):
    r"""An LMM class for the OpenAI LMMs."""

    def __init__(
        self,
        model_name: str = "o4-mini-2025-04-16",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
        image_size: int = 768,
        image_detail: str = "low",
        **kwargs: Any,
    ):
        if not api_key:
            self.client = OpenAI()
        else:
            self.client = OpenAI(api_key=api_key)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail
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
    
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Union[str, Path]]] = None,
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
        if media and len(media) > 0 and self.model_name != "o3-mini":
            for m in media:
                resize = kwargs["resize"] if "resize" in kwargs else None
                image_detail = (
                    kwargs["image_detail"]
                    if "image_detail" in kwargs
                    else self.image_detail
                )
                encoded_media = encode_media(m, resize=resize)
                message[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                encoded_media
                                if encoded_media.startswith(("http", "https"))
                                or encoded_media.startswith("data:image/")
                                else f"data:image/png;base64,{encoded_media}"
                            ),
                            "detail": image_detail,
                        },
                    },
                )

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


class OpenRouterLMM(OpenAILMM):
    r"""An LLM class for the OpenRouter LLMs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "google/gemini-2.5-pro-preview",
        max_tokens: int = MAX_NUM_TOKENS,
        json_mode: bool = False,
        image_size: int = 768,
        image_detail: str = "low",
        **kwargs: Any,
    ):
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model_name = model_name
        self.image_size = image_size
        self.image_detail = image_detail

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        self.kwargs = kwargs
        

def create_lmm_client(
    model_name,
    max_tokens=MAX_NUM_TOKENS,
    judge_mode=False,
):  
    k = None
    for key, value in AVAILABLE_LMMS.items():
        if model_name in value:
            k = key
            continue
    if k == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if judge_mode:
            return AnthropicLMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, temperature=0.0)
        else:
            return AnthropicLMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif k == "OpenRouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if judge_mode:
            return OpenRouterLMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, json_mode=True, temperature=0.0)
        else:
            return OpenRouterLMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    elif k == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
            reasoning_effort = "high"
            return OpenAILMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        else:
            return OpenAILMM(model_name=model_name, api_key=api_key, max_tokens=max_tokens)
    else:
        raise ValueError(f"Model {model_name} not supported.")
