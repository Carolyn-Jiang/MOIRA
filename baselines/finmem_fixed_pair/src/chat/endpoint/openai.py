import json
import os
import time
from llm.prompt_time import PromptTime
from typing import Any, Dict, Union

import httpx
from loguru import logger
from pydantic import ValidationError

from .base import (
    StructuredGenerationChatEndPoint,
    StructureGenerationFailure,
    StructureOutputResponse,
)


class OpenAIStructureGeneration(StructuredGenerationChatEndPoint):
    """OpenAI Chat Completions endpoint with strict structured output."""

    def __init__(self, chat_config: Dict[str, Any]) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI inference")
        self.request_url = chat_config.get(
            "chat_openai_endpoint", "https://api.openai.com/v1/chat/completions"
        )
        self.chat_model = chat_config["chat_model"]
        self.system_message = chat_config["chat_system_message"]
        self.max_tokens = chat_config["chat_max_new_token"]
        self.timeout = chat_config["chat_request_timeout"]
        self.parameters = chat_config.get("chat_parameters", {})
        self.max_retries = chat_config.get("chat_max_retries", 5)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def __call__(
        self, prompt: str, schema: Any
    ) -> Union[StructureGenerationFailure, StructureOutputResponse]:
        strict_schema = {**schema, "additionalProperties": False}
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": PromptTime().text(self.system_message)},
                {"role": "user", "content": PromptTime().text(prompt)},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "finmem_pair_action",
                    "strict": True,
                    "schema": strict_schema,
                },
            },
            **self.parameters,
        }
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.request_url, headers=self.headers, json=payload
                    )
                response.raise_for_status()
                response_dict = json.loads(
                    response.json()["choices"][0]["message"]["content"]
                )
                for field in (
                    "short_memory_ids",
                    "mid_memory_ids",
                    "long_memory_ids",
                    "reflection_memory_ids",
                ):
                    if field in response_dict:
                        response_dict[field] = list(dict.fromkeys(response_dict[field]))
                return StructureOutputResponse(**response_dict)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    f"CHAT-OpenAI attempt {attempt + 1}/{self.max_retries} failed: {exc}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 16))
        return StructureGenerationFailure()
