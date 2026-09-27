"""Hosted-model transport for the same MOIRA policy and simulation engine."""
from __future__ import annotations
import json
import math
import os
import re
from urllib.parse import urlsplit
from llm.open_source_llm import OpenSourceLLM
from llm.errors import InferenceContractError

ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class ClosedSourceLLM(OpenSourceLLM):
    """Share bounded retries and telemetry, but use each API's request contract.

    Reasoning may be hidden by hosted APIs. Only final message.content reaches
    task parsers and memory; visible reasoning must be in a separate server field.
    Credentials are never inherited from another provider.
    """
    @classmethod
    def from_env(cls, *, provider, model=None, log_dir=None, seed=None, thinking_mode=None):
        prefix = provider.upper()
        model = model or os.getenv(f"{prefix}_MODEL") or os.getenv("LLM_MODEL")
        if not model:
            raise ValueError("An exact served model identifier is required")
        if thinking_mode is not None:
            raise ValueError("--thinking on/off is reserved for hybrid Qwen3")
        endpoint = os.getenv(f"{prefix}_BASE_URL") or ENDPOINTS[provider]
        url = urlsplit(endpoint)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("Base URL must not contain credentials, query or fragment")
        key = os.getenv(f"{prefix}_API_KEY")
        if not key:
            raise ValueError(f"Set {prefix}_API_KEY in the local .env or environment")
        obj = cls(model=model, api_key=key, base_url=endpoint, provider=provider,
                  seed=seed, log_dir=log_dir,
                  max_output_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "4096")))
        obj.timeout = float(os.getenv(f"{prefix}_TIMEOUT", "600"))
        obj.request_attempts = int(os.getenv(f"{prefix}_REQUEST_ATTEMPTS", "3"))
        obj.retry_delay = float(os.getenv(f"{prefix}_RETRY_DELAY", "120"))
        if (obj.max_output_tokens <= 0 or not math.isfinite(obj.timeout) or obj.timeout <= 0
                or obj.request_attempts < 1 or not math.isfinite(obj.retry_delay) or obj.retry_delay < 0):
            raise ValueError("Invalid token, timeout or retry configuration")
        seed_setting = os.getenv(f"{prefix}_SEND_SEED", "1" if provider == "openai" else "0")
        if seed_setting not in ("0", "1"):
            raise ValueError(f"{prefix}_SEND_SEED must be 0 or 1")
        obj.send_seed = seed_setting == "1"
        obj.reasoning_effort = os.getenv(f"{prefix}_REASONING_EFFORT", "low" if provider == "gemini" else "")
        obj.deepseek_thinking = os.getenv("DEEPSEEK_THINKING", "disabled")
        if obj.deepseek_thinking not in ("enabled", "disabled"):
            raise ValueError("DEEPSEEK_THINKING must be enabled or disabled")
        obj.thinking_mode = "provider_managed"
        obj.extra_body = {}
        obj.failed_call_count = 0
        return obj

    def _completion_request(self, messages):
        req = {"model": self.model, "messages": messages}
        if self.seed is not None and self.send_seed:
            req["seed"] = self.seed
        token_key = "max_completion_tokens" if self.provider == "openai" else "max_tokens"
        req[token_key] = self.max_output_tokens
        if self.reasoning_effort:
            req["reasoning_effort"] = self.reasoning_effort
        if self.provider == "deepseek":
            req["extra_body"] = {"thinking": {"type": self.deepseek_thinking}}
        return req

    def _extract_final_content(self, response):
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise InferenceContractError("Expected exactly one completion")
        choice = choices[0]
        if choice.finish_reason != "stop":
            raise InferenceContractError("Incomplete hosted response; check output token limit")
        text = getattr(choice.message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise InferenceContractError("Missing final message.content")
        if re.search(r"<\s*/?\s*think\b", text, re.I):
            raise InferenceContractError("Reasoning leaked into final content; fix the serving endpoint")
        fields = {k: getattr(choice.message, k, None) for k in ("reasoning", "reasoning_content")}
        if any(v is not None and not isinstance(v, str) for v in fields.values()):
            raise InferenceContractError("Reasoning fields must be text or null")
        self._last_response_metadata = {
            "finish_reason": "stop", "thinking_mode": self.thinking_mode,
            "reasoning_fields": [k for k, v in fields.items() if v],
            "reasoning_characters": {k: len(v) for k, v in fields.items() if v},
            "final_content_only": True,
        }
        return text.strip()

    def inference_config(self):
        return {**super().inference_config(), "response_contract": "hosted_final_content_v1",
                "reasoning_effort": self.reasoning_effort or "provider_default",
                "deepseek_thinking": self.deepseek_thinking if self.provider == "deepseek" else None}
