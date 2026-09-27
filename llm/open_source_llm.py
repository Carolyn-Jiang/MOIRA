"""Open-weight models served through a Chat Completions endpoint (vLLM/Ollama/etc.)."""
from __future__ import annotations

import json
import math
import os
import re
import time
from urllib.parse import urlsplit

from llm.pair_trading_llm import PairTradingLLM
from llm.errors import InferenceContractError, InferenceServiceError, TransientInferenceError


def is_transient_service_error(exc):
    from openai import APIConnectionError, APIStatusError
    return isinstance(exc, APIConnectionError) or (
        isinstance(exc, APIStatusError) and
        (exc.status_code in (408, 409, 429) or exc.status_code >= 500))


def thinking_configuration(model, extra_body, explicit=None):
    """Require an explicit mode for hybrid Qwen3; never trust a server default."""
    body = dict(extra_body)
    template = body.get("chat_template_kwargs", {})
    if not isinstance(template, dict):
        raise ValueError("chat_template_kwargs must be a JSON object")
    template = dict(template)
    requested = template.get("enable_thinking")
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("enable_thinking must be a JSON boolean")
    if explicit not in (None, "on", "off"):
        raise ValueError("OPEN_SOURCE_THINKING must be on or off")
    if explicit is not None:
        enabled = explicit == "on"
        if requested is not None and requested != enabled:
            raise ValueError("Conflicting thinking mode and OPEN_SOURCE_EXTRA_BODY")
        requested = enabled
    is_qwen3 = bool(re.match(r"qwen3(?:[-:]|$)", model.rsplit("/", 1)[-1].lower()))
    if is_qwen3 and requested is None:
        raise ValueError("Qwen3 requires an explicit --thinking on/off (or OPEN_SOURCE_THINKING)")
    if requested is not None:
        if not is_qwen3:
            raise ValueError("Thinking on/off batches must contain only hybrid Qwen3 models")
        template["enable_thinking"] = requested
        body["chat_template_kwargs"] = template
    return ("on" if requested else "off") if requested is not None else "not_applicable", body


class OpenSourceLLM(PairTradingLLM):
    """Reuse every Moira prompt and parser, changing only the inference transport.

    Deliberately never inherit OPENAI_API_KEY or OPENAI_BASE_URL: selecting a local
    model must not accidentally send a commercial provider's credentials elsewhere.
    """

    @classmethod
    def from_env(cls, *, provider="open_source", model=None, log_dir=None, seed=None,
                 thinking_mode=None):
        prefix = provider.upper()
        model = model or os.getenv(f"{prefix}_MODEL") or os.getenv("LLM_MODEL") or os.getenv("MODEL")
        base_url = (os.getenv(f"{prefix}_BASE_URL") or os.getenv("OPEN_SOURCE_BASE_URL")
                    or os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1")
        if not model or not base_url:
            raise SystemExit(f"{provider} requires LLM_MODEL (or {prefix}_MODEL) and "
                             f"{prefix}_BASE_URL (or OPEN_SOURCE_BASE_URL, e.g. http://localhost:8000/v1).")
        url = urlsplit(base_url)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment):
            raise SystemExit("OPEN_SOURCE_BASE_URL must be an http(s) URL without "
                             "embedded credentials, query, or fragment.")
        try:
            max_tokens = int(os.getenv("OPEN_SOURCE_MAX_TOKENS", "4096"))
            timeout = float(os.getenv("OPEN_SOURCE_TIMEOUT", "120"))
            extra_body = json.loads(os.getenv("OPEN_SOURCE_EXTRA_BODY", "{}"))
            attempts = int(os.getenv("OPEN_SOURCE_REQUEST_ATTEMPTS", "3"))
            retry_delay = float(os.getenv("OPEN_SOURCE_RETRY_DELAY", "120"))
        except (ValueError, TypeError) as exc:
            raise SystemExit(f"Invalid open_source configuration: {exc}") from exc
        if max_tokens <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise SystemExit("OPEN_SOURCE_MAX_TOKENS and OPEN_SOURCE_TIMEOUT must be positive.")
        if attempts < 1 or not math.isfinite(retry_delay) or retry_delay < 0:
            raise SystemExit("Request attempts must be positive and retry delay nonnegative.")
        allowed = {"chat_template_kwargs", "repetition_penalty", "top_k", "min_p"}
        if not isinstance(extra_body, dict) or set(extra_body) - allowed:
            raise SystemExit(f"OPEN_SOURCE_EXTRA_BODY allows only: {', '.join(sorted(allowed))}")
        try:
            thinking, extra_body = thinking_configuration(
                model, extra_body, thinking_mode or os.getenv("OPEN_SOURCE_THINKING") or None)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        send_seed = os.getenv("OPEN_SOURCE_SEND_SEED", "1")
        if send_seed not in ("0", "1"):
            raise SystemExit("OPEN_SOURCE_SEND_SEED must be 0 or 1.")
        obj = cls(model=model, base_url=base_url,
                  api_key=(os.getenv(f"{prefix}_API_KEY") or os.getenv("OPEN_SOURCE_API_KEY")
                           or os.getenv("LLM_API_KEY") or "local-no-key"),
                  provider=provider, log_dir=log_dir, seed=seed,
                  max_output_tokens=max_tokens)
        obj.timeout = timeout
        obj.send_seed = send_seed == "1"
        obj.extra_body = extra_body
        obj.thinking_mode = thinking
        obj.failed_call_count = 0
        obj.request_attempts = attempts
        obj.retry_delay = retry_delay
        return obj

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout, max_retries=0)
        return self._client

    def _completion_request(self, messages):
        request = super()._completion_request(messages)
        if not self.send_seed:
            request.pop("seed", None)
        request["max_tokens"] = self.max_output_tokens
        if self.extra_body:
            request["extra_body"] = self.extra_body
        return request

    def _call_openai(self, system_prompt, payload):
        from openai import APIStatusError
        for attempt in range(1, self.request_attempts + 1):
            self._last_usage = None
            self._last_response_metadata = None
            try:
                return super()._call_openai(system_prompt, payload)
            except Exception as exc:
                self.failed_call_count += 1
                retryable = is_transient_service_error(exc)
                reason = str(exc) if isinstance(exc, InferenceContractError) else type(exc).__name__
                self._last_response_metadata = {**(self._last_response_metadata or {}),
                    "request_attempt": attempt, "retryable": retryable}
                self._log_call("failed_call", system_prompt, payload, f"FAILED: {reason}")
                if retryable:
                    if attempt == self.request_attempts:
                        raise TransientInferenceError(reason) from exc
                    print(f"REQUEST_RETRY attempt={attempt + 1}/{self.request_attempts} "
                          f"wait_seconds={self.retry_delay:g} error={reason}", flush=True)
                    time.sleep(self.retry_delay)
                elif isinstance(exc, APIStatusError):
                    raise InferenceServiceError(reason) from exc
                else:
                    raise

    def _extract_final_content(self, response):
        """Accept only the server-separated final answer; never strip CoT locally."""
        choices = getattr(response, "choices", None)
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise InferenceContractError("Expected exactly one completion")
        choice = choices[0]
        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)
        reasoning = {name: getattr(message, name, None)
                     for name in ("reasoning", "reasoning_content")}
        populated = {name: value for name, value in reasoning.items()
                     if isinstance(value, str) and value.strip()}
        self._last_response_metadata = {
            "finish_reason": finish_reason, "thinking_mode": self.thinking_mode,
            "reasoning_fields": list(populated),
            "reasoning_characters": {k: len(v) for k, v in populated.items()},
            "server_separation_verified": False,
        }
        if finish_reason != "stop":
            raise InferenceContractError("Incomplete response: finish_reason must be stop "
                                         "(check token budget and server configuration)")
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise InferenceContractError("Missing final message.content")
        if re.search(r"<\s*/?\s*think\b", text, re.IGNORECASE):
            raise InferenceContractError("Reasoning tags leaked into content; configure the "
                                         "server with --reasoning-parser qwen3")
        if any(v is not None and not isinstance(v, str) for v in reasoning.values()):
            raise InferenceContractError("Reasoning fields must be text or null")
        if self.thinking_mode == "on" and not populated:
            raise InferenceContractError("Thinking on requires nonempty server-separated "
                                         "reasoning or reasoning_content")
        if self.thinking_mode != "on" and populated:
            raise InferenceContractError("Unexpected reasoning in a non-thinking/Instruct batch")
        self._last_response_metadata["server_separation_verified"] = True
        return text.strip()

    def validate_server(self):
        """One real probe before training; the same contract is checked on every call."""
        text = self.call_text(
            system_prompt='Return only a JSON object with the final answer: {"answer": integer}.',
            payload={"question": "What is 17 multiplied by 19?"}, task_type="server_preflight")
        try:
            valid = json.loads(text) == {"answer": 323}
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise InferenceContractError("Preflight final content must be exactly the requested JSON answer")
        return {"status": "passed", **self._last_response_metadata}

    def inference_config(self):
        """Non-secret settings saved alongside each seed."""
        return {"base_url": self.base_url, "max_tokens": self.max_output_tokens,
                "timeout": self.timeout, "send_seed": self.send_seed,
                "thinking_mode": self.thinking_mode, "response_contract": "server_separated_v1",
                "extra_body": self.extra_body, "temperature": "server_default"}

    def usage_summary(self):
        return {**super().usage_summary(), "failed_call_count": self.failed_call_count,
                "seed_sent": self.send_seed and self.seed is not None}
