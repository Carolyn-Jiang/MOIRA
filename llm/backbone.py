"""Model-provider routing; model names pass unchanged to the server."""
from __future__ import annotations
import os
import re

PROVIDERS = {
    "openai": {"family": "OpenAI", "self_hosted": False},
    "deepseek": {"family": "DeepSeek", "self_hosted": False},
    "gemini": {"family": "Gemini", "self_hosted": False},
    "llama": {"family": "Meta Llama", "self_hosted": True},
    "qwen": {"family": "Qwen", "self_hosted": True},
    "open_source": {"family": "Other open-weight models", "self_hosted": True},
}


def infer_family(model: str) -> str:
    name = model.strip().lower()
    if not name or name.startswith("-") or any(c.isspace() for c in name):
        raise ValueError("Supply a non-empty served model ID without whitespace")
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen"
    if "deepseek" in name:
        return "deepseek"
    if name.startswith("gemini"):
        return "gemini"
    if re.search(r"(^|/)(gpt-(?!oss)|o[134](?:-|$))", name):
        return "openai"
    if "claude" in name:
        raise ValueError("This release supports the paper's OpenAI, DeepSeek and Gemini APIs")
    return "open_source"


def supported_providers():
    return dict(PROVIDERS)


def resolve_provider(provider=None):
    name = (provider or os.getenv("LLM_PROVIDER") or "open_source").lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unsupported provider {name!r}; choose {', '.join(PROVIDERS)}")
    return name, PROVIDERS[name]


def build_llm(provider=None, model=None, log_dir=None, seed=None, thinking_mode=None):
    from llm.open_source_llm import OpenSourceLLM
    family, _ = resolve_provider(provider)
    if family in ("openai", "deepseek", "gemini"):
        from llm.closed_source_llm import ClosedSourceLLM
        return ClosedSourceLLM.from_env(provider=family, model=model, log_dir=log_dir,
                                       seed=seed, thinking_mode=thinking_mode)
    return OpenSourceLLM.from_env(provider=family, model=model, log_dir=log_dir, seed=seed,
                                  thinking_mode=thinking_mode)
