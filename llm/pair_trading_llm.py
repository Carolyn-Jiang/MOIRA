from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from strategies.types import Action, PositionState

_PROMPT_DIRNAME = "prompts"
_PAIR_PROMPT_NAME = "pair_selection.txt"
_ACTION_PROMPT_NAME = "action_selection.txt"

_ACTION_LABELS = {
    Action.long_short: "LONG_SHORT",
    Action.short_long: "SHORT_LONG",
    Action.flat: "FLAT",
}
_LABEL_TO_ACTION = {v.lower(): k for k, v in _ACTION_LABELS.items()}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

PriceHistoryType = Union[
    Mapping[Union[date, datetime, str], Mapping[str, float]],
    Sequence[Tuple[Union[date, datetime, str], Mapping[str, float]]],
]

NewsType = Union[
    Mapping[str, List[str]],
    Mapping[Union[date, datetime, str], Mapping[str, List[str]]],
]


@dataclass(frozen=True)
class PairSelection:
    pair: Tuple[str, str]
    rationale: str


@dataclass(frozen=True)
class ActionSelection:
    action: Action
    rationale: str


def _prompt_dir(base_dir: Optional[str]) -> str:
    if base_dir:
        return base_dir
    return os.path.join(os.path.dirname(__file__), _PROMPT_DIRNAME)


def _load_prompt(prompt_dir: str, filename: str) -> str:
    path = os.path.join(prompt_dir, filename)
    with open(path, "r") as f:
        return f.read()


def _infer_task_type(system_prompt: str) -> str:
    """Best-effort LLM task label from the system prompt (for call logging)."""
    s = (system_prompt or "").lower()
    if "statistical arbitrage" in s and "select" in s:
        return "pair_selection"
    if "market-neutral pair trader" in s:
        return "trading_action"
    if "pre-break" in s or "relationship breaking" in s or "relationship_status" in s:
        return "relationship_break"
    if "pair reflection" in s:
        return "pair_reflection"
    if "trading coach" in s or "overall reflection" in s:
        return "policy_reflection"
    if "tune a pair-trading policy" in s:
        return "policy_update"
    return "other"


def _extract_json(text: str) -> Dict[str, object]:
    if not text:
        raise ValueError("Empty response text")
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)
    match = _JSON_RE.search(cleaned)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    return [_normalize_symbol(s) for s in symbols]


def _to_date_str(value: Union[date, datetime, str]) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _is_nan(value: object) -> bool:
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _normalize_price_history(
    price_history: PriceHistoryType,
    symbols: Sequence[str],
    *,
    max_points: int = 30,
) -> List[Dict[str, object]]:
    if isinstance(price_history, Mapping):
        items = list(price_history.items())
        items.sort(key=lambda item: _to_date_str(item[0]))
    else:
        items = list(price_history)

    rows: List[Dict[str, object]] = []
    symbols_norm = _normalize_symbols(symbols)
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Price history entries must be (date, prices) tuples")
        date_value, prices = item
        if isinstance(prices, Mapping) and "price" in prices:
            prices = prices["price"]
        if not isinstance(prices, Mapping):
            raise ValueError("Price history prices must be a mapping of symbol to price")
        price_row = {}
        for symbol in symbols_norm:
            val = prices.get(symbol)
            if val is None:
                val = prices.get(symbol.lower())
            if val is None or _is_nan(val):
                continue
            price_row[symbol] = float(val)
        if not price_row:
            continue
        rows.append({"date": _to_date_str(date_value), "prices": price_row})

    if max_points and len(rows) > max_points:
        rows = rows[-max_points:]
    return rows


def _normalize_news(
    news: Optional[NewsType],
    *,
    allowed_symbols: Optional[Sequence[str]] = None,
    max_items_per_ticker: int = 5,
) -> Dict[str, List[str]]:
    if not news:
        return {}

    allowed_set: Optional[set[str]] = None
    if allowed_symbols is not None:
        allowed_set = set(_normalize_symbols(allowed_symbols))

    if all(isinstance(v, dict) for v in news.values()):
        # Handle shape like {date: {ticker: [..]}} or {date: {"news": {ticker: [..]}}}
        if any(isinstance(v, dict) and "news" in v for v in news.values()):
            news = {k: (v.get("news", {}) if isinstance(v, dict) else {}) for k, v in news.items()}

        per_ticker: Dict[str, List[str]] = {}
        date_items = list(news.items())
        date_items.sort(key=lambda item: _to_date_str(item[0]))

        for _, ticker_map in date_items:
            if not isinstance(ticker_map, Mapping):
                continue
            for ticker, items in ticker_map.items():
                if not isinstance(items, list):
                    continue
                tnorm = _normalize_symbol(str(ticker))
                if allowed_set is not None and tnorm not in allowed_set:
                    continue
                bucket = per_ticker.setdefault(tnorm, [])
                for item in items:
                    if not item or item in bucket:
                        continue
                    bucket.append(str(item))
    else:
        # Handle shape like {ticker: [..]}
        per_ticker = {
            _normalize_symbol(str(ticker)): [str(item) for item in items if item]
            for ticker, items in news.items()
            if isinstance(items, list)
        }
        if allowed_set is not None:
            per_ticker = {k: v for k, v in per_ticker.items() if k in allowed_set}

    trimmed: Dict[str, List[str]] = {}
    for ticker, items in per_ticker.items():
        trimmed[ticker] = items[-max_items_per_ticker:] if max_items_per_ticker else items
    return trimmed


def _normalize_open_positions(
    open_positions: Optional[Mapping[str, Mapping[str, object]]],
) -> Dict[str, Dict[str, float]]:
    if not open_positions:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for symbol, info in open_positions.items():
        if not isinstance(info, Mapping):
            continue
        sym = _normalize_symbol(str(symbol))
        item: Dict[str, float] = {}
        for key in ("entry_price", "shares", "current_price"):
            val = info.get(key)
            if val is None or _is_nan(val):
                continue
            if isinstance(val, (int, float)):
                item[key] = float(val)
        if item:
            out[sym] = item
    return out


def _coerce_action_value(value: object, allowed_actions: Sequence[Action]) -> Action:
    if isinstance(value, Action):
        action = value
    elif isinstance(value, int):
        action = Action(value)
    elif isinstance(value, str):
        cleaned = value.strip().lower()
        cleaned = cleaned.replace("action.", "")
        if cleaned not in _LABEL_TO_ACTION:
            raise ValueError(f"Unknown action label: {value}")
        action = _LABEL_TO_ACTION[cleaned]
    else:
        raise ValueError("Unsupported action value type")

    if action not in allowed_actions:
        raise ValueError(f"Action {action} not in allowed actions {allowed_actions}")
    return action


def _position_state_label(value: Optional[Union[PositionState, str]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, PositionState):
        return value.name
    return str(value).strip().lower()


def parse_pair_selection(text: str, allowed_symbols: Sequence[str]) -> PairSelection:
    data = _extract_json(text)
    pair_value = data.get("pair") or data.get("tickers")
    if not pair_value:
        raise ValueError("Missing 'pair' in response JSON")
    if isinstance(pair_value, str):
        parts = [p.strip() for p in pair_value.split(",") if p.strip()]
    else:
        parts = list(pair_value)
    if len(parts) != 2:
        raise ValueError(f"Expected 2 tickers, got {parts}")
    normalized_allowed = set(_normalize_symbols(allowed_symbols))
    pair = tuple(_normalize_symbol(p) for p in parts)
    if pair[0] == pair[1]:
        raise ValueError("Pair tickers must be distinct")
    if pair[0] not in normalized_allowed or pair[1] not in normalized_allowed:
        raise ValueError(f"Pair tickers must be in allowed symbols: {allowed_symbols}")
    rationale = str(data.get("rationale", "")).strip()
    return PairSelection(pair=pair, rationale=rationale)


def parse_action_selection(
    text: str,
    allowed_actions: Sequence[Action],
) -> ActionSelection:
    data = _extract_json(text)
    action_value = data.get("action")
    if action_value is None:
        raise ValueError("Missing 'action' in response JSON")
    action = _coerce_action_value(action_value, allowed_actions)
    rationale = str(data.get("rationale", "")).strip()

    return ActionSelection(action=action, rationale=rationale)


class PairTradingLLM:
    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        prompts_dir: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 400,
        provider: Optional[str] = None,
        log_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("OPEN_SOURCE_BASE_URL") or "http://localhost:11434/v1"
        self.provider = provider or "open_source"
        self.model = model
        self.api_key = api_key or os.getenv("OPEN_SOURCE_API_KEY") or "local-no-key"
        self.prompts_dir = _prompt_dir(prompts_dir)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.log_dir = log_dir          # if set, every call is logged (Step 11)
        env_seed = os.getenv("LLM_SEED")
        self.seed = int(env_seed) if seed is None and env_seed else seed
        self.call_count = 0
        self.usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.system_fingerprints = set()
        self._client = None

        self.pair_prompt = _load_prompt(self.prompts_dir, _PAIR_PROMPT_NAME)
        self.action_prompt = _load_prompt(self.prompts_dir, _ACTION_PROMPT_NAME)
    # in pair_trading_llm.py inside class PairTradingLLM

    def set_action_prompt(self, prompt: str) -> None:
        self.action_prompt = prompt

    def set_pair_prompt(self, prompt: str) -> None:
        self.pair_prompt = prompt

    def call_text(self, *, system_prompt: str, payload: dict,
                  task_type: Optional[str] = None) -> str:
        """Call the LLM with a system prompt + JSON payload; return raw text.

        This is the single entry point every PolicyAgent LLM call goes through
        (pair selection, trading action, relationship-break, reflections). When a
        ``log_dir`` was configured, the call is recorded (Step 11).
        """
        response = self._call_openai(system_prompt, payload)
        self._log_call(task_type or _infer_task_type(system_prompt),
                       system_prompt, payload, response)
        return response

    def _log_call(self, task_type: str, system_prompt: str, payload: dict,
                  response: str) -> None:
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "provider": self.provider,
                "model": self.model,
                "task_type": task_type,
                "mode": "real_api",
                "prompt": system_prompt,
                "payload": payload,
                "response": response,
                "usage": getattr(self, "_last_usage", None),
                "response_metadata": getattr(self, "_last_response_metadata", None),
            }
            with open(os.path.join(self.log_dir, "llm_calls.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass  # logging must never break a run

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Missing server authentication key."
                )
            from openai import OpenAI

            if self.base_url:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call_openai(self, system_prompt: str, payload: dict) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'openai'. Install it to use PairTradingLLM."
            ) from e

        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]

        # Section 7: do NOT set temperature -- use the provider default.
        request = self._completion_request(messages)
        resp = client.chat.completions.create(**request)

        # Capture token usage for cost accounting (instrumentation only).
        u = getattr(resp, "usage", None)
        self._last_usage = ({
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        } if u is not None else None)
        self.call_count += 1
        if self._last_usage:
            for key in self.usage_totals:
                self.usage_totals[key] += int(self._last_usage.get(key) or 0)
        fingerprint = getattr(resp, "system_fingerprint", None)
        if fingerprint:
            self.system_fingerprints.add(str(fingerprint))

        return self._extract_final_content(resp)

    def _extract_final_content(self, response) -> str:
        """Transport-specific response validation before any task parser or memory."""
        return response.choices[0].message.content

    def _completion_request(self, messages: list) -> dict:
        request = {"model": self.model, "messages": messages}
        if self.seed is not None:
            # Chat Completions treats seed as best-effort, not strict determinism.
            request["seed"] = self.seed
        return request

    def usage_summary(self) -> Dict[str, object]:
        """Return aggregate telemetry without persisting prompts or responses."""
        return {
            "call_count": self.call_count,
            **self.usage_totals,
            "system_fingerprints": sorted(self.system_fingerprints),
            "seed": self.seed,
        }

    def select_pair(
        self,
        *,
        symbols: Sequence[str],
        recent_prices: PriceHistoryType,
        recent_news: Optional[NewsType] = None,
        max_price_points: int = 30,
        max_news_items: int = 5,
    ) -> PairSelection:
        allowed = _normalize_symbols(symbols)
        payload = {
            "allowed_symbols": allowed,
            "recent_prices": _normalize_price_history(
                recent_prices, allowed, max_points=max_price_points
            ),
            "recent_news": _normalize_news(
                recent_news, allowed_symbols=allowed, max_items_per_ticker=max_news_items
            ),
        }
        text = self._call_openai(self.pair_prompt, payload)
        return parse_pair_selection(text, allowed)

    def decide_action(
        self,
        *,
        trade_pair: Tuple[str, str],
        position_state: Optional[Union[PositionState, str]],
        permissible_actions: Optional[Sequence[Action]] = None,
        recent_prices: PriceHistoryType,
        recent_news: Optional[NewsType] = None,
        open_positions: Optional[Mapping[str, Mapping[str, object]]] = None,
        max_price_points: int = 10,
        max_news_items: int = 5,
    ) -> ActionSelection:
        allowed_actions = list(permissible_actions or list(Action))
        trade_pair_norm = (_normalize_symbol(trade_pair[0]), _normalize_symbol(trade_pair[1]))
        payload = {
            "trade_pair": [trade_pair_norm[0], trade_pair_norm[1]],
            "position_state": _position_state_label(position_state),
            "permissible_actions": [_ACTION_LABELS[a] for a in allowed_actions],
            "action_semantics": {
                "LONG_SHORT": "long first ticker, short second",
                "SHORT_LONG": "short first ticker, long second",
                "FLAT": "no exposure; close both legs (chosen explicitly each day)",
            },
            "recent_prices": _normalize_price_history(
                recent_prices, trade_pair_norm, max_points=max_price_points
            ),
            "recent_news": _normalize_news(
                recent_news, allowed_symbols=trade_pair_norm, max_items_per_ticker=max_news_items
            ),
            "open_positions": _normalize_open_positions(open_positions),
        }
        text = self._call_openai(self.action_prompt, payload)
        return parse_action_selection(text, allowed_actions)
