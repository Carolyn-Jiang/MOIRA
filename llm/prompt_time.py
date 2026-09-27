"""Calendar-free views of model inputs; simulation dates are never modified."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
import re


TIME_INSTRUCTION = (
    "Calendar dates are masked as step labels. current_step identifies the current "
    "trading step within this training or test window, starting at step 001. "
    "Historical records use the same numbering. step XXX masks a date outside "
    "the decision window or without a mapped trading step. Do not infer calendar dates. "
    "Use only the supplied observations and settled past-step feedback. Do not "
    "reconstruct historical market outcomes from model pretraining or treat "
    "predicted future outcomes as observed facts."
)

_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
          r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE = re.compile(
    r"(?<![\w.])(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}/\d{1,2}/\d{4}"
    r"|" + _MONTH + r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r"\s+\d{4}"
    r"|" + _MONTH + r"\s+\d{4}"
    r")(?!\w)", re.IGNORECASE)
_YEAR = re.compile(r"(?<![\w.$])(?:FY\s*)?(?:19|20)\d{2}(?![\w.%])", re.IGNORECASE)


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    value = str(value).strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        pass
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bSept\b", "Sep", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "").replace(".", "")
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%b %d %Y", "%B %d %Y",
                "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


class PromptTime:
    def __init__(self, dates=()):
        days = [_day(d) for d in dates]
        if any(d is None for d in days) or days != sorted(set(days)):
            raise ValueError("Prompt calendar must contain unique, ordered decision dates")
        self.labels = {d: f"step {i:03d}" for i, d in enumerate(days, 1)}
        self.current_step = None

    def set_step(self, day):
        self.current_step = self.labels[_day(day)]

    def text(self, value):
        text = _DATE.sub(lambda m: self.labels.get(_day(m.group()), "step XXX"), str(value))
        # Fiscal/year-only references have no unique trading-step correspondence.
        return _YEAR.sub("step XXX", text)

    def value(self, value):
        if isinstance(value, (date, datetime)):
            return self.labels.get(_day(value), "step XXX")
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            # Lists retain every observation even when several unmapped dates mask
            # to the same placeholder; never collapse date-keyed history entries.
            if value and all(_day(k) is not None for k in value):
                return [{"step": self.labels.get(_day(k), "step XXX"),
                         "value": self.value(v)} for k, v in value.items()]
            return {(re.sub(r"(^|_)date$", r"\1step", str(k))): self.value(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(v) for v in value]
        return value

    def prepare(self, system_prompt, payload):
        system, body = self.text(system_prompt), self.value(payload)
        if self.current_step is not None:
            body = {**body, "current_step": self.current_step}
            if TIME_INSTRUCTION not in system:
                system += "\n\n" + TIME_INSTRUCTION
        return system, body
