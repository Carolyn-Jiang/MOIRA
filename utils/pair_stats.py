"""Lightweight news-sentiment helper.

NOTE: The previous low-correlation / multi-metric pair-selection logic that used
to live here has been REMOVED. Classical statistical-arbitrage pair construction
(cointegration, ADF, half-life, stability, opportunity, composite scoring) now
lives in ``utils/statarb.py``. This module retains only the shared lexical
sentiment helper used by the trading engine.
"""

from __future__ import annotations

from typing import Sequence


def _sentiment_label(news_items: Sequence[str]) -> str:
    """Crude lexical sentiment used only to summarise news for prompts/labels."""
    if not news_items:
        return "Neutral / No recent news"
    text = " ".join(news_items).lower()
    pos = sum(text.count(w) for w in ("beat", "bullish", "surge", "growth", "upgrade",
                                      "record", "gain", "strong", "rally", "positive"))
    neg = sum(text.count(w) for w in ("miss", "bearish", "plunge", "downgrade", "lawsuit",
                                      "decline", "weak", "fall", "cut", "negative", "drop"))
    if pos > neg * 1.2:
        return "Positive"
    if neg > pos * 1.2:
        return "Negative"
    return "Mixed / Neutral"
