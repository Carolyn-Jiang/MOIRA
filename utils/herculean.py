"""Causal loader for TheFinAI/Herculean prices and news highlights."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from utils.data_utils import align_news_to_trading_days


DATASET_ID = "TheFinAI/Herculean"
DATASET_REVISION = "refs/convert/parquet"
HERCULEAN_SYMBOLS = ("AAPL", "ADBE", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
PARQUET_URLS = {
    "prices": (
        "https://huggingface.co/datasets/TheFinAI/Herculean/resolve/"
        "refs%2Fconvert%2Fparquet/prices/train/0000.parquet"
    ),
    "news": (
        "https://huggingface.co/datasets/TheFinAI/Herculean/resolve/"
        "refs%2Fconvert%2Fparquet/news/train/0000.parquet"
    ),
}


def _download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def download_herculean(cache_dir: Path, force: bool = False) -> Dict[str, Path]:
    """Download the two immutable experiment inputs into a local cache."""
    return {
        name: _download(url, cache_dir / f"{name}.parquet", force=force)
        for name, url in PARQUET_URLS.items()
    }


def _as_date(value) -> date:
    return pd.Timestamp(value).date()


def build_herculean_env(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    symbols: Sequence[str] = HERCULEAN_SYMBOLS,
) -> Dict[date, Dict]:
    """Convert Herculean tables into the engine's causal session representation.

    The action at session D sees highlights published in [previous session, D), never
    same-session or future highlights. The dataset's ``adj_close`` field is used.
    """
    symbols = tuple(str(s).upper() for s in symbols)
    required_prices = {"symbol", "date", "adj_close"}
    required_news = {"symbol", "date", "highlights"}
    if missing := required_prices - set(prices.columns):
        raise ValueError(f"Herculean prices missing columns: {sorted(missing)}")
    if missing := required_news - set(news.columns):
        raise ValueError(f"Herculean news missing columns: {sorted(missing)}")

    px = prices.loc[prices["symbol"].isin(symbols), ["symbol", "date", "adj_close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.date
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px = px.dropna(subset=["adj_close"])
    if px.duplicated(["date", "symbol"]).any():
        raise ValueError("Herculean prices contain duplicate date/symbol rows")
    wide = px.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    missing_symbols = set(symbols) - set(wide.columns)
    if missing_symbols:
        raise ValueError(f"Herculean prices missing symbols: {sorted(missing_symbols)}")
    wide = wide.loc[:, list(symbols)].dropna(how="any")
    trading_dates = list(wide.index)
    if len(trading_dates) < 29:
        raise ValueError("Herculean must contain at least 29 complete price sessions")

    raw_news: Dict[date, Dict[str, List[str]]] = {}
    for row in news.loc[news["symbol"].isin(symbols)].itertuples(index=False):
        text = str(getattr(row, "highlights", "") or "").strip()
        if text:
            raw_news.setdefault(_as_date(row.date), {}).setdefault(
                str(row.symbol).upper(), []
            ).append(text)
    aligned = align_news_to_trading_days(trading_dates, raw_news)

    return {
        d: {
            "price": {s: float(wide.at[d, s]) for s in symbols},
            "news": aligned.get(d, {}),
        }
        for d in trading_dates
    }


def dataset_fingerprint(paths: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode("utf-8"))
        with paths[name].open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_herculean_dataset(
    cache_dir: Path,
    symbols: Sequence[str] = HERCULEAN_SYMBOLS,
    force_download: bool = False,
) -> Tuple[Dict[date, Dict], Dict[str, object]]:
    paths = download_herculean(cache_dir, force=force_download)
    prices = pd.read_parquet(paths["prices"])
    news = pd.read_parquet(paths["news"])
    env = build_herculean_env(prices, news, symbols)
    dates = sorted(env)
    metadata = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "license": "CC-BY-4.0",
        "sha256": dataset_fingerprint(paths),
        "symbols": list(symbols),
        "sessions": len(dates),
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "price_field": "adj_close",
        "news_policy": "prior-session aligned highlights; never written to agent memory",
    }
    return env, metadata
