"""Refresh the bundled adjusted closing prices from 2015-01-01 through today."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from experiments.main_artifacts import write_json
from experiments.run_redesign import SYMBOLS
from utils.data_utils import filter_to_trading_days


def refresh_prices(cache_dir, *, as_of=None):
    as_of = as_of or date.today()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for symbol in SYMBOLS:
        raw = yf.download(symbol, start="2015-01-01",
                          end=(as_of + timedelta(days=1)).isoformat(),
                          auto_adjust=True, progress=False, threads=False, timeout=30)
        if raw is None or raw.empty:
            raise RuntimeError(f"No Yahoo Finance prices returned for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw[["Close"]].rename(columns={"Close": symbol})
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame = frame.loc["2015-01-01":as_of.isoformat()].sort_index()
        values = frame[symbol]
        if (frame.index.has_duplicates or not np.isfinite(values).all()
                or (values <= 0).any() or frame.index.min() > pd.Timestamp("2015-01-08")):
            raise ValueError(f"Invalid or incomplete historical prices for {symbol}")
        required = filter_to_trading_days(pd.date_range("2025-01-02", "2025-06-02").date)
        if frame.reindex(pd.to_datetime(required))[symbol].isna().any():
            raise ValueError(f"{symbol} is missing a main-experiment or terminal-settlement price")
        frames[symbol] = frame
        print(f"{symbol}: {len(frame)} rows, {frame.index.min().date()} -> {frame.index.max().date()}", flush=True)
    # Validate all downloads before replacing any existing cache.
    metadata = {"source": "yfinance", "yfinance_version": yf.__version__,
                "requested_start": "2015-01-01", "requested_through": as_of.isoformat(),
                "auto_adjust": True, "field": "Close", "symbols": {}}
    for symbol, frame in frames.items():
        path = cache_dir / f"pool_{symbol}_10y.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index_label="date")
        temporary.replace(path)
        metadata["symbols"][symbol] = {"rows": len(frame),
            "first_date": frame.index.min().date().isoformat(),
            "last_date": frame.index.max().date().isoformat(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(cache_dir / "prices_metadata.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=date.fromisoformat, default=date.today())
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/cache")
    args = parser.parse_args()
    refresh_prices(args.cache_dir, as_of=args.through)
