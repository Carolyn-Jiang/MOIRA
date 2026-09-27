"""Build a FinMem fixed-pair input directly from TheFinAI/Herculean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--symbol-a", default="AMZN")
    parser.add_argument("--symbol-b", default="MSFT")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = (args.symbol_a.upper(), args.symbol_b.upper())
    prices = pd.read_parquet(args.data_dir / "prices.parquet")
    news = pd.read_parquet(args.data_dir / "news.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    news["date"] = pd.to_datetime(news["date"]).dt.date
    wide = (
        prices[prices["symbol"].isin(symbols)]
        .pivot(index="date", columns="symbol", values="adj_close")
        .dropna()
        .sort_index()
    )
    raw_news: dict = {}
    for row in news[news["symbol"].isin(symbols)].itertuples(index=False):
        text = str(row.highlights or "").strip()
        if text:
            raw_news.setdefault(row.date, {}).setdefault(row.symbol, []).append(text)

    dates = list(wide.index)
    aligned: dict = {}
    for index, day in enumerate(dates):
        previous = dates[index - 1] if index else None
        aligned[day] = {
            symbol: [
                text
                for news_day, by_symbol in raw_news.items()
                if (previous is None or news_day >= previous) and news_day < day
                for text in by_symbol.get(symbol, [])
            ]
            for symbol in symbols
        }

    index_value = 100.0
    records = {}
    previous_prices = None
    for day in dates:
        pa, pb = (float(wide.at[day, symbol]) for symbol in symbols)
        if previous_prices is not None:
            prev_a, prev_b = previous_prices
            index_value *= 1.0 + ((pa - prev_a) - (pb - prev_b)) / (prev_a + prev_b)
        records[day.isoformat()] = {
            "prices": index_value,
            "news": [
                f"[{symbol}] {text}"
                for symbol in symbols
                for text in aligned[day][symbol]
            ],
            "pair": {
                "leg_a": symbols[0],
                "leg_b": symbols[1],
                "price_a": pa,
                "price_b": pb,
            },
        }
        previous_prices = (pa, pb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} Herculean sessions to {args.output}")


if __name__ == "__main__":
    main()
