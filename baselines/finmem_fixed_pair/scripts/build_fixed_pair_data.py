"""Build a FinMem-compatible synthetic asset for a fixed equity pair.

The adapter changes only the input representation. FinMem still observes one
symbol and retains its original memory, retrieval, reflection, and portfolio
logic. The synthetic price compounds the equal-share LONG_SHORT return, while
the original leg prices are retained for exact pair-level evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_prices(path: Path, symbol: str) -> Dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {row["date"]: float(row[symbol]) for row in rows if row.get(symbol)}


def load_news(path: Path) -> Dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tagged_news(records: Dict[str, dict], date: str, symbol: str) -> List[str]:
    item = records.get(date) or {}
    news = item.get("news") or []
    return [f"[{symbol}] {text}" for text in news if text]


def build_records(
    prices_a: Dict[str, float],
    prices_b: Dict[str, float],
    news_a: Dict[str, dict],
    news_b: Dict[str, dict],
    symbol_a: str,
    symbol_b: str,
    start_date: str,
    end_date: str,
) -> Dict[str, dict]:
    dates = sorted(
        date
        for date in prices_a.keys() & prices_b.keys()
        if start_date <= date <= end_date
    )
    if len(dates) < 2:
        raise ValueError("The selected date range must contain at least two shared sessions")

    all_dates = sorted(prices_a.keys() & prices_b.keys())
    index_value = 100.0
    output: Dict[str, dict] = {}
    previous = None
    for date in dates:
        pa, pb = prices_a[date], prices_b[date]
        if previous is not None:
            prev_a, prev_b = previous
            pair_return = ((pa - prev_a) - (pb - prev_b)) / (prev_a + prev_b)
            index_value *= 1.0 + pair_return
        position = all_dates.index(date)
        previous_day = all_dates[position - 1] if position else date
        # Raw news is available only from preceding sessions, as in the main loader.
        prior_news = []
        for source, symbol in ((news_a, symbol_a), (news_b, symbol_b)):
            for news_day in sorted(source):
                if previous_day <= news_day < date:
                    prior_news.extend(tagged_news(source, news_day, symbol))
        output[date] = {
            "prices": index_value,
            "news": prior_news,
            "pair": {
                "leg_a": symbol_a,
                "leg_b": symbol_b,
                "price_a": pa,
                "price_b": pb,
            },
        }
        previous = (pa, pb)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-a", type=Path, required=True)
    parser.add_argument("--price-b", type=Path, required=True)
    parser.add_argument("--news-a", type=Path, required=True)
    parser.add_argument("--news-b", type=Path, required=True)
    parser.add_argument("--symbol-a", default="BMRN")
    parser.add_argument("--symbol-b", default="META")
    parser.add_argument("--start-date", default="2024-11-01")
    # June 2 is the next session used to settle the May 30 decision.
    parser.add_argument("--end-date", default="2025-06-02")
    parser.add_argument("--output", type=Path, default=Path("data/bmrn_meta_fixed_pair.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records(
        load_prices(args.price_a, args.symbol_a),
        load_prices(args.price_b, args.symbol_b),
        load_news(args.news_a),
        load_news(args.news_b),
        args.symbol_a,
        args.symbol_b,
        args.start_date,
        args.end_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} sessions to {args.output}")


if __name__ == "__main__":
    main()
