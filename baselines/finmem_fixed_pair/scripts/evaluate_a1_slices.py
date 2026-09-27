"""Legacy slice analysis of one A1 trajectory, not independent main/A2 runs.

The original settlement boundaries are retained here. Use experiments.prepare_finmem
for fresh independent runs with the current full terminal settlement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.eval_pipeline import evaluate_fixed_pair


WINDOWS = {
    "A1": ("2025-01-14", "2026-03-31"),
    "Main": ("2025-02-12", "2025-06-02"),
    "A2": ("2025-07-14", "2025-10-27"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.data.read_text(encoding="utf-8"))
    portfolio = json.loads(
        (args.checkpoint / "portfolio_checkpoint.json").read_text(encoding="utf-8")
    )
    actions = pd.DataFrame(
        {
            "date": portfolio["trading_dates"],
            "direction": portfolio["trading_position"],
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for window, (start, end) in WINDOWS.items():
        daily, metrics = evaluate_fixed_pair(records, actions, start, end)
        daily.insert(0, "window", window)
        metrics.insert(0, "window", window)
        daily.to_csv(args.output / f"{window.lower()}_daily.csv", index=False)
        summary.append(metrics)
    result = pd.concat(summary, ignore_index=True)
    result.to_csv(args.output / "window_metrics.csv", index=False)
    actions.to_csv(args.output / "a1_action_trajectory.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
