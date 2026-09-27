"""Validate seed batches, write CSV/JSON comparisons, and render representative runs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from experiments.main_artifacts import write_json
from experiments.plot_main_results import generate_figures, generate_sizing_comparisons, load_run

METRICS = ("final_value", "cumulative_return", "annualized_return", "sharpe_ratio",
           "max_drawdown", "annualized_volatility", "pair_switches",
           "counterfactual_accuracy", "counterfactual_regret", "decision_failures",
           "llm_failed_calls_total")
PROTOCOL_KEYS = ("test_dates", "valuation_dates", "settlement_date", "train_dates", "dataset_hash", "input_hash", "pool_hash", "config", "git_rev", "code_hash")


def select_representatives(runs, mode, expected_seeds):
    """Rank whole runs, never splice time series or independently optimize metrics."""
    seeds = [r["manifest"]["seed"] for r in runs]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) != set(expected_seeds):
        raise ValueError("Ranking requires all distinct, expected seeds")
    for run in runs:
        value = run["metrics"][mode].get("cumulative_return")
        if value is None or not math.isfinite(value):
            raise ValueError("Cannot rank a run without finite cumulative_return")
    ordered = sorted(runs, key=lambda r: (r["metrics"][mode]["cumulative_return"], r["manifest"]["seed"]))
    best = min(runs, key=lambda r: (-r["metrics"][mode]["cumulative_return"], r["manifest"]["seed"]))
    return {"median": ordered[len(ordered) // 2], "max": best}


def summarize_batch(batch, output_dir=None):
    batch = Path(batch).resolve()
    plan = json.loads((batch / "batch.json").read_text())
    modes, seeds = plan["sizing"], plan["seeds"]
    expected = {(m["provider"], m["model"], s) for m in plan["models"] for s in seeds}
    entries = plan["runs"]
    actual = [(r["provider"], r["model"], r["seed"]) for r in entries]
    if not seeds or len(set(seeds)) != len(seeds) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Batch does not contain all requested seeds for every model")
    if any(r["status"] != "completed" or r["exit_code"] != 0 for r in entries):
        raise ValueError("Failed or incomplete runs cannot be summarized")
    runs = []
    for entry in entries:
        path = (batch / entry["path"]).resolve()
        if not path.is_relative_to(batch):
            raise ValueError("Run path must stay inside its batch")
        run = load_run(path, modes)
        if any(run["manifest"][key] != entry[key] for key in ("provider", "model", "seed")):
            raise ValueError("Run identity disagrees with batch manifest")
        runs.append(run)
    for key in PROTOCOL_KEYS:
        if any(r["manifest"].get(key) != runs[0]["manifest"].get(key) for r in runs[1:]):
            raise ValueError(f"Cannot compare runs with different {key}")
    for spec in plan["models"]:
        group = [r for r in runs if all(r["manifest"][k] == spec[k] for k in ("provider", "model"))]
        inference = group[0]["manifest"].get("inference")
        if any(r["manifest"].get("inference") != inference for r in group[1:]):
            raise ValueError("Cannot mix thinking modes or inference configurations within a model batch")
        if spec["model"] in plan.get("inference", {}) and plan["inference"][spec["model"]] != inference:
            raise ValueError("Seed inference configuration disagrees with its batch")
    output = Path(output_dir).resolve() if output_dir else batch / "summary"
    output.mkdir(parents=True, exist_ok=False)
    all_rows = [r["metrics"][mode] for r in runs for mode in modes]
    pd.DataFrame(all_rows).to_csv(output / "all_seeds.csv", index=False)
    write_json(output / "all_seeds.json", all_rows)
    statistics, selected_rows = [], []
    selections = {kind: {mode: [] for mode in modes} for kind in ("median", "max")}
    for spec in plan["models"]:
        group = [r for r in runs if all(r["manifest"][k] == spec[k] for k in ("provider", "model"))]
        for mode in modes:
            representatives = select_representatives(group, mode, seeds)
            for kind, run in representatives.items():
                selections[kind][mode].append(run)
                selected_rows.append({"selection": kind, "rank_metric": "cumulative_return",
                    "source": str(run["path"].relative_to(batch)), **run["metrics"][mode]})
            for metric in METRICS:
                values = pd.Series([r["metrics"][mode].get(metric) for r in group], dtype=float)
                statistics.append({**spec, "sizing": mode, "metric": metric, "n_seeds": len(seeds),
                    "thinking_mode": group[0]["manifest"].get("inference", {}).get("thinking_mode", "not_applicable"),
                    "n_valid": int(values.notna().sum()),
                    "median": values.median() if values.notna().any() else None,
                    "max": values.max() if values.notna().any() else None,
                    "min": values.min() if values.notna().any() else None,
                    "std": values.std(ddof=1) if values.notna().sum() > 1 else None,
                    "seeds_with_fallbacks": sum(r["metrics"][mode].get("quality") == "contains_fallbacks" for r in group)})
    pd.DataFrame(statistics).to_csv(output / "comparison.csv", index=False)
    write_json(output / "comparison.json", statistics)
    pd.DataFrame(selected_rows).to_csv(output / "selected_seeds.csv", index=False)
    write_json(output / "selection.json", {"metric": "cumulative_return", "n_seeds": len(seeds),
        "scope": "per model and sizing", "records": selected_rows,
        "interpretation": "Representative complete seed trajectories; max is a descriptive best-observed test run, not an unbiased performance estimate. Numeric maxima in comparison.csv do not mean every metric is better when larger."})
    for kind, by_mode in selections.items():
        for mode, selected in by_mode.items():
            generate_figures(selected, [mode], output / kind / mode)
        if set(modes) == {"nobeta", "beta"}:
            generate_sizing_comparisons(by_mode, output / kind)
        from experiments.plot_architecture import generate_architecture
        generate_architecture(output / kind)
    print(pd.DataFrame(selected_rows)[["selection", "model", "sizing", "seed", "cumulative_return", "sharpe_ratio", "quality"]].to_string(index=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory when regenerating figures")
    args = parser.parse_args(argv)
    summarize_batch(args.batch, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
