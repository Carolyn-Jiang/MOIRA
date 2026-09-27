"""Seed-addressed main-experiment artifacts; no model or plotting imports."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


def model_slug(model: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip(".-")[:80] or "model"
    return f"{readable}-{hashlib.sha256(model.encode()).hexdigest()[:8]}"


def run_directory(output_root, provider: str, model: str, seed: int) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", provider):
        raise ValueError("Invalid provider directory name")
    return Path(output_root) / provider / model_slug(model) / f"seed_{seed}"


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def result_metrics(result):
    import numpy as np
    import pandas as pd
    from utils.metrics import compute_performance_metrics

    perf = compute_performance_metrics(pd.Series(result.normalized_equity, dtype=float), 1.0)
    finalized = [d for d in result.decisions if d.get("actual_pnl") is not None]
    actions = ("LONG_SHORT", "SHORT_LONG", "FLAT")
    correct, regret = 0, 0.0
    for row in finalized:
        values = [float(row.get(k) or 0.0) for k in ("cf_long_short", "cf_short_long", "cf_flat")]
        correct += row.get("final_executed_action") == actions[int(np.argmax(values))]
        regret += max(values) - float(row["actual_pnl"])
    return clean_json({**perf, "cumulative_return": perf["final_value"] - 1.0,
                       "pair_switches": len(result.switch_events),
                       "decision_failures": sum(bool(d.get("direction_decision_failed"))
                                                for d in result.decisions),
                       "finalized_transitions": len(finalized),
                       "counterfactual_accuracy": correct / len(finalized) if finalized else None,
                       "counterfactual_regret": regret})
