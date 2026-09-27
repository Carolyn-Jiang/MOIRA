"""A1/A2 window controls and paired A3/A4 interventions on the shared MOIRA engine."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from experiments.main_artifacts import result_metrics, write_json
from experiments.recovery import batch_lock, digest
from experiments.run_redesign import (SYMBOLS, build_agent, split_train_test, _code_hash,
    run_discovery_training, run_discovery_test, _assign_realized_pnl, episode_status)
from llm.backbone import infer_family, build_llm
from llm.errors import TransientInferenceError
from llm.policy_agent import (DirectionMemory, DirectionPromptMemory, PairSelectionMemory,
                              load_direction_discovery_prompt)
from utils.data_utils import (load_offline_dataset, load_pair_selection_price_history,
                             validate_pair_selection_history, apply_cached_prices, next_trading_day)
from utils.engine import write_counterfactual_logs, write_discovery_logs
from utils.herculean import HERCULEAN_SYMBOLS, load_herculean_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1103, 2207, 3301, 4409, 5501)
TRAIN_SESSIONS = 28  # A1/A2 retain their pre-specified windows; main A3/A4 use 27.
CROSS_REGIME_START = "2025-06-02"
CROSS_REGIME_SESSIONS = 103

@dataclass(frozen=True)
class Setting:
    key: str
    model: str
    beta_hedged: bool

SETTINGS = {"gpt41_beta": Setting("gpt41_beta", "gpt-4.1-mini", True),
            "gpt54_nobeta": Setting("gpt54_nobeta", "gpt-5.4-mini", False)}
ABLATIONS = {"a1": "long_horizon_full", "a2": "cross_regime_full",
             "a3": "warmup_only", "a4": "no_reselection"}
ABLATION_DATASETS = {"a1": "herculean", "a2": "herculean_regime", "a3": "main", "a4": "main"}


def _split_dates(env_data):
    dates = sorted(env_data)
    if len(dates) <= TRAIN_SESSIONS:
        raise ValueError("Insufficient sessions for the A1/A2 training window")
    return dates[:TRAIN_SESSIONS], dates[TRAIN_SESSIONS:]


def _cross_regime_dates(env_data):
    selected = [d for d in sorted(env_data) if d >= date.fromisoformat(CROSS_REGIME_START)][:CROSS_REGIME_SESSIONS]
    if len(selected) != CROSS_REGIME_SESSIONS:
        raise ValueError("A2 needs 103 sessions from its pre-specified start")
    return selected


def _prepare_dataset(name, *, cache_dir, force_download=False):
    if name == "main":
        env = load_offline_dataset(SYMBOLS, news_dir=str(PROJECT_ROOT / "data/news"))
        symbols = list(SYMBOLS)
        train, test = split_train_test(sorted(env))
        meta = {"dataset": "bundled_primary_news", "sessions": len(env)}
        expected = (27, 75)
    else:
        env, meta = load_herculean_dataset(cache_dir, force_download=force_download)
        symbols = list(HERCULEAN_SYMBOLS)
        if name == "herculean_regime":
            chosen = _cross_regime_dates(env)
            env = {d: env[d] for d in chosen}
        elif name != "herculean":
            raise ValueError("Unknown dataset")
        if name == "herculean":
            env = {d: row for d, row in env.items() if date(2024, 12, 2) <= d <= date(2026, 3, 31)}
        train, test = _split_dates(env)
        expected = (28, 75 if name == "herculean_regime" else 304)
    if (len(train), len(test)) != expected:
        raise ValueError(f"Unexpected {name} windows: {(len(train), len(test))}; expected {expected}")
    terminal = next_trading_day(test[-1])
    pool, source = load_pair_selection_price_history(
        symbols, decision_dates=[*train, *test, terminal], cache_dir=str(PROJECT_ROOT / "data/cache"))
    env = apply_cached_prices(env, pool, symbols, terminal)
    meta.update(price_source=source, pair_history_years=10,
                pair_history_window="[selection_date - 10 calendar years, selection_date)")
    return dict(env_data=env, dataset_meta=meta, symbols=symbols, train_dates=train,
                test_dates=test, settlement_date=terminal, pool_df=pool, profile_cache={})


def _snapshot(name, directory, cache_dir):
    """Resume from the same hashed inputs, never refresh a partly completed study."""
    folder = directory / "inputs" / name
    marker = folder / "snapshot.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if any(digest(folder / f) != h for f, h in saved["hashes"].items()):
            raise ValueError("A saved study input has changed")
        env = {date.fromisoformat(d): row for d, row in json.loads((folder / "env.json").read_text()).items()}
        pool = pd.read_csv(folder / "pool.csv", parse_dates=["date"]).set_index("date")
        spec = {k: saved[k] for k in ("dataset_meta", "symbols")}
        spec.update(env_data=env, pool_df=pool, profile_cache={},
                    train_dates=[date.fromisoformat(d) for d in saved["train_dates"]],
                    test_dates=[date.fromisoformat(d) for d in saved["test_dates"]],
                    settlement_date=date.fromisoformat(saved["settlement_date"]))
        validate_pair_selection_history(pool, spec["symbols"],
            [*spec["train_dates"], *spec["test_dates"], spec["settlement_date"]])
        return spec
    if folder.exists():
        raise ValueError("Incomplete input snapshot: use a new output directory")
    spec = _prepare_dataset(name, cache_dir=cache_dir)
    folder.mkdir(parents=True)
    write_json(folder / "env.json", {d.isoformat(): v for d, v in spec["env_data"].items()})
    spec["pool_df"].to_csv(folder / "pool.csv", index_label="date")
    saved = {k: spec[k] for k in ("dataset_meta", "symbols", "train_dates", "test_dates", "settlement_date")}
    saved["hashes"] = {f: digest(folder / f) for f in ("env.json", "pool.csv")}
    write_json(marker, saved)
    return spec


def _agent(setting, seed, symbols, directory=None):
    random.seed(seed)
    np.random.seed(seed)
    return build_agent(infer_family(setting.model), setting.model, symbols,
                       log_dir=str(directory / "llm_calls") if directory else None, seed=seed)


def _warmup(agent, spec):
    """One shared training trajectory; keep sanitized reflection without a replay gate."""
    pm = PairSelectionMemory()
    prompt = DirectionPromptMemory(load_direction_discovery_prompt(), symbols=spec["symbols"])
    dm = DirectionMemory()
    result = run_discovery_test(test_dates=spec["train_dates"], env_data=spec["env_data"],
        agent=agent, pool_df=spec["pool_df"], symbols=spec["symbols"], frozen_prompt=prompt.active_prompt(),
        pair_memory=pm, direction_memory=dm, lifecycle=True, profile_cache=spec["profile_cache"])
    _assign_realized_pnl(dm, result, 23)
    reflection = agent.reflect_direction(dm) if episode_status(dm) != "NO_EVIDENCE" else {"lessons": []}
    # Warmup reflections are available in both conditions but are not labeled as
    # replay-accepted edits. Future consolidation uses the normal acceptance gate.
    prompt.base_prompt = prompt.candidate_prompt(reflection["lessons"])
    dm.clear()
    return prompt, pm


def _training(setting, seed, ablation, spec, root, max_cycles):
    checkpoint = root / "training.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    _archive_partial(root / "training")
    agent = _agent(setting, seed, spec["symbols"], root / "training")
    if ablation == "a3":
        prompt, pair = _warmup(agent, spec)
    else:
        prompt = DirectionPromptMemory(load_direction_discovery_prompt(), symbols=spec["symbols"])
        pair = PairSelectionMemory()
    warmup_prompt = prompt.active_prompt()
    prompt, pair, cycles = run_discovery_training(None, train_dates=spec["train_dates"],
        env_data=spec["env_data"], agent=agent, pool_df=spec["pool_df"], symbols=spec["symbols"],
        prompt_memory=prompt, pair_memory=pair, max_cycles=max_cycles, beta_hedged=False,
        lifecycle=True, selection_frequency=23, break_check_frequency=23, profile_cache=spec["profile_cache"])
    state = {"full_prompt": prompt.active_prompt(), "warmup_prompt": warmup_prompt,
             "pair_records": pair.records, "cycles": cycles, "usage": agent.llm.usage_summary()}
    write_json(checkpoint, state)
    return state


def _archive_partial(folder):
    if folder.exists():
        archive = folder.parent / "attempts" / f"{folder.name}-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), archive)


def _evaluate(setting, seed, spec, prompt, pair_records, output, *, initial_pair=None, allow_reselection=True):
    if (output / "summary.json").is_file():
        return json.loads((output / "summary.json").read_text())
    _archive_partial(output)
    output.mkdir(parents=True, exist_ok=True)
    agent = _agent(setting, seed, spec["symbols"], output)
    dm = DirectionMemory()
    result = run_discovery_test(test_dates=spec["test_dates"], env_data=spec["env_data"],
        agent=agent, pool_df=spec["pool_df"], symbols=spec["symbols"], frozen_prompt=prompt,
        pair_memory=PairSelectionMemory(records=copy.deepcopy(pair_records)), direction_memory=dm,
        lifecycle=True, selection_frequency=23, break_check_frequency=23,
        beta_hedged=setting.beta_hedged, settlement_date=spec["settlement_date"],
        initial_pair=initial_pair, allow_reselection=allow_reselection,
        profile_cache=spec["profile_cache"])
    metrics = result_metrics(result)
    if (metrics["finalized_transitions"] != len(spec["test_dates"])
            or len(result.normalized_equity) != len(spec["test_dates"]) + 1):
        raise RuntimeError("Unsettled ablation decision")
    _assign_realized_pnl(dm, result, 23)
    write_discovery_logs(result, str(output / "logs"), backend=infer_family(setting.model))
    write_counterfactual_logs(result, str(output / "logs"), "test")
    pd.DataFrame({"date": [*spec["test_dates"], spec["settlement_date"]],
                  "equity": result.normalized_equity}).to_csv(output / "equity.csv", index=False)
    first_pair = result.pair_selections[0].split("/") if result.pair_selections else None
    summary = {"metrics": metrics, "initial_pair": first_pair, "usage": agent.llm.usage_summary(),
               "reselection_enabled": allow_reselection, "lifecycle_monitoring": True,
               "test_steps": len(spec["test_dates"]), "settlement_date": spec["settlement_date"]}
    write_json(output / "summary.json", summary)
    return summary


def run_cell(setting, seed, ablation, spec, root, max_cycles):
    marker = root / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    root.mkdir(parents=True, exist_ok=True)
    state = _training(setting, seed, ablation, spec, root, max_cycles)
    # Save a matched full control for A3/A4. Pair-quality memory is identical at
    # test entry, so A3 changes only the global prompt after shared warmup.
    records = state["pair_records"]
    full_path = root / "full/summary.json"
    full = json.loads(full_path.read_text()) if full_path.exists() else _evaluate(
        setting, seed, spec, state["full_prompt"], records, root / "full")
    output = {"full": full}
    if ablation in ("a3", "a4"):
        if ablation == "a4" and full["initial_pair"] is None:
            raise ValueError("A4 requires a valid shared initial pair")
        output[ABLATIONS[ablation]] = _evaluate(setting, seed, spec,
            state["warmup_prompt"] if ablation == "a3" else state["full_prompt"],
            records, root / ABLATIONS[ablation], initial_pair=full["initial_pair"] if ablation == "a4" else None,
            allow_reselection=ablation != "a4")
    write_json(marker, output)
    return output


def main(argv=None):
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1103])
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=120)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/herculean")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/ablations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if (len(args.seeds) != len(set(args.seeds)) or any(not 0 <= s < 2**32 for s in args.seeds)
            or args.max_cycles < 1 or args.attempts < 1 or not np.isfinite(args.retry_delay) or args.retry_delay < 0):
        parser.error("Invalid seeds, cycles or retry configuration")
    plan = {"settings": [asdict(SETTINGS[k]) for k in args.settings], "ablations": args.ablations,
            "seeds": args.seeds, "max_cycles": args.max_cycles, "selection_frequency": 23,
            "break_check_frequency": 23, "pair_history_years": 10,
            "main_windows": [27, 75], "a1_windows": [28, 304], "a2_windows": [28, 75],
            "terminal_settlement": "next_session_price_only", "training_exposure": "nobeta"}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return 0
    clients = [build_llm(infer_family(s["model"]), s["model"], seed=args.seeds[0]) for s in plan["settings"]]
    plan["inference"] = [c.inference_config() for c in clients]
    plan["code_hash"] = _code_hash()
    plan["requirements_hash"] = digest(PROJECT_ROOT / "requirements.txt")
    plan["news_hashes"] = {p.name: digest(p) for p in sorted((PROJECT_ROOT / "data/news").glob("*.json"))}
    if args.check_only:
        for client in clients:
            client.max_output_tokens = min(1024, client.max_output_tokens)
            client.request_attempts = 1
            client.validate_server()
        print("CHECK_OK ablation model interfaces"); return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(args.output_dir):
        manifest = args.output_dir / "plan.json"
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError("Study configuration changed; use a new output directory")
        write_json(manifest, plan)
        specs = {n: _snapshot(n, args.output_dir, args.cache_dir) for n in set(ABLATION_DATASETS[a] for a in args.ablations)}
        rows = []
        for key in args.settings:
            setting = SETTINGS[key]
            for seed in args.seeds:
                for ablation in args.ablations:
                    root = args.output_dir / key / str(seed) / ablation
                    for attempt in range(args.attempts):
                        try:
                            output = run_cell(setting, seed, ablation, specs[ABLATION_DATASETS[ablation]], root, args.max_cycles)
                            break
                        except TransientInferenceError:
                            if attempt + 1 == args.attempts: raise
                            # Preserve partial test calls before restarting this cell;
                            # completed training/full-control checkpoints are reused.
                            for part in ("full", ABLATIONS[ablation]):
                                folder = root / part
                                if folder.exists() and not (folder / "summary.json").exists():
                                    archive = root / "attempts" / f"{part}-{time.time_ns()}"
                                    archive.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.move(str(folder), archive)
                            print(f"RETRY ablation={ablation} seed={seed} wait_seconds={args.retry_delay:g}", flush=True)
                            time.sleep(args.retry_delay)
                    for condition, result in output.items():
                        rows.append({"model": setting.model, "sizing": "beta" if setting.beta_hedged else "nobeta",
                                     "seed": seed, "ablation": ablation, "condition": condition, **result["metrics"]})
                    pd.DataFrame(rows).to_csv(args.output_dir / "comparison.csv", index=False)
                    print(f"CELL_OK {key} {seed} {ablation}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
