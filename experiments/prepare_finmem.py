"""Prepare fixed-pair FinMem inputs from the same frozen panel as MOIRA."""
import argparse
import copy
import json
from pathlib import Path
from experiments.baseline_protocol import DATASETS, prepare_study, study_metadata
from experiments.main_artifacts import write_json
from experiments.recovery import batch_lock
from experiments.run_ablations import PROJECT_ROOT


def pair_records(spec, pair):
    """Retain leg prices; compound the shared engine's No-Beta LS return."""
    days = [*spec["train_dates"], *spec["test_dates"], spec["settlement_date"]]
    records, previous, index = {}, None, 100.0
    a, b = pair
    for day in days:
        row = spec["env_data"][day]
        pa, pb = float(row["price"][a]), float(row["price"][b])
        if previous is not None:
            qa, qb = previous
            index *= 1 + ((pa - qa) - (pb - qb)) / (qa + qb)
        # News is already aligned by the shared loader; do not shift it again.
        news = [] if day == spec["settlement_date"] else [
            f"[{symbol}] {text}" for symbol in pair for text in row.get("news", {}).get(symbol, [])]
        records[day.isoformat()] = {"prices": index, "news": news,
            "pair": {"leg_a": a, "leg_b": b, "price_a": pa, "price_b": pb}}
        previous = pa, pb
    return records


def make_config(spec, pair, output, seed):
    template = PROJECT_ROOT / "baselines/finmem_fixed_pair/configs/main_fixed_pair.json"
    config = copy.deepcopy(json.loads(template.read_text()))
    symbol, label = "__".join(pair), "/".join(pair)
    output = Path(output).resolve()
    config["chat_config"]["chat_parameters"]["seed"] = seed
    config["chat_config"]["chat_system_message"] = (
        f"You are a financial decision agent evaluating the fixed equity pair {label}. "
        "Follow the action mapping in the user prompt exactly and return only valid JSON.")
    config["env_config"].update(trading_symbols=[symbol],
        warmup_start_time=spec["train_dates"][0].isoformat(),
        warmup_end_time=spec["train_dates"][-1].isoformat(),
        test_start_time=spec["test_dates"][0].isoformat(),
        test_end_time=spec["settlement_date"].isoformat(),
        env_data_path={symbol: str(output / "pair_data.json")})
    config["portfolio_config"]["trading_symbols"] = [symbol]
    agent = config["agent_config"]
    agent.update(agent_name="finmem_fixed_pair_agent", trading_symbols=[symbol],
        character_string={symbol: f"You manage the fixed equity pair {label}. "
            f"{pair[0]} is always the first leg and {pair[1]} is always the second leg. "
            "Your task is relative-value trading. The pair cannot be replaced or reordered."})
    agent["memory_db_config"].update(trading_symbols=[symbol],
        memory_db_endpoint=f"local:{output / 'qdrant'}")
    meta = config["meta_config"]
    meta.update(run_name=f"finmem_{symbol}_seed{seed}",
        decision_end_time=spec["test_dates"][-1].isoformat())
    for key, folder in {"warmup_checkpoint_save_path": "warmup_checkpoint",
        "warmup_output_save_path": "warmup_output", "test_checkpoint_save_path": "test_checkpoint",
        "test_output_save_path": "test_output", "result_save_path": "final_result",
        "log_save_path": "logs"}.items():
        meta[key] = str(output / folder)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=DATASETS, default="main")
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/finmem-main"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(args.output_dir):
        spec = prepare_study(args.study, args.output_dir)
        # Preserve the paper adapter's main fixed pair; A1/A2 use matched study pairs.
        pair = ("BMRN", "META") if args.study == "main" else spec["pair"]
        config = make_config(spec, pair, args.output_dir, args.seed)
        path = args.output_dir / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("FinMem configuration changed; use a new output directory")
        write_json(path, config)
        write_json(args.output_dir / "pair_data.json", pair_records(spec, pair))
        write_json(args.output_dir / "protocol.json", {**study_metadata(spec), "pair": list(pair),
            "model": config["chat_config"]["chat_model"], "seed": args.seed,
            "exposure_evaluation": "same fixed-pair actions replayed under both exposure rules"})
        print(f"FINMEM_INPUTS_OK study={args.study} test_steps={len(spec['test_dates'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
