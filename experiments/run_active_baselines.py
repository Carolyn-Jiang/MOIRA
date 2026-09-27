"""Run EG+Z or EG-DDQN; keep all new output separate from the paper snapshot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from experiments.baseline_protocol import DATASETS, prepare_study, study_metadata
from experiments.baselines import simulate_engle_granger_zscore
from experiments.eg_ddqn import DDQNConfig, EGDoubleDQN
from experiments.main_artifacts import write_json
from experiments.recovery import batch_lock
from experiments.run_redesign import _code_hash
from utils.metrics import compute_performance_metrics


def main(kind, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", choices=DATASETS, default=["main"])
    parser.add_argument("--sizing", choices=["nobeta", "beta", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / kind)
    if kind == "eg-ddqn":
        parser.add_argument("--seeds", nargs="+", type=int, default=[1103, 2207, 3301])
        parser.add_argument("--episodes", type=int, default=600)
    else:
        parser.add_argument("--entry-z", type=float, default=1.0)
        parser.add_argument("--exit-band", type=float, default=0.25)
    args = parser.parse_args(argv)
    if kind == "eg-ddqn" and (args.episodes < 1 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use positive episodes and distinct seeds")
    if kind == "eg-z" and not 0 <= args.exit_band < args.entry_z:
        parser.error("Require 0 <= exit-band < entry-z")
    plan = {k: v for k, v in vars(args).items() if k != "output_dir"}
    plan.update(baseline=kind, code_hash=_code_hash())
    if kind == "eg-ddqn":
        plan["ddqn"] = asdict(DDQNConfig(episodes=args.episodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with batch_lock(args.output_dir):
        marker = args.output_dir / "plan.json"
        if marker.exists() and json.loads(marker.read_text()) != plan:
            raise ValueError("Configuration changed; choose a new output directory")
        write_json(marker, plan)
        rows = []
        modes = ["nobeta", "beta"] if args.sizing == "both" else [args.sizing]
        for study in args.studies:
            folder = args.output_dir / study
            spec = prepare_study(study, folder)
            dates, train = spec["test_dates"], spec["train_dates"]
            env, pair = spec["env_data"], spec["pair"]
            write_json(folder / "protocol.json", study_metadata(spec))
            history = {d: env[d] for d in train}
            curves = {"date": [*dates, spec["settlement_date"]]}
            for mode in modes:
                for seed in (args.seeds if kind == "eg-ddqn" else [None]):
                    name = f"{mode}_seed_{seed}" if seed is not None else mode
                    checkpoint = folder / f"{name}.json"
                    if checkpoint.exists():
                        result = json.loads(checkpoint.read_text())
                    else:
                        if kind == "eg-ddqn":
                            agent = EGDoubleDQN(seed, DDQNConfig(episodes=args.episodes))
                            training = agent.fit(train, env, pair, beta_hedged=mode == "beta")
                            result = agent.evaluate(dates, env, spec["symbols"], pair,
                                beta_hedged=mode == "beta", history_seed=history,
                                settlement_date=spec["settlement_date"])
                            result["training"] = training
                        else:
                            result = simulate_engle_granger_zscore(dates=dates, env_data=env,
                                symbols=spec["symbols"], pair=pair, beta_hedged=mode == "beta",
                                history_seed=history, settlement_date=spec["settlement_date"],
                                entry_z=args.entry_z, exit_band=args.exit_band)
                        write_json(checkpoint, result)
                    equity = result["normalized_equity"]
                    if len(equity) != len(dates) + 1 or len(result["actions"]) != len(dates):
                        raise RuntimeError("Incomplete baseline settlement")
                    metrics = compute_performance_metrics(pd.Series(equity), 1.0)
                    rows.append({"study": study, "baseline": kind, "sizing": mode,
                        "seed": seed, "pair": "/".join(pair), "CR": equity[-1] - 1,
                        "Sharpe": metrics["sharpe_ratio"], "MDD": metrics["max_drawdown"]})
                    curves[name] = equity
            pd.DataFrame(curves).to_csv(folder / "equity.csv", index=False)
            print(f"BASELINE_OK baseline={kind} study={study} test_steps={len(dates)}")
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / "comparison.csv", index=False)
        if kind == "eg-ddqn":
            frame.groupby(["study", "sizing"])[["CR", "Sharpe", "MDD"]].median().to_csv(
                args.output_dir / "metric_medians.csv")
    return 0
