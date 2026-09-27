"""Run open-weight or hosted models with a shared protocol, then tables and median/max figures.

Example: python -m experiments.run_main Qwen/Qwen3-4B --thinking on
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from dotenv import load_dotenv
from experiments.main_artifacts import model_slug, run_directory, write_json
from llm.backbone import build_llm, infer_family
from llm.open_source_llm import is_transient_service_error
from experiments.recovery import (batch_lock, input_signature, digest, completed_seed,
                                  archive_attempt, finish_summary)
from utils.data_utils import PAIR_HISTORY_YEARS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1103, 2207, 3301)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="One or more exact served model names")
    parser.add_argument("--models", "--model", dest="named_models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--sizing", choices=["nobeta", "beta", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, help="New batch directory; existing directories are refused")
    parser.add_argument("--max-cycles", type=int, default=int(os.getenv("LLM_MAX_CYCLES", "8")))
    parser.add_argument("--base-url", help="Shared Chat Completions endpoint; overrides family URLs for this run")
    parser.add_argument("--thinking", choices=["on", "off"],
                        help="One Qwen3 thinking mode per batch; use separate output directories")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without inference or file writes")
    parser.add_argument("--check-only", action="store_true",
                        help="Check data and one response per model (at most 1024 output tokens); no seeds")
    parser.add_argument("--seed-attempts", type=int, default=3,
                        help="Maximum attempts for a transiently failed seed per invocation")
    parser.add_argument("--retry-delay", type=float, default=120,
                        help="Seconds between seed attempts (default: 120)")
    args = parser.parse_args(argv)
    if args.check_only and args.dry_run:
        parser.error("Choose --check-only OR --dry-run")
    import math
    if args.seed_attempts < 1 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("Seed attempts must be positive and retry delay nonnegative")
    if args.models and args.named_models:
        parser.error("Use positional models OR --models, not both")
    args.models = [part.strip() for name in (args.models or args.named_models or [])
                   for part in name.split(",") if part.strip()]
    if not args.models or len(set(args.models)) != len(args.models):
        parser.error("Supply at least one model, without duplicates")
    try:
        args.families = [infer_family(m) for m in args.models]
    except ValueError as exc:
        parser.error(str(exc))
    if args.thinking and any(f != "qwen" for f in args.families):
        parser.error("--thinking batches must contain only Qwen3 models")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Supply distinct seeds in [0, 2**32)")
    if args.max_cycles < 1:
        parser.error("--max-cycles must be at least 1")
    args.output_dir = (args.output_dir or ROOT / "runs" /
                       (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])).resolve()
    return args


def run_seed_process(command, env, console_path):
    """Stream complete stdout/stderr to both terminal and a per-seed log."""
    console_path.parent.mkdir(parents=True, exist_ok=True)
    with console_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1,
                              pass_fds=((int(env["MAIN_BATCH_LOCK_FD"]),)
                                        if "MAIN_BATCH_LOCK_FD" in env else ())) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def main(argv=None):
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args(argv)
    modes = ["nobeta", "beta"] if args.sizing == "both" else [args.sizing]
    env = os.environ.copy()
    env.update(LLM_MAX_CYCLES=str(args.max_cycles), PYTHONUNBUFFERED="1")
    if args.thinking:
        env["OPEN_SOURCE_THINKING"] = args.thinking
    if args.base_url:
        for family in set(args.families):
            env[f"{family.upper()}_BASE_URL"] = args.base_url
    requested = {"schema_version": 2, "status": "planned", "models": [
        {"provider": family, "model": model} for family, model in zip(args.families, args.models)],
        "seeds": args.seeds, "sizing": modes, "max_cycles": args.max_cycles,
        "check_only": args.check_only,
        "thinking_mode_requested": args.thinking or os.getenv("OPEN_SOURCE_THINKING"),
        "pair_history_years": PAIR_HISTORY_YEARS,
        "selection_metric": "cumulative_return", "selection_scope": "per_model_per_sizing",
        "median_rule": "upper-middle seed after sorting by (cumulative_return, seed)",
        "max_rule": "highest cumulative_return; smallest seed breaks ties", "runs": []}
    if args.dry_run:
        print(json.dumps({**requested, "output_dir": str(args.output_dir)}, indent=2))
        return 0
    batch = args.output_dir
    existed = batch.exists()
    batch.mkdir(parents=True, exist_ok=True)
    with batch_lock(batch) as lock_fd:
        if existed and not (batch / "batch.json").is_file():
            raise FileExistsError("Existing directory is not a resumable batch; choose a new output directory")
        requested["input_signature"] = input_signature(ROOT, env)
        plan = json.loads((batch / "batch.json").read_text()) if existed else requested
        if existed:
            for key in ("schema_version", "models", "seeds", "sizing", "max_cycles", "check_only", "input_signature"):
                if plan.get(key) != requested[key]:
                    raise ValueError(f"Cannot resume with different {key}; use a new output directory")
            print(f"RESUME batch={batch}", flush=True)
        else:
            from experiments.run_redesign import _git_rev
            plan["code_revision"] = _git_rev()
            write_json(batch / "batch.json", plan)
        environment = {"python": sys.version, "packages": {d.metadata["Name"]: d.version
                       for d in importlib.metadata.distributions() if d.metadata.get("Name")}}
        environment_path = batch / "environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
            raise ValueError("Client environment changed; restore it or use a new batch")
        write_json(environment_path, environment)
        env["MAIN_BATCH_LOCK_FD"] = str(lock_fd)
        env["MAIN_CODE_REVISION"] = plan["code_revision"]
        try:
            return execute_batch(args, env, plan, batch)
        except BaseException as exc:
            plan.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__)
            write_json(batch / "batch.json", plan)
            print(f"Batch stopped ({type(exc).__name__}); saved logs: {batch}", file=sys.stderr)
            raise


def execute_batch(args, env, plan, batch):
    # Build configuration without inference, including on completed-batch restarts.
    clients = {}
    for family, model in zip(args.families, args.models):
        previous = os.environ.get(f"{family.upper()}_BASE_URL")
        try:
            if args.base_url:
                os.environ[f"{family.upper()}_BASE_URL"] = args.base_url
            client = build_llm(family, model=model, seed=args.seeds[0], thinking_mode=args.thinking,
                               log_dir=str(batch / "preflight" / family / model_slug(model)))
            if args.check_only:
                client.max_output_tokens = min(client.max_output_tokens, 1024)
                client.request_attempts = 1
                client._get_client().max_retries = 0
            config = client.inference_config()
            previous_config = plan.setdefault("inference", {}).get(model)
            if previous_config is not None and config != previous_config:
                raise ValueError("Inference configuration changed; use a new output directory")
            plan["inference"][model] = config
            clients[model] = client
        finally:
            if previous is None:
                os.environ.pop(f"{family.upper()}_BASE_URL", None)
            else:
                os.environ[f"{family.upper()}_BASE_URL"] = previous
    plan.update(status="running", retry_policy={"seed_attempts": args.seed_attempts,
                                               "delay_seconds": args.retry_delay})
    write_json(batch / "batch.json", plan)

    # Reuse the exact saved snapshot on restart; never refresh a resumed batch.
    from experiments.run_redesign import SYMBOLS, split_train_test, TEST_SESSIONS
    from utils.data_utils import (load_pair_selection_price_history, load_offline_dataset,
                                 next_trading_day, validate_pair_selection_history, apply_cached_prices)
    import pandas as pd
    data = load_offline_dataset(SYMBOLS, news_dir=str(ROOT / "data/news"))
    dates = [*sorted(data), next_trading_day(max(data))]
    snapshot = batch / "inputs/pool.csv"
    if plan.get("snapshot_sha256"):
        if not snapshot.is_file() or digest(snapshot) != plan["snapshot_sha256"]:
            raise ValueError("Saved price snapshot changed or is missing; refusing to refresh")
        pool = validate_pair_selection_history(pd.read_csv(snapshot, parse_dates=["date"]).set_index("date"),
                                               SYMBOLS, dates)
    else:
        if plan["runs"]:
            raise ValueError("Cannot resume seed results without their original price snapshot")
        pool, source = load_pair_selection_price_history(SYMBOLS, decision_dates=dates,
                                                        cache_dir=str(ROOT / "data/cache"))
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(snapshot, index_label="date")
        plan.update(price_source=source, snapshot_sha256=digest(snapshot))
        write_json(batch / "batch.json", plan)
    env.update(MAIN_POOL_SNAPSHOT=str(snapshot), MAIN_POOL_SOURCE=plan["price_source"])

    if args.check_only:
        train_dates, test_dates = split_train_test(sorted(data))
        if not train_dates or len(test_dates) != TEST_SESSIONS:
            raise ValueError("Check requires the configured main train/test windows")
        settlement = next_trading_day(test_dates[-1])
        apply_cached_prices(data, pool, SYMBOLS, settlement)
        for model, client in clients.items():
            probe_model(client, model, args, plan, batch)
        plan.update(status="checked", data_check={"train_steps": len(train_dates),
                    "test_steps": len(test_dates), "settlement_date": settlement.isoformat()})
        write_json(batch / "batch.json", plan)
        print(f"CHECK_OK models={len(args.models)} train_steps={len(train_dates)} "
              f"test_steps={len(test_dates)} settlement={settlement} output_dir={batch}")
        return 0

    for family, model in zip(args.families, args.models):
        pending = []
        for seed in args.seeds:
            path = run_directory(batch / "runs", family, model, seed)
            relative = path.relative_to(batch)
            entry = next((r for r in plan["runs"] if
                          (r["provider"], r["model"], r["seed"]) == (family, model, seed)), None)
            if entry is None:
                entry = {"provider": family, "model": model, "seed": seed, "path": str(relative),
                         "status": "pending", "exit_code": None, "attempts": []}
                plan["runs"].append(entry)
            if completed_seed(path, family, model, seed, plan["sizing"], plan["input_signature"]["code_hash"]):
                entry.update(status="completed", exit_code=0)
                if entry.get("attempts") and entry["attempts"][-1]["status"] == "running":
                    entry["attempts"][-1].update(status="completed", exit_code=0, recovered=True)
                print(f"SKIP_COMPLETED model={model} seed={seed}", flush=True)
            else:
                pending.append((seed, path, relative, entry))
        write_json(batch / "batch.json", plan)
        if not pending:
            continue
        probe_model(clients[model], model, args, plan, batch)
        for seed, path, relative, entry in pending:
            if input_signature(ROOT, env) != plan["input_signature"]:
                raise ValueError("Code, news, or protocol changed during the batch")
            console = batch / "console" / relative.relative_to("runs") / "console.log"
            for attempt in range(1, args.seed_attempts + 1):
                archive = archive_attempt(batch, relative, console)
                if archive:
                    entry.setdefault("archives", []).append(archive)
                    if entry.get("attempts") and entry["attempts"][-1]["status"] == "running":
                        entry["attempts"][-1].update(status="interrupted", archive=archive)
                entry.update(status="running", exit_code=None)
                attempt_record = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
                entry.setdefault("attempts", []).append(attempt_record)
                write_json(batch / "batch.json", plan)
                command = [sys.executable, "-m", "experiments.run_seed", "--provider", family,
                           "--model", model, "--seed", str(seed), "--sizing", args.sizing,
                           "--output-dir", str(batch / "runs")]
                code = run_seed_process(command, {**env, "PYTHONHASHSEED": str(seed)}, console)
                if code == 0 and not completed_seed(path, family, model, seed, plan["sizing"],
                                                   plan["input_signature"]["code_hash"]):
                    raise RuntimeError("Seed exited successfully without complete artifacts")
                error_type = None
                manifest_file = path / "manifest.json"
                if code and manifest_file.is_file():
                    error_type = json.loads(manifest_file.read_text()).get("error_type")
                entry.update(exit_code=code, status="completed" if code == 0 else "failed")
                attempt_record.update(status=entry["status"], exit_code=code, error_type=error_type)
                write_json(batch / "batch.json", plan)
                if code == 0:
                    break
                if error_type in ("InferenceContractError", "InferenceServiceError", "SystemExit", "ValueError"):
                    raise RuntimeError(f"Non-retryable seed failure: {error_type}; logs preserved")
                retryable = error_type == "TransientInferenceError" or (code < 0 and code != -2)
                if not retryable or attempt == args.seed_attempts:
                    break
                print(f"SEED_RETRY seed={seed} attempt={attempt + 1}/{args.seed_attempts} "
                      f"wait_seconds={args.retry_delay:g}", flush=True)
                time.sleep(args.retry_delay)
    if any(r["status"] != "completed" for r in plan["runs"]):
        raise RuntimeError("Some seeds failed; rerun the same command to resume. All logs preserved.")
    finish_summary(batch)
    plan.update(status="completed")
    plan.pop("error_type", None)
    write_json(batch / "batch.json", plan)
    print(f"BATCH_OK models={len(args.models)} seeds_per_model={len(args.seeds)} sizing={','.join(plan['sizing'])} output_dir={batch}")
    print(f"Saved seed tables and median/max figures: {batch}")
    return 0


def probe_model(client, model, args, plan, batch):
    # Model-list readiness uses the same bounded backoff; inference retries live in the transport.
    attempts = 1 if args.check_only else args.seed_attempts
    for attempt in range(1, attempts + 1):
        try:
            if getattr(client, "provider", None) in ("openai", "deepseek", "gemini"):
                break  # Hosted aliases need not appear in model listings; probe actual inference.
            available = {entry.id for entry in client._get_client().models.list().data}
            if model not in available:
                raise ValueError(f"Model {model!r} is not served at {client.base_url}")
            break
        except Exception as exc:
            if not is_transient_service_error(exc) or attempt == attempts:
                raise
            print(f"SERVER_RETRY attempt={attempt + 1}/{attempts} wait_seconds={args.retry_delay:g}", flush=True)
            time.sleep(args.retry_delay)
    started = time.monotonic()
    proof = client.validate_server()
    plan.setdefault("preflight", {})[model] = {**proof, "elapsed_seconds": time.monotonic() - started,
                                              "usage": client.usage_summary()}
    write_json(batch / "batch.json", plan)


if __name__ == "__main__":
    raise SystemExit(main())
