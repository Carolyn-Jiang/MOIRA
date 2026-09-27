"""Main pair/direction discovery protocol, training and per-seed artifacts."""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
from llm.policy_agent import PolicyAgent
from utils.data_utils import (load_offline_dataset, load_pair_selection_price_history,
                             validate_pair_selection_history, PAIR_HISTORY_YEARS,
                             next_trading_day, apply_cached_prices)
from utils.statarb import rank_multihorizon_profiles, discover_pairs_coint10y
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
TRAIN_SESSIONS = int(os.getenv("TRAIN_SESSIONS", "27"))
TEST_SESSIONS = 75
SYMBOLS = ["AAPL", "ADBE", "AMZN", "BMRN", "CRM", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
HORIZON_YEARS = {"10Y": PAIR_HISTORY_YEARS}
EPS = 1e-9
MIN_TRADES_FOR_REFLECTION = 2
EARLY_STOP_PROMPT_REJECTS = 2
EARLY_STOP_PNL_REL = 0.05

class RunLogger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dirs = {
            name: root / "logs" / name
            for name in (
                "policies", "reflections", "pair_selection",
                "policy_updates", "convergence", "training", "testing",
                "pair_monitoring", "pair_reflections", "pair_lifecycle",
                "action_prompt", "llm_calls",
            )
        }
        # Clean managed dirs so logs reflect ONLY the current run (reproducibility).
        import shutil
        for d in self.dirs.values():
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        out = root / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        for png in out.glob("*.png"):
            png.unlink()

    def write_text(self, group: str, filename: str, text: str) -> Path:
        p = self.dirs[group] / filename
        p.write_text(text, encoding="utf-8")
        return p

    def write_json(self, group: str, filename: str, obj) -> Path:
        p = self.dirs[group] / filename
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        return p


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _dataset_hash(env_data: Dict[date, Dict]) -> str:
    h = hashlib.sha256()
    for d in sorted(env_data.keys()):
        h.update(d.isoformat().encode())
        for s in sorted(env_data[d]["price"]):
            h.update(f"{s}:{env_data[d]['price'][s]:.6f}".encode())
    return h.hexdigest()[:16]


def _code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(PROJECT_ROOT)
    for folder in ("experiments", "llm", "utils", "strategies"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".txt"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def build_agent(backend: str, model: str, symbols: Sequence[str],
                log_dir: Optional[str] = None, seed: Optional[int] = None) -> PolicyAgent:
    """Offline is a test fixture only; production calls use the selected provider."""
    from llm.backbone import build_llm
    if backend == "offline":
        return PolicyAgent(backend="offline", symbols=symbols)
    llm = build_llm(provider=backend, model=model, log_dir=log_dir, seed=seed)
    return PolicyAgent(backend="open_source", llm=llm, symbols=symbols)



def early_stop_decision(
    consecutive_prompt_rejects: int, pnl_rel_changes: Sequence[float],
    prompt_rejects_limit: int = EARLY_STOP_PROMPT_REJECTS,
    pnl_rel_limit: float = EARLY_STOP_PNL_REL,
) -> Tuple[bool, str]:
    """Step 3 early-stop rule (``max_cycles`` is only an upper bound).

    early_stop = (consecutive_prompt_rejects >= 2)
                 OR (two consecutive relative PnL changes < 5%)

    Returns ``(early_stop, reason)``.
    """
    by_rejects = consecutive_prompt_rejects >= prompt_rejects_limit
    by_pnl = (len(pnl_rel_changes) >= 2
              and pnl_rel_changes[-1] < pnl_rel_limit
              and pnl_rel_changes[-2] < pnl_rel_limit)
    reason = ("two_consecutive_prompt_rejections" if by_rejects
              else "two_consecutive_relative_pnl_changes_below_5_percent" if by_pnl else "")
    return bool(by_rejects or by_pnl), reason


def split_train_test(dates: List[date]) -> Tuple[List[date], List[date]]:
    """First ``TRAIN_SESSIONS`` trading sessions -> train; the remainder -> test.

    Count-based split (the boundary is the ``TRAIN_SESSIONS``-th session). On the bundled
    offline dataset (102 sessions, 2025-01-02 -> 2025-05-30) the default ``TRAIN_SESSIONS
    = 27`` yields train = 2025-01-02 -> 2025-02-11 (27 sessions) and test = 2025-02-12 ->
    2025-05-30 (75 sessions). The 2025-01-09 special market closure is excluded, keeping
    the original test dates unchanged.
    """
    n = max(0, min(TRAIN_SESSIONS, len(dates)))
    return list(dates[:n]), list(dates[n:])


def causal_horizon_dfs(pool_df, as_of) -> Dict[str, "pd.DataFrame"]:
    """Only the preceding ten calendar years: [as_of - 10 years, as_of).

    The selection day's price and all later prices are excluded.
    """
    ts = pd.Timestamp(as_of)
    hist = pool_df[pool_df.index < ts]
    return {h: hist[hist.index >= (ts - pd.DateOffset(years=y))]
            for h, y in HORIZON_YEARS.items()}


def causal_profiles(pool_df, symbols, as_of, cache=None):
    """Ten-year profiles computed from prices strictly before ``as_of`` (cached by date)."""
    key = pd.Timestamp(as_of).normalize()
    if cache is not None and key in cache:
        return cache[key]
    profs = rank_multihorizon_profiles(causal_horizon_dfs(pool_df, as_of), symbols)
    if cache is not None:
        cache[key] = profs
    return profs


def make_discovery_provider(pool_df, symbols, cache=None, top_k: int = 5):
    """Stage-1 provider: at decision date ``t``, causal 10Y profiles (prices < t) ->
    Top-``top_k`` pairs ranked by 10Y cointegration p-value. Returns a callable
    ``provider(as_of, prices=None) -> [{"pair":(A,B), "pc": StatArbPair}, ...]``."""
    cache = {} if cache is None else cache

    def provider(as_of, prices=None):
        profs = causal_profiles(pool_df, symbols, as_of, cache)
        selected, diag = discover_pairs_coint10y(profs, top_k=top_k, return_diagnostics=True)
        diag["date"] = as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of)
        provider.last_diagnostics = diag      # surfaced to the engine for diagnostics logging
        out = []
        for prof, _cp in selected:
            pc = prof.horizon_metrics("10Y")
            if pc is not None:
                out.append({"pair": tuple(prof.pair), "pc": pc})
        return out
    provider.last_diagnostics = None
    return provider


def _assign_realized_pnl(direction_memory, result, selection_frequency: int) -> None:
    """Attribute realized FORWARD PnL to each Direction-Memory decision (daily-aware).

    A decision recorded at engine step ``start_k`` (its ``decision_index``) sets the
    position that earns its return on the *following* steps, until the next decision
    ``start_{k+1}``. Realized PnL is therefore summed over ``(start_k, start_{k+1}]`` =
    ``daily_pnls[start_k+1 : start_{k+1}+1]``. For daily decisions this is the next day's
    return, including the price-only terminal settlement when supplied. This is robust to skipped
    decision steps (they simply have no record, so no PnL is shifted). Records without a
    stored index fall back to the legacy fixed-stride mapping for backward compatibility.
    """
    pnls = result.daily_pnls()
    n = len(pnls)
    dm_records = getattr(direction_memory, "records", [])
    have_idx = all(r.get("decision_index") is not None for r in dm_records) and bool(dm_records)
    if have_idx:
        starts = [int(r["decision_index"]) for r in dm_records]
        for k, start in enumerate(starts):
            lo = start + 1
            hi = (starts[k + 1] + 1) if k + 1 < len(starts) else n
            pnl = sum(pnls[lo:hi])
            direction_memory.set_realized_pnl(k, pnl)
        return
    # Legacy fallback: fixed-stride blocks (no decision_index recorded).
    starts = list(range(0, n, selection_frequency))
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        pnl = sum(pnls[start:end])
        direction_memory.set_realized_pnl(k, pnl)


def _episode_counts(direction_memory) -> Dict[str, int]:
    """Count actions across a round's Direction-Memory decisions (final executed)."""
    recs = list(getattr(direction_memory, "records", []) or [])
    ls = sum(1 for r in recs if str(r.get("action")) == "LONG_SHORT")
    sl = sum(1 for r in recs if str(r.get("action")) == "SHORT_LONG")
    flat = sum(1 for r in recs if str(r.get("action")) == "FLAT")
    return {"trade_count": ls + sl, "flat_count": flat,
            "long_short_count": ls, "short_long_count": sl}


def _pair_switch_summary(result, cycle: int) -> Dict[str, object]:
    """Per-cycle pair-selection / switching detail from a training run's WindowResult.

    Captures, for each decision in the (active-prompt) baseline run: the date, selected
    pair, executed action, and whether it was break-triggered; plus the ordered pair
    timeline, the number of pair switches (consecutive distinct pairs), and any
    break/switch events. This is what was previously NOT persisted for training rounds.
    """
    decs = [{"date": d.get("date"), "decision_index": d.get("decision_index"),
             "pair": d.get("selected_pair"), "action": d.get("final_executed_action"),
             "break_triggered": d.get("break_triggered")}
            for d in getattr(result, "decisions", [])]
    timeline = [d["pair"] for d in decs]
    n_switches = sum(1 for k in range(1, len(timeline)) if timeline[k] != timeline[k - 1])
    return {"cycle": cycle, "pair_timeline": timeline, "n_pair_switches": n_switches,
            "decisions": decs, "break_events": list(getattr(result, "break_events", []))}


def episode_status(direction_memory, min_trades: int = MIN_TRADES_FOR_REFLECTION) -> str:
    """Classify a training round's evidence level (anti-FLAT-degeneration).

      * NO_EVIDENCE  -- zero executed directional trades (all FLAT): no learning at all.
      * LOW_EVIDENCE -- fewer than ``min_trades`` executed: reflection for analysis only.
      * VALID        -- ``>= min_trades`` executed: full reflection / accept-reject / early-stop.
    """
    trades = _episode_counts(direction_memory)["trade_count"]
    if trades == 0:
        return "NO_EVIDENCE"
    if trades < min_trades:
        return "LOW_EVIDENCE"
    return "VALID"


def run_discovery_training(
    logger, *, train_dates, env_data, agent, pool_df, symbols,
    lookback: int = 20, selection_frequency: int = 23, beta_hedged: bool = False,
    max_cycles: int = 8, pair_memory=None, prompt_memory=None,
    lifecycle: bool = False, break_check_frequency: int = 23,
    profile_cache=None,
):
    """Train the Direction-Discovery prompt: each round, run the discovery window on the
    train set, store decisions + realized PnL in a FRESH DirectionMemory, reflect into
    transferable (filtered) lessons, evaluate a candidate prompt, accept iff training PnL
    improves (else keep), apply the early-stop rule, then CLEAR DirectionMemory. Pair
    Selection Memory persists. Returns ``(prompt_memory, pair_memory, cycles)``."""
    from llm.policy_agent import (PairSelectionMemory, DirectionMemory,
                                  DirectionPromptMemory, load_direction_discovery_prompt)
    from utils.engine import simulate_discovery_window
    provider = make_discovery_provider(pool_df, symbols, profile_cache)
    if pair_memory is None:
        pair_memory = PairSelectionMemory()
    if prompt_memory is None:
        prompt_memory = DirectionPromptMemory(base_prompt=load_direction_discovery_prompt(),
                                              symbols=list(symbols))

    def run(prompt):
        dm = DirectionMemory()
        res = simulate_discovery_window(
            dates=train_dates, env_data=env_data, symbols=symbols, agent=agent,
            discovery_provider=provider, lookback=lookback,
            selection_frequency=selection_frequency, beta_hedged=beta_hedged,
            pair_memory=pair_memory, action_prompt=prompt, direction_memory=dm,
            lifecycle=lifecycle, break_check_frequency=break_check_frequency)
        _assign_realized_pnl(dm, res, selection_frequency)
        return res, dm

    cycles, round_pnls, pnl_rel, consec_rej = [], [], [], 0
    accepted_updates = 0          # number of prompt updates accepted so far
    train_switches = []           # per-cycle pair-switch detail (training-side log)
    for cycle in range(1, max_cycles + 1):
        baseline, dm = run(prompt_memory.active_prompt())
        reward_before = baseline.cumulative_pnl
        # training-side pair-switch logging (active-prompt baseline run)
        switch_detail = _pair_switch_summary(baseline, cycle)
        train_switches.append(switch_detail)
        switch_fields = {"pair_timeline": switch_detail["pair_timeline"],
                         "n_pair_switches": switch_detail["n_pair_switches"]}
        # ----- episode evidence gating (anti-FLAT-collapse) -----
        status = episode_status(dm)
        counts = _episode_counts(dm)
        if status == "NO_EVIDENCE":
            # A fully-FLAT episode provides NO evidence about Direction quality: no
            # reflection, no prompt update, no accept/reject, no early-stop contribution.
            cycles.append({"cycle": cycle, "reward_before": reward_before,
                           "reward_after": reward_before, "prompt_accepted": False,
                           "episode_status": status, "n_decisions": len(dm.records),
                           "reflection": "(skipped: NO_EVIDENCE — all FLAT)",
                           "early_stop": False, "early_stop_reason": None,
                           "accepted_prompt_updates": accepted_updates,
                           "early_stop_eligible": accepted_updates >= 1,
                           "early_stop_blocked_reason": None, **switch_fields, **counts})
            dm.clear()
            continue
        refl = agent.reflect_direction(dm)
        new_lessons = refl["lessons"]
        if status == "LOW_EVIDENCE":
            # Reflection generated for ANALYSIS only: not accepted, no rollback, and it
            # must NOT contribute to early-stop convergence.
            cycles.append({"cycle": cycle, "reward_before": reward_before,
                           "reward_after": reward_before, "prompt_accepted": False,
                           "episode_status": status, "n_decisions": len(dm.records),
                           "reflection": refl["reflection_text"],
                           "early_stop": False, "early_stop_reason": None,
                           "accepted_prompt_updates": accepted_updates,
                           "early_stop_eligible": accepted_updates >= 1,
                           "early_stop_blocked_reason": None, **switch_fields, **counts})
            dm.clear()
            continue
        # ----- VALID episode: normal accept/reject/rollback + early-stop -----
        cand_run, _ = run(prompt_memory.candidate_prompt(new_lessons))
        reward_after = cand_run.cumulative_pnl
        accepted = reward_after > reward_before          # strict improvement
        if accepted:
            prompt_memory.accept(new_lessons, source_round=cycle,
                                 reward_before=reward_before, reward_after=reward_after)
            accepted_updates += 1
        else:
            prompt_memory.reject(new_lessons, source_round=cycle,
                                 reward_before=reward_before, reward_after=reward_after,
                                 reason="candidate prompt did not improve training PnL")
        consec_rej = consec_rej + 1 if not accepted else 0
        if round_pnls:
            pnl_rel.append(abs(reward_before - round_pnls[-1]) / max(abs(round_pnls[-1]), EPS))
        round_pnls.append(reward_before)
        # Early stop is GATED: it may only fire once at least one prompt update has been
        # accepted. Before that, all early-stop criteria (consecutive rejections,
        # convergence, relative-PnL thresholds) are blocked so training cannot freeze at
        # the base prompt before any accepted learning.
        would_stop, reason = early_stop_decision(consec_rej, pnl_rel)
        early_stop_eligible = accepted_updates >= 1
        early_stop = bool(would_stop and early_stop_eligible)
        blocked_reason = ("no_accepted_prompt_update_yet"
                          if (would_stop and not early_stop_eligible) else None)
        cycles.append({"cycle": cycle, "reward_before": reward_before,
                       "reward_after": reward_after, "prompt_accepted": accepted,
                       "episode_status": status, "n_decisions": len(dm.records),
                       "reflection": refl["reflection_text"],
                       "early_stop": early_stop,
                       "early_stop_reason": (reason if early_stop else None),
                       "accepted_prompt_updates": accepted_updates,
                       "early_stop_eligible": early_stop_eligible,
                       "early_stop_blocked_reason": blocked_reason, **switch_fields, **counts})
        dm.clear()                                       # clear Direction Memory each round
        if early_stop:
            break
    if logger is not None:
        logger.write_json("training", "training_pair_switches.json", train_switches)
    return prompt_memory, pair_memory, cycles


def run_discovery_test(
    *, test_dates, env_data, agent, pool_df, symbols, frozen_prompt,
    lookback: int = 20, selection_frequency: int = 23, beta_hedged: bool = False,
    pair_memory=None, lifecycle: bool = False, break_check_frequency: int = 23,
    direction_memory=None, history_seed=None, profile_cache=None, settlement_date=None,
    initial_pair=None, allow_reselection=True,
):
    """Frozen global-prompt test pass with no prompt updates.

    Daily counterfactual records still condition later actions inside the test sequence;
    ``direction_memory`` only exports those decisions for analysis. ``history_seed`` adds
    pre-test prices to causal rolling statistics without importing learned prompt state.
    """
    from utils.engine import simulate_discovery_window
    provider = make_discovery_provider(pool_df, symbols, profile_cache)
    return simulate_discovery_window(
        dates=test_dates, env_data=env_data, symbols=symbols, agent=agent,
        discovery_provider=provider, lookback=lookback,
        history_seed=history_seed,
        selection_frequency=selection_frequency, beta_hedged=beta_hedged,
        pair_memory=pair_memory, action_prompt=frozen_prompt, direction_memory=direction_memory,
        lifecycle=lifecycle, break_check_frequency=break_check_frequency,
        settlement_date=settlement_date, initial_pair=initial_pair,
        allow_reselection=allow_reselection)


def run_redesign_experiment(*, seed=None, output_root=None, sizing=None) -> Path:
    """Run the main protocol, preserving each seed. Plot only via plot_main_results.

    Existing seed directories are refused, including failed/partial runs.
    """
    import random
    import numpy as np
    from llm.backbone import PROVIDERS, resolve_provider
    from experiments.main_artifacts import run_directory, write_json

    backend = os.getenv("LLM_BACKEND", "offline").lower()
    provider = ("offline" if backend == "offline" else resolve_provider(
        backend)[0])
    model = (os.getenv("LLM_MODEL") or os.getenv(f"{provider.upper()}_MODEL"))
    if provider == "offline":
        model = "offline-deterministic"
    seed = int(os.getenv("LLM_SEED", "1103")) if seed is None else int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    sizing = sizing or os.getenv("LLM_SIZING", "both").lower()
    modes = ["nobeta", "beta"] if sizing in ("both", "all") else [sizing]
    if any(m not in ("nobeta", "beta") for m in modes):
        raise ValueError("LLM_SIZING must be 'nobeta', 'beta', or 'both'")
    agent = build_agent(backend, model, SYMBOLS, seed=seed)
    root = run_directory(output_root or os.getenv("MAIN_OUTPUT_DIR") or
                         Path(PROJECT_ROOT) / "runs" / "main", provider, model, seed)
    # Reserve before constructing RunLogger, whose legacy behavior cleans its dirs.
    root.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now().isoformat(timespec="microseconds")
    manifest = {"schema_version": 1, "status": "running", "run_id": run_id,
                "backend": backend, "provider": provider, "model": model, "seed": seed,
                "sizing": modes, "git_rev": os.getenv("MAIN_CODE_REVISION") or _git_rev(),
                "execution_git_rev": _git_rev(), "code_hash": _code_hash(), "plots_generated": False}
    if hasattr(agent.llm, "inference_config"):
        manifest["inference"] = agent.llm.inference_config()
    write_json(root / "manifest.json", manifest)
    random.seed(seed)
    np.random.seed(seed)
    try:
        _run_redesign_experiment(root, agent, manifest)
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        write_json(root / "manifest.json", manifest)
        if hasattr(agent.llm, "usage_summary"):
            write_json(root / "usage.json", agent.llm.usage_summary())
        print(f"SEED_RESULT seed={seed} status=failed error={type(exc).__name__} run_dir={root}",
              flush=True)
        raise
    manifest["status"] = "completed"
    write_json(root / "manifest.json", manifest)
    print(f"Saved seed={seed}: {root}\nBatch runner generates median/max figures after all five seeds complete.")
    return root


def _run_redesign_experiment(root, agent, manifest) -> None:
    import copy
    from experiments.baselines import main_strategy_labels
    from experiments.main_artifacts import result_metrics, write_json
    backend, model, run_id = (manifest[k] for k in ("backend", "model", "run_id"))
    run_seed, modes = manifest["seed"], manifest["sizing"]
    selection_frequency = int(os.getenv("LLM_SELECTION_FREQUENCY", "23"))
    lookback = int(os.getenv("LLM_LOOKBACK", "20"))
    max_cycles = int(os.getenv("LLM_MAX_CYCLES", "8"))
    logger = RunLogger(root)
    live = backend != "offline"
    if live:
        agent.llm.log_dir = str(logger.dirs["llm_calls"])
        manifest["preflight"] = agent.llm.validate_server()
        write_json(root / "manifest.json", manifest)
    print("=" * 70)
    print(f"PAIR/DIRECTION DISCOVERY  |  backend={backend}  model={model}  seed={run_seed}")
    print("=" * 70)

    env_data = load_offline_dataset(SYMBOLS, news_dir=str(Path(PROJECT_ROOT) / "data" / "news"))
    all_dates = sorted(env_data.keys())
    train_dates, test_dates = split_train_test(all_dates)
    if not train_dates or not test_dates:
        raise ValueError("TRAIN_SESSIONS must leave non-empty train and test windows")
    if len(test_dates) != TEST_SESSIONS:
        raise ValueError(f"Main experiment requires {TEST_SESSIONS} test decisions; got {len(test_dates)}")
    settlement_date = next_trading_day(test_dates[-1])
    price_dates = [*all_dates, settlement_date]
    valuation_dates = [*test_dates, settlement_date]
    print(f"Train {len(train_dates)} ({train_dates[0]}->{train_dates[-1]}) | "
          f"Test {len(test_dates)} ({test_dates[0]}->{test_dates[-1]})")
    snapshot = os.getenv("MAIN_POOL_SNAPSHOT")
    if snapshot:
        pool_df = pd.read_csv(snapshot, parse_dates=["date"]).set_index("date")
        pool_df = validate_pair_selection_history(pool_df, SYMBOLS, price_dates)
        price_source = os.environ["MAIN_POOL_SOURCE"]
    else:
        pool_df, price_source = load_pair_selection_price_history(
            SYMBOLS, decision_dates=price_dates,
            cache_dir=str(Path(PROJECT_ROOT) / "data" / "cache"))
    env_data = apply_cached_prices(env_data, pool_df, SYMBOLS, settlement_date)
    manifest.update(dataset_hash=_dataset_hash(env_data), price_source=price_source,
                    input_hash=hashlib.sha256(json.dumps(
                        {d.isoformat(): env_data[d] for d in sorted(env_data)},
                        sort_keys=True, default=str).encode()).hexdigest(),
                    pool_hash=hashlib.sha256(pool_df.to_csv().encode()).hexdigest(),
                    symbols=SYMBOLS, train_dates=[d.isoformat() for d in train_dates],
                    test_dates=[d.isoformat() for d in test_dates],
                    valuation_dates=[d.isoformat() for d in valuation_dates],
                    settlement_date=settlement_date.isoformat(),
                    config={"pair_history_years": PAIR_HISTORY_YEARS,
                            "pair_selection_rule": "validation_window_break_only",
                            "pair_history_window": "[decision_date - 10 calendar years, decision_date)",
                            "lookback": lookback, "selection_frequency": selection_frequency,
                            "max_cycles": max_cycles, "train_sessions": TRAIN_SESSIONS,
                            "test_sessions": TEST_SESSIONS,
                            "terminal_settlement": "next_session_price_only",
                            "break_check_frequency": int(os.getenv("LLM_BREAK_CHECK_FREQUENCY", "23"))})
    write_json(root / "manifest.json", manifest)

    # ---- Stage-2 prompt training (No-Beta), then freeze ----
    print("\n[TRAINING] Direction-Discovery prompt learning (No-Beta)")
    break_check_frequency = int(os.getenv("LLM_BREAK_CHECK_FREQUENCY", "23"))
    prompt_memory, pair_memory, cycles = run_discovery_training(
        logger, train_dates=train_dates, env_data=env_data, agent=agent, pool_df=pool_df,
        symbols=SYMBOLS, lookback=lookback, selection_frequency=selection_frequency,
        beta_hedged=False, max_cycles=max_cycles,
        lifecycle=True, break_check_frequency=break_check_frequency)
    for c in cycles:
        print(f"  cyc{c['cycle']}: before={c['reward_before']:+.4f} after={c['reward_after']:+.4f} "
              f"accepted={c['prompt_accepted']} | {c['early_stop_reason'] or ''}")
    frozen = prompt_memory.active_prompt()
    logger.write_text("policies", "final_direction_prompt.txt", frozen)
    logger.write_json("training", "direction_training_cycles.json", cycles)

    # ---- Frozen test: SEPARATE no-beta / beta runs (Step 8) ----
    from experiments.baselines import build_mode_comparison
    from utils.engine import write_discovery_logs, write_counterfactual_logs
    from llm.policy_agent import DirectionMemory
    history_seed = {d: env_data[d] for d in train_dates}
    label_nb, label_b = main_strategy_labels(backend, model)

    # training counterfactual log (one frozen train-window pass, no learning)
    train_dm = DirectionMemory()
    train_run = run_discovery_test(test_dates=train_dates, env_data=env_data, agent=agent,
        pool_df=pool_df, symbols=SYMBOLS, frozen_prompt=frozen, lookback=lookback,
        selection_frequency=selection_frequency, beta_hedged=False, pair_memory=copy.deepcopy(pair_memory),
        lifecycle=True, break_check_frequency=break_check_frequency, direction_memory=train_dm)
    write_counterfactual_logs(train_run, str(logger.dirs["testing"]), "train")

    summaries = []
    for mode in modes:
        bh = (mode == "beta")
        tag = f"test_{mode}"
        main_label = label_b if bh else label_nb
        print(f"\n[TESTING] frozen Direction-Discovery policy ({mode})")
        dm = DirectionMemory()
        res = run_discovery_test(test_dates=test_dates, env_data=env_data, agent=agent,
            pool_df=pool_df, symbols=SYMBOLS, frozen_prompt=frozen, lookback=lookback,
            selection_frequency=selection_frequency, beta_hedged=bh, pair_memory=copy.deepcopy(pair_memory),
            lifecycle=True, break_check_frequency=break_check_frequency, direction_memory=dm,
            settlement_date=settlement_date)
        if (len(res.records) != TEST_SESSIONS or len(res.normalized_equity) != TEST_SESSIONS + 1
                or len(res.decisions) != TEST_SESSIONS
                or any(d.get("actual_pnl") is None for d in res.decisions)):
            raise RuntimeError("Incomplete main test: every decision must have a settled return")
        _assign_realized_pnl(dm, res, selection_frequency)
        # When both passes run into the same dir, tag the audit logs per mode so the
        # Beta pass does not OVERWRITE the No-Beta pass's lifecycle_log/decision_log/etc.
        disc_tag = mode
        write_discovery_logs(res, str(logger.dirs["testing"]), backend=backend, tag=disc_tag)
        write_counterfactual_logs(res, str(logger.dirs["testing"]), tag)
        write_json(root / f"switch_events_{mode}.json", res.switch_events)
        main_sim = {"normalized_equity": res.normalized_equity, "n_switches": len(res.switch_events)}
        rows, equity = build_mode_comparison(
            mode, dates=test_dates, env_data=env_data, symbols=SYMBOLS,
            main_records=res.records, main_sim=main_sim, pool_df=pool_df,
            main_label=main_label, lookback=lookback, history_seed=history_seed,
            settlement_date=settlement_date)
        pd.DataFrame(rows).to_csv(root / f"comparison_{mode}.csv", index=False)
        write_json(root / f"comparison_{mode}.json", rows)
        pd.DataFrame({"date": [d.isoformat() for d in valuation_dates], **equity}).to_csv(
            root / f"equity_{mode}.csv", index=False)
        write_json(root / f"settlement_{mode}.json", res.settlement)
        metric = result_metrics(res)
        if live and hasattr(agent.llm, "usage_summary"):
            metric["llm_failed_calls_total"] = agent.llm.usage_summary().get("failed_call_count")
        metric["quality"] = ("contains_fallbacks" if metric["decision_failures"] else
                             "recovered_service_errors" if metric.get("llm_failed_calls_total") else
                             "no_recorded_failures")
        summary = {"provider": manifest["provider"], "model": model, "seed": run_seed,
                   "thinking_mode": manifest.get("inference", {}).get("thinking_mode", "not_applicable"),
                   "sizing": mode, **metric}
        summaries.append(summary)
        write_json(root / "metrics.json", summaries)
        pd.DataFrame(summaries).to_csv(root / "metrics.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        print("SEED_RESULT " + json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)

    usage = agent.llm.usage_summary() if live and hasattr(agent.llm, "usage_summary") else {}
    write_json(root / "usage.json", usage)
