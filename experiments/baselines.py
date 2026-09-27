"""Part 3: the 12-strategy comparison for the Pair/Direction Discovery redesign.

Strategy set (test window, frictionless 0 bps):
  Main LLM (run by ``run_redesign``):
    1. LLM Direction Discovery (No Beta)
    2. LLM Direction Discovery (Beta Hedged)
  Timeline baselines -- reuse the EXACT per-day pair the main LLM held, but apply a
  fixed mechanical action (flatten-before-switch enforced, no overlapping exposure):
    3. Timeline LONG_SHORT          4. Timeline SHORT_LONG
    5. Timeline Equal-Weight Long   6. Timeline Beta-Hedged LONG_SHORT
    7. Timeline Beta-Hedged SHORT_LONG
  Fixed 10Y-cointegration baselines -- hold the single most-cointegrated pair
  (selected causally as-of the test start) for the whole window:
    8. 10Y-Coint Fixed LONG_SHORT       9. 10Y-Coint Fixed SHORT_LONG
   10. 10Y-Coint Fixed Equal-Weight Long
   11. 10Y-Coint Fixed Beta-Hedged LONG_SHORT
   12. 10Y-Coint Fixed Beta-Hedged SHORT_LONG

All baselines are deterministic and use NO LLM, NO news, NO memory.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
from datetime import date

import pandas as pd

from strategies.types import Action
from utils.portfolio import (PortfolioManager, size_pair_position_fractional,
                             effective_hedge_ratio)
from utils.engine import compute_spread_z, _zeroed_targets
from utils.metrics import compute_performance_metrics
from utils.statarb import discover_pairs_coint10y, rank_multihorizon_profiles

# Fixed-action rules. EQUAL_WEIGHT_LONG longs both legs 0.5/0.5 (no beta variant).
LONG_SHORT = "LONG_SHORT"
SHORT_LONG = "SHORT_LONG"
EQUAL_WEIGHT_LONG = "EQUAL_WEIGHT_LONG"
_ACTION_MAP = {LONG_SHORT: Action.long_short, SHORT_LONG: Action.short_long}


def select_coint_pair(pool_df, symbols: Sequence[str], as_of) -> Tuple[str, str]:
    """The single most-cointegrated pair (Top-1 of the 10Y ranking), causal as-of date."""
    from experiments.run_redesign import causal_horizon_dfs  # lazy: avoids import cycle
    profs = rank_multihorizon_profiles(causal_horizon_dfs(pool_df, as_of), list(symbols))
    top = discover_pairs_coint10y(profs, top_k=1)
    if not top:
        raise ValueError("no pairs available for fixed-coint selection")
    return tuple(top[0][0].pair)


def _equal_weight_long_targets(prices, pair, portfolio_value, gross_fraction=1.0):
    """Long BOTH legs at half the gross notional each (market-directional baseline)."""
    a, b = pair
    tgt = {a: 0.0, b: 0.0}
    half = portfolio_value * gross_fraction / 2.0
    if prices.get(a, 0) > 0:
        tgt[a] = half / prices[a]
    if prices.get(b, 0) > 0:
        tgt[b] = half / prices[b]
    return tgt


def simulate_baseline_window(
    *, dates: Sequence[date], env_data: Dict[date, Dict], symbols: Sequence[str],
    pair_for_index: Callable[[int], Optional[Tuple[str, str]]], action: str,
    beta_hedged: bool = False, lookback: int = 20,
    history_seed: Optional[Dict[date, Dict]] = None, gross_allocation: float = 1.0,
    settlement_date: Optional[date] = None,
) -> Dict[str, object]:
    """Deterministic fixed-policy pair simulation.

    ``pair_for_index(i)`` gives the pair HELD on day ``i`` (``None`` = flat). On any
    pair change the target contains ONLY the new pair's legs (every other symbol is
    zeroed), so the prior pair is fully liquidated before the new one is established --
    no overlapping exposure and no leg carried across (frictionless self-financing).
    """
    symbols = list(symbols)
    portfolio = PortfolioManager(symbols, initial_cash=1.0)
    price_history: Dict[date, Dict[str, Dict[str, float]]] = {}
    if history_seed:
        for d in sorted(history_seed.keys()):
            price_history[d] = {"price": history_seed[d]["price"]}

    norm_equity: List[float] = []
    daily_pnls: List[float] = []
    held: List[Optional[Tuple[str, str]]] = []
    prev_norm, cum = 1.0, 0.0

    for i, d in enumerate(dates):
        prices = env_data[d]["price"]
        price_history[d] = {"price": prices}
        pair = pair_for_index(i)
        target = _zeroed_targets(symbols)
        if pair is not None:
            pv = max(portfolio._compute_equity(prices), 1e-9)
            if action == EQUAL_WEIGHT_LONG:
                target.update(_equal_weight_long_targets(prices, pair, pv, gross_allocation))
            else:
                beta = 1.0
                if beta_hedged:
                    _z, b = compute_spread_z(price_history, pair, lookback)
                    beta = effective_hedge_ratio(b, pair=pair, as_of=d.isoformat())
                target.update(size_pair_position_fractional(
                    prices=prices, pair=pair, portfolio_value=pv,
                    action=_ACTION_MAP[action], gross_fraction=gross_allocation,
                    hedge_ratio=beta))
        state = portfolio.step(prices, target)
        if state.get("rejected_target"):
            portfolio.step(prices, _zeroed_targets(symbols))
        norm = float(portfolio.normalized_equity_history[-1])
        reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
        prev_norm, cum = norm, cum + reward
        norm_equity.append(norm)
        daily_pnls.append(reward)
        held.append(pair)

    if settlement_date is not None:
        from utils.data_utils import next_trading_day
        if not dates or settlement_date != next_trading_day(dates[-1]):
            raise ValueError("Settlement must follow the last baseline decision")
        prices = env_data[settlement_date]["price"]
        import math
        if any(not math.isfinite(float(prices.get(s, float("nan"))))
               or float(prices.get(s, 0)) <= 0 for s in symbols):
            raise ValueError("Missing valid baseline settlement prices")
        portfolio.step(prices, _zeroed_targets(symbols))
        norm = float(portfolio.normalized_equity_history[-1])
        reward = (norm / prev_norm - 1.0) if prev_norm > 0 else 0.0
        norm_equity.append(norm)
        daily_pnls.append(reward)
        cum += reward

    switches = sum(1 for j in range(1, len(held))
                   if held[j] is not None and held[j - 1] is not None and held[j] != held[j - 1])
    return {"normalized_equity": norm_equity, "daily_pnls": daily_pnls,
            "cumulative_pnl": cum, "held": held, "n_switches": switches}


def _schedule_from_records(records) -> Callable[[int], Optional[Tuple[str, str]]]:
    """Per-day held pair taken verbatim from a main-LLM run's StepRecords
    ("A/B" -> (A, B); "FLAT"/no_trade -> None). This is the timeline baselines reuse."""
    sched: List[Optional[Tuple[str, str]]] = []
    for r in records:
        sp = getattr(r, "selected_pair", None) or (r.get("selected_pair") if isinstance(r, dict) else None)
        flat = getattr(r, "no_trade", None)
        if flat is None and isinstance(r, dict):
            flat = r.get("no_trade")
        if not sp or sp == "FLAT" or flat:
            sched.append(None)
        else:
            a, b = sp.split("/")
            sched.append((a, b))
    return lambda i: sched[i] if 0 <= i < len(sched) else None


def metrics_row(name: str, sim: Dict[str, object]) -> Dict[str, object]:
    eq = pd.Series(sim["normalized_equity"]) if sim["normalized_equity"] else pd.Series([1.0])
    m = compute_performance_metrics(eq, 1.0)
    return {"strategy": name, "final_value": round(m["final_value"], 4),
            "cum_return": round(m["final_value"] - 1.0, 4),
            "sharpe": (round(m["sharpe_ratio"], 3) if m["sharpe_ratio"] == m["sharpe_ratio"] else None),
            "max_drawdown": round(m["max_drawdown"], 4),
            "n_switches": sim.get("n_switches", 0)}


def main_strategy_labels(backend: str, model: str) -> Tuple[str, str]:
    """Labels for the two main (No-Beta / Beta-Hedged) strategies (Step 6).

    Offline backend results are explicitly an OFFLINE DETERMINISTIC VALIDATION and must
    NOT be reported as LLM output. A live backend is labeled with the actual model.
    """
    if str(backend).lower() == "offline":
        return ("Offline Deterministic Validation (No Beta)",
                "Offline Deterministic Validation (Beta Hedged)")
    return (f"LLM Direction Discovery — {model} (No Beta)",
            f"LLM Direction Discovery — {model} (Beta Hedged)")


def build_baseline_comparison(
    *, dates, env_data, symbols, main_records, main_nobeta_sim, main_beta_sim,
    pool_df, lookback: int = 20, history_seed=None,
    main_labels: Optional[Tuple[str, str]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict]]:
    """Build all 12 strategy rows + their equity curves. ``main_*_sim`` carry the main
    discovery runs' normalized equity (strategies 1-2); the timeline schedule comes from
    ``main_records`` (the No-Beta test records). ``main_labels`` names strategies 1-2
    (defaults to the LLM names; pass offline labels via ``main_strategy_labels`` for an
    offline run). Returns (rows, equity_by_name)."""
    timeline = _schedule_from_records(main_records)
    fixed_pair = select_coint_pair(pool_df, symbols, dates[0])
    fixed_for = lambda i: fixed_pair
    nobeta_label, beta_label = (main_labels or
                                ("LLM Direction Discovery (No Beta)",
                                 "LLM Direction Discovery (Beta Hedged)"))

    specs = [
        # (number, name, schedule, action, beta_hedged, precomputed_sim)
        (1, nobeta_label, None, None, False, main_nobeta_sim),
        (2, beta_label, None, None, True, main_beta_sim),
        (3, "Timeline LONG_SHORT", timeline, LONG_SHORT, False, None),
        (4, "Timeline SHORT_LONG", timeline, SHORT_LONG, False, None),
        (5, "Timeline Equal-Weight Long", timeline, EQUAL_WEIGHT_LONG, False, None),
        (6, "Timeline Beta-Hedged LONG_SHORT", timeline, LONG_SHORT, True, None),
        (7, "Timeline Beta-Hedged SHORT_LONG", timeline, SHORT_LONG, True, None),
        (8, f"10Y-Coint Fixed LONG_SHORT ({fixed_pair[0]}/{fixed_pair[1]})", fixed_for, LONG_SHORT, False, None),
        (9, f"10Y-Coint Fixed SHORT_LONG ({fixed_pair[0]}/{fixed_pair[1]})", fixed_for, SHORT_LONG, False, None),
        (10, f"10Y-Coint Fixed Equal-Weight Long ({fixed_pair[0]}/{fixed_pair[1]})", fixed_for, EQUAL_WEIGHT_LONG, False, None),
        (11, f"10Y-Coint Fixed Beta-Hedged LONG_SHORT ({fixed_pair[0]}/{fixed_pair[1]})", fixed_for, LONG_SHORT, True, None),
        (12, f"10Y-Coint Fixed Beta-Hedged SHORT_LONG ({fixed_pair[0]}/{fixed_pair[1]})", fixed_for, SHORT_LONG, True, None),
    ]
    rows, equity = [], {}
    for num, name, sched, action, bh, pre in specs:
        if pre is not None:
            sim = pre
        else:
            sim = simulate_baseline_window(
                dates=dates, env_data=env_data, symbols=symbols, pair_for_index=sched,
                action=action, beta_hedged=bh, lookback=lookback, history_seed=history_seed)
        row = metrics_row(f"{num}. {name}", sim)
        row["group"] = ("main" if num <= 2 else "timeline" if num <= 7 else "fixed_coint")
        rows.append(row)
        equity[f"{num}. {name}"] = sim["normalized_equity"]
    return rows, equity


def build_mode_comparison(
    mode: str, *, dates, env_data, symbols, main_records, main_sim, pool_df,
    main_label: str, lookback: int = 20, history_seed=None, settlement_date=None,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict]]:
    """Single-sizing comparison (Step 8): NO mixing of beta and no-beta in one table.

    ``mode='nobeta'`` -> 1 LLM (no-beta) + 6 no-beta baselines (Timeline LS/SL/EW-Long,
    Fixed LS/SL/EW-Long). ``mode='beta'`` -> 1 LLM (beta-hedged) + 4 beta baselines
    (Timeline BH LS/SL, Fixed BH LS/SL). ``main_sim`` carries the LLM run's normalized
    equity; ``main_records`` drives the timeline schedule. Returns (rows, equity_by_name).
    """
    if mode not in ("nobeta", "beta"):
        raise ValueError(f"mode must be 'nobeta' or 'beta', got {mode!r}")
    timeline = _schedule_from_records(main_records)
    fixed_pair = select_coint_pair(pool_df, symbols, dates[0])
    fixed_for = lambda i: fixed_pair
    fp = f"({fixed_pair[0]}/{fixed_pair[1]})"
    if mode == "nobeta":
        specs = [
            (1, main_label, None, None, False, main_sim),
            (2, "Timeline LONG_SHORT", timeline, LONG_SHORT, False, None),
            (3, "Timeline SHORT_LONG", timeline, SHORT_LONG, False, None),
            (4, "Timeline Equal-Weight Long", timeline, EQUAL_WEIGHT_LONG, False, None),
            (5, f"10Y-Coint Fixed LONG_SHORT {fp}", fixed_for, LONG_SHORT, False, None),
            (6, f"10Y-Coint Fixed SHORT_LONG {fp}", fixed_for, SHORT_LONG, False, None),
            (7, f"10Y-Coint Fixed Equal-Weight Long {fp}", fixed_for, EQUAL_WEIGHT_LONG, False, None),
        ]
    elif mode == "beta":
        specs = [
            (1, main_label, None, None, True, main_sim),
            (2, "Timeline Beta-Hedged LONG_SHORT", timeline, LONG_SHORT, True, None),
            (3, "Timeline Beta-Hedged SHORT_LONG", timeline, SHORT_LONG, True, None),
            (4, f"10Y-Coint Fixed Beta-Hedged LONG_SHORT {fp}", fixed_for, LONG_SHORT, True, None),
            (5, f"10Y-Coint Fixed Beta-Hedged SHORT_LONG {fp}", fixed_for, SHORT_LONG, True, None),
        ]
    else:
        raise ValueError(f"mode must be 'nobeta' or 'beta', got {mode!r}")

    rows, equity = [], {}
    for num, name, sched, action, bh, pre in specs:
        if pre is not None:
            sim = pre
        else:
            sim = simulate_baseline_window(
                dates=dates, env_data=env_data, symbols=symbols, pair_for_index=sched,
                action=action, beta_hedged=bh, lookback=lookback, history_seed=history_seed,
                settlement_date=settlement_date)
        row = metrics_row(f"{num}. {name}", sim)
        row["group"] = "main" if num == 1 else "baseline"
        rows.append(row)
        equity[f"{num}. {name}"] = sim["normalized_equity"]
    return rows, equity
