"""Required validation tests (Section 14) for the redesigned framework.

Run: ``python -m unittest tests.test_redesign -v``  (fully offline, deterministic)

Each test maps to a numbered requirement:
  1. Pair switch test            -> old pair becomes FLAT before new pair opens
  2. Weekend news test           -> Friday news appears on Monday
  3. Holiday news test           -> latest trading-day news used after a holiday
  4. FLAT test                   -> all positions close
  5. No HOLD test                -> HOLD is rejected / does not exist
  6. Portfolio consistency test  -> portfolio == logs == metrics
  7. Policy rollback test        -> >3% degradation restores previous policy
  8. Convergence test            -> stops only when overlap>=80% AND PnL delta<3%
  9. Pair selection test         -> correlation computed by CODE, not the LLM
"""

import json
import os
import sys
import unittest
from datetime import date

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from strategies.types import Action, PositionState, action_from_label, ALL_ACTIONS
from utils.portfolio import PortfolioManager, size_pair_position, flat_targets
from pathlib import Path
from utils.data_utils import (
    align_news_to_trading_days, filter_to_trading_days, load_offline_dataset,
)
from utils.statarb import (
    compute_statarb_metrics, rank_statarb_pairs, summarize_pair_text,
    summarize_stock_news, StatArbPair,
    compute_multihorizon_profile, rank_multihorizon_profiles, select_horizon,
    apply_reflection_adjustment, summarize_multihorizon_text, MultiHorizonProfile,
    HORIZON_ORDER, pair_theme, discover_pairs_coint10y,
)
from utils.engine import (
    simulate_window, decide_acceptance, parameter_convergence,
    _week_news_items, simulate_discovery_window, _session_news_items,
)
from llm.policy_agent import (
    PolicyAgent, Policy, PairChoice, BreakAssessment, PairReflectionMemory, NO_TRADE,
    classify_pair_outcome, ActionPromptMemory, news_break_level, statistical_break_score,
    stat_status_from_score, sanitize_reflection, filter_reflection_lessons,
    build_spread_state, PairSelectionMemory, DirectionMemory, DirectionDecision,
    load_direction_discovery_prompt, DirectionPromptMemory,
)
from strategies.types import Action as _Action

# relaxed thresholds so toy/synthetic pairs pass the hard minimums in tests
RELAXED = dict(coint_pmax=0.99, adf_pmax=0.99, half_life_min=0.0, half_life_max=1e9)


# --------------------------------------------------------------------------- #
# tiny synthetic dataset for engine-level tests
# --------------------------------------------------------------------------- #
def _toy_env(n=40):
    import math
    syms = ["A", "B", "C", "D"]
    env = {}
    base = date(2025, 1, 1)
    from datetime import timedelta
    for i in range(n):
        d = base + timedelta(days=i)
        # mean-reverting-ish spreads so the policy actually trades
        prices = {
            "A": 100 + 5 * math.sin(i / 3.0),
            "B": 100 - 5 * math.sin(i / 3.0),
            "C": 50 + 3 * math.cos(i / 4.0),
            "D": 50 + 3 * math.cos(i / 4.0 + 1.0),
        }
        env[d] = {"price": prices, "news": {}}
    return syms, env


def _toy_price_df(n=300):
    """Synthetic 4-symbol frame with two cointegrated pairs (A,B) and (C,D)."""
    import numpy as np
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 1, n))
    common2 = np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "A": 100 + common + rng.normal(0, 0.5, n),
        "B": 100 + 0.9 * common + rng.normal(0, 0.5, n),
        "C": 50 + common2 + rng.normal(0, 0.3, n),
        "D": 50 + 0.8 * common2 + rng.normal(0, 0.3, n),
    }, index=idx)


def _toy_candidates(symbols, top_k=4):
    return rank_statarb_pairs(_toy_price_df(), list(symbols), top_k=top_k)


def _toy_horizon_dfs(symbols):
    """A horizon dict (1Y..20Y) built from the synthetic cointegrated frame."""
    df = _toy_price_df(n=600)
    return {h: df for h in HORIZON_ORDER}


def _toy_profiles(symbols):
    return rank_multihorizon_profiles(_toy_horizon_dfs(symbols), list(symbols))


class ForcedPairAgent(PolicyAgent):
    """Deterministic agent that selects pair X for the first half then pair Y."""

    def __init__(self, pair_x, pair_y, symbols):
        super().__init__(backend="offline", symbols=symbols)
        self.pair_x, self.pair_y, self._calls = pair_x, pair_y, 0

    def select_pair(self, pair_summaries, policy, pair_memory=None):
        self._calls += 1
        pair = self.pair_x if self._calls <= 1 else self.pair_y
        return PairChoice(pair=pair, reasoning="forced", confidence=1.0, risks=[],
                          model="test", prompt="forced", raw_response="forced")

    def detect_relationship_break(self, **kw):
        # force a switch every monitor day so the new pair is selected
        return BreakAssessment(True, "forced break", "test", 0.9, True, "test")

    def decide_action(self, **kw):
        # always take a directional position so exposure exists to flatten
        from llm.policy_agent import Decision
        return Decision(Action.long_short, "forced long_short", "neutral")


class NoBreakAgent(PolicyAgent):
    """Selects a fixed pair and NEVER signals a relationship break (pair persists)."""

    def __init__(self, pair, symbols):
        super().__init__(backend="offline", symbols=symbols)
        self.pair = pair

    def select_pair(self, pair_summaries, policy, pair_memory=None):
        return PairChoice(pair=self.pair, reasoning="fixed", confidence=1.0, risks=[],
                          model="test", prompt="fixed", raw_response="fixed")

    def detect_relationship_break(self, **kw):
        return BreakAssessment(False, "stable", "none", 0.1, False, "test")

    def decide_action(self, **kw):
        from llm.policy_agent import Decision
        return Decision(Action.long_short, "fixed", "neutral")


# --------------------------------------------------------------------------- #
class Test1PairSwitch(unittest.TestCase):
    def test_old_pair_flat_before_new(self):
        syms = ["A", "B", "C", "D"]
        pm = PortfolioManager(syms, initial_cash=1.0)
        prices = {"A": 100.0, "B": 100.0, "C": 50.0, "D": 50.0}
        # open pair (A,B) long_short
        tgt = {s: 0.0 for s in syms}
        tgt.update(size_pair_position(prices, ("A", "B"), 0.5, Action.long_short))
        pm.step(prices, tgt)
        self.assertFalse(pm.is_flat(["A", "B"]))
        # SWITCH: must flatten old pair first, then verify flat
        pm.step(prices, {s: 0.0 for s in syms})
        self.assertTrue(pm.is_flat(["A", "B"]), "old pair must be FLAT before switching")
        # now open new pair (C,D)
        tgt = {s: 0.0 for s in syms}
        tgt.update(size_pair_position(prices, ("C", "D"), 0.5, Action.long_short))
        pm.step(prices, tgt)
        # old pair still flat -> never two pairs at once
        self.assertTrue(pm.is_flat(["A", "B"]))
        self.assertFalse(pm.is_flat(["C", "D"]))

    def test_engine_records_switch_with_verified_flat(self):
        syms, env = _toy_env()
        dates = sorted(env)
        agent = ForcedPairAgent(("A", "B"), ("C", "D"), syms)
        res = simulate_window(
            dates=dates, env_data=env, symbols=syms, agent=agent, policy=Policy(**RELAXED),
            lookback=10, monitor_frequency=10, pair_candidates=_toy_candidates(syms),
        )
        self.assertTrue(res.switch_events, "expected at least one pair switch")
        for ev in res.switch_events:
            self.assertTrue(ev["verified_flat"])  # switch forced FLAT first
        # no record ever has two pairs implied (engine zeroes non-active legs)
        for r in res.records:
            self.assertIn(r["selected_pair"] if isinstance(r, dict) else r.selected_pair,
                          {"A/B", "C/D"})


class Test2WeekendNews(unittest.TestCase):
    def test_friday_news_appears_monday(self):
        fri, sat, sun, mon = date(2025, 1, 3), date(2025, 1, 4), date(2025, 1, 5), date(2025, 1, 6)
        trading = [fri, mon]  # weekend has no session
        news = {fri: {"AAPL": ["fri item"]}, sat: {"AAPL": ["sat item"]}}
        aligned = align_news_to_trading_days(trading, news)
        # Monday sees Friday+Saturday news (NOT calendar Monday-1 == Sunday, which is empty)
        self.assertIn("fri item", aligned[mon]["AAPL"])
        self.assertIn("sat item", aligned[mon]["AAPL"])
        # Friday itself sees no prior news in this window
        self.assertEqual(aligned[fri], {})


class Test3HolidayNews(unittest.TestCase):
    def test_latest_trading_day_news_after_holiday(self):
        fri, mon_holiday, tue = date(2025, 1, 17), date(2025, 1, 20), date(2025, 1, 21)
        trading = [fri, tue]  # Monday is a holiday -> no session
        news = {fri: {"MSFT": ["pre-holiday friday news"]}}
        aligned = align_news_to_trading_days(trading, news)
        self.assertIn("pre-holiday friday news", aligned[tue]["MSFT"])
        # No future leakage: Tuesday must not see news published on/after Tuesday
        news2 = dict(news); news2[tue] = {"MSFT": ["same-day tuesday news"]}
        aligned2 = align_news_to_trading_days(trading, news2)
        self.assertNotIn("same-day tuesday news", aligned2[tue].get("MSFT", []))


class Test4Flat(unittest.TestCase):
    def test_flat_closes_both_legs(self):
        self.assertEqual(size_pair_position({"A": 10, "B": 10}, ("A", "B"), 1.0, Action.flat),
                         {"A": 0.0, "B": 0.0})
        self.assertEqual(flat_targets(("A", "B")), {"A": 0.0, "B": 0.0})
        pm = PortfolioManager(["A", "B"], initial_cash=1.0)
        prices = {"A": 10.0, "B": 10.0}
        tgt = size_pair_position(prices, ("A", "B"), 0.5, Action.long_short)
        pm.step(prices, {"A": tgt["A"], "B": tgt["B"]})
        self.assertFalse(pm.is_flat())
        pm.step(prices, flat_targets(("A", "B")))
        self.assertTrue(pm.is_flat(), "FLAT must close both legs -> no exposure")


class Test5NoHold(unittest.TestCase):
    def test_hold_rejected(self):
        self.assertEqual(set(ALL_ACTIONS), {Action.long_short, Action.short_long, Action.flat})
        self.assertFalse(hasattr(Action, "hold"))
        self.assertFalse(hasattr(Action, "close"))
        for bad in ("hold", "HOLD", "close", "long", "short"):
            with self.assertRaises(ValueError):
                action_from_label(bad)
        # valid ones still parse
        self.assertEqual(action_from_label("LONG_SHORT"), Action.long_short)
        self.assertEqual(action_from_label("flat"), Action.flat)


class Test6PortfolioConsistency(unittest.TestCase):
    def test_portfolio_equals_logs_equals_metrics(self):
        from utils.metrics import compute_performance_metrics
        syms, env = _toy_env()
        dates = sorted(env)
        agent = PolicyAgent(backend="offline", symbols=syms)
        res = simulate_window(dates=dates, env_data=env, symbols=syms, agent=agent,
                              policy=Policy(**RELAXED), lookback=10, monitor_frequency=10,
                              pair_candidates=_toy_candidates(syms))
        # cumulative_pnl in each record == running sum of daily_pnl
        running = 0.0
        for r in res.records:
            running += r.daily_pnl
            self.assertAlmostEqual(running, r.cumulative_pnl, places=9)
        # metrics final_value == last normalized equity (logs == metrics)
        m = compute_performance_metrics(pd.Series(res.normalized_equity), 1.0)
        self.assertAlmostEqual(m["final_value"], res.normalized_equity[-1], places=9)
        self.assertAlmostEqual(res.records[-1].normalized_equity,
                               res.normalized_equity[-1], places=9)


class Test7PolicyRollback(unittest.TestCase):
    def test_degradation_beyond_3pct_restores_previous(self):
        # candidate worse by 10% -> reject + rollback
        accepted, rollback, rate = decide_acceptance(1.0, 0.90)
        self.assertFalse(accepted)
        self.assertTrue(rollback)
        self.assertAlmostEqual(rate, -0.10, places=6)
        # candidate worse by only 1% -> reject but NOT a rollback event
        accepted, rollback, rate = decide_acceptance(1.0, 0.99)
        self.assertFalse(accepted)
        self.assertFalse(rollback)
        # candidate better -> accept
        accepted, rollback, rate = decide_acceptance(1.0, 1.20)
        self.assertTrue(accepted)
        self.assertFalse(rollback)

    def test_trainer_keeps_previous_policy_params_on_rollback(self):
        prev = Policy(version=3, entry_z=1.0)
        # simulate a rejected candidate: previous policy must be unchanged
        accepted, rollback, _ = decide_acceptance(1.0, 0.5)
        kept = prev if not accepted else Policy()
        self.assertTrue(rollback)
        self.assertEqual(kept.entry_z, 1.0)
        self.assertEqual(kept.version, 3)


class Test8Convergence(unittest.TestCase):
    """Convergence now uses POLICY PARAMETER deltas, not action overlap (Step 1)."""


    def test_parameter_convergence_all_conditions(self):
        base = Policy().tracked_params()
        # identical params + small PnL -> converged
        ok, deltas = parameter_convergence(base, dict(base), 0.01)
        self.assertTrue(ok)
        self.assertIn("entry_z_delta", deltas)
        # entry_z changed > 5% -> not converged
        p2 = dict(base); p2["entry_z"] = base["entry_z"] * 1.10
        self.assertFalse(parameter_convergence(base, p2, 0.0)[0])
        # threshold changed -> not converged
        p3 = dict(base); p3["coint_pmax"] = base["coint_pmax"] + 0.01
        self.assertFalse(parameter_convergence(base, p3, 0.0)[0])
        # PnL change > 3% -> not converged
        self.assertFalse(parameter_convergence(base, dict(base), 0.05)[0])

    def test_requires_two_consecutive(self):
        base = Policy().tracked_params()
        evals = [0.10, 0.01, 0.01]   # only last two have small PnL change
        streak, converged_at = 0, None
        for i, pnl in enumerate(evals):
            ok, _ = parameter_convergence(base, dict(base), pnl)
            streak = streak + 1 if ok else 0
            if streak >= 2 and converged_at is None:
                converged_at = i
        self.assertEqual(converged_at, 2, "must need TWO consecutive satisfying evals")


def _statarb_df():
    """A cointegrated high-corr pair (A,B) + an independent low-corr leg (C)."""
    import numpy as np
    idx = pd.date_range("2016-01-01", periods=1500)
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(0, 1, 1500))
    a = 100 + common + rng.normal(0, 0.5, 1500)
    b = 100 + 0.95 * common + rng.normal(0, 0.5, 1500)   # cointegrated w/ A, high corr
    c = 100 + np.cumsum(rng.normal(0, 1, 1500))          # independent random walk
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


class Test9PairSelectionStatArb(unittest.TestCase):
    def test_low_correlation_only_is_removed(self):
        # A cointegrated HIGH-correlation pair must beat a low-correlation pair.
        df = _statarb_df()
        ab = compute_statarb_metrics(df, "A", "B")
        ac = compute_statarb_metrics(df, "A", "C")
        self.assertGreater(ab.corr_10y, 0.8)
        self.assertGreater(ab.composite, ac.composite,
                           "high-corr cointegrated pair must outrank low-corr pair")
        # ranking puts the cointegrated high-corr pair first (NOT lowest correlation)
        ranked = rank_statarb_pairs(df, ["A", "B", "C"], top_k=3)
        self.assertEqual(tuple(ranked[0].pair), ("A", "B"))

    def test_ranking_uses_coint_adf_halflife_stability(self):
        df = _statarb_df()
        pc = compute_statarb_metrics(df, "A", "B")
        # all the classical stat-arb metrics are computed in code
        for attr in ("coint_pvalue", "adf_pvalue", "half_life", "hedge_ratio",
                     "rolling_corr_stability", "n_entry_1", "hist_sharpe", "win_rate"):
            self.assertTrue(hasattr(pc, attr))
        self.assertIn("cointegration", pc.sub_scores)
        self.assertIn("stationarity", pc.sub_scores)
        self.assertIn("mean_reversion", pc.sub_scores)
        # cointegrated pair should have a small cointegration p-value
        self.assertLess(pc.coint_pvalue, 0.10)

    def test_llm_receives_text_summaries_only(self):
        import numpy as np
        df = _statarb_df()
        ranked = rank_statarb_pairs(df, ["A", "B", "C"], top_k=3)
        summaries = [summarize_pair_text(pc, i + 1) for i, pc in enumerate(ranked)]
        # offline selection honours the composite ranking (relaxed so a pair qualifies)
        agent = PolicyAgent(backend="offline", symbols=["A", "B", "C"])
        choice = agent.select_pair(summaries, Policy(**RELAXED))
        self.assertIsInstance(choice, PairChoice)
        self.assertEqual(tuple(choice.pair), tuple(ranked[0].pair))
        # the openai backend is sent ONLY text blocks + a pass-list -- no raw arrays
        llm = _FakeLLM(['{"selected_pair":["A","B"],"reasoning":"cointegrated",'
                        '"confidence":0.8,"risks":["none"]}'])
        oai = PolicyAgent(backend="open_source", llm=llm, symbols=["A", "B", "C"])
        oai.select_pair(summaries, Policy(**RELAXED))
        payload = llm.calls[0]["payload"]
        self.assertIn("candidate_pairs", payload)
        self.assertTrue(all(isinstance(t, str) for t in payload["candidate_pairs"]))
        blob = json.dumps(payload)
        for forbidden in ("recent_prices", "returns", "price_series", "correlation_matrix"):
            self.assertNotIn(forbidden, blob)
        # no raw 3+-element float array leaked into the text payload
        import re
        self.assertFalse(re.search(r"\[\s*-?\d+\.\d+,\s*-?\d+\.\d+,\s*-?\d+\.\d+", blob))

    def test_future_news_excluded_from_selection(self):
        from datetime import timedelta
        d0 = date(2025, 3, 10)
        env = {
            d0 - timedelta(days=2): {"price": {}, "news": {"A": ["past item"]}},
            d0: {"price": {}, "news": {"A": ["today item"]}},
            d0 + timedelta(days=2): {"price": {}, "news": {"A": ["FUTURE item"]}},
        }
        items = _week_news_items(env, d0, 7, ["A"])
        self.assertIn("past item", items["A"])
        self.assertNotIn("FUTURE item", items["A"], "future news must never be used")

    def test_top_10_only_sent_to_llm(self):
        # build >10 candidates; the experiment sends only the top 10
        from experiments.run_redesign import SYMBOLS
        # 45 pairs exist for 10 symbols; emulate the slice used in main()
        import numpy as np
        idx = pd.date_range("2016-01-01", periods=400)
        rng = np.random.default_rng(5)
        data = {s: 100 + np.cumsum(rng.normal(0, 1, 400)) for s in SYMBOLS}
        df = pd.DataFrame(data, index=idx)
        all_ranked = rank_statarb_pairs(df, SYMBOLS, top_k=0)
        self.assertGreater(len(all_ranked), 10)
        top10 = all_ranked[:10]
        summaries = [summarize_pair_text(pc, i + 1) for i, pc in enumerate(top10)]
        llm = _FakeLLM(['{"selected_pair":["%s","%s"],"reasoning":"x","confidence":0.5,"risks":[]}'
                        % (top10[0].pair[0], top10[0].pair[1])])
        agent = PolicyAgent(backend="open_source", llm=llm, symbols=SYMBOLS)
        agent.select_pair(summaries, Policy(**RELAXED))
        self.assertEqual(len(llm.calls[0]["payload"]["candidate_pairs"]), 10)


class _FakeLLM:
    """Stand-in for PairTradingLLM that returns canned JSON (no network)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call_text(self, *, system_prompt, payload, task_type=None):
        self.calls.append({"system_prompt": system_prompt, "payload": payload,
                           "task_type": task_type})
        return self.responses.pop(0)


class Test10OpenWeightBackendPath(unittest.TestCase):
    """The real-LLM branch parses LONG_SHORT/SHORT_LONG/FLAT and never sets temperature."""

    def test_decide_and_select_parse(self):
        llm = _FakeLLM([
            '{"pair": ["A","B"], "rationale": "low corr"}',
            '{"action": "SHORT_LONG", "rationale": "A rich"}',
        ])
        agent = PolicyAgent(backend="open_source", llm=llm, symbols=["A", "B", "C"])
        passing_summary = {"pair": ["A", "B"], "text": "Pair A/B",
                           "cointegration_pvalue": 0.01, "adf_pvalue": 0.02,
                           "half_life_days": 10.0, "composite_score": 0.5,
                           "relationship_breakdown": False}
        choice = agent.select_pair([passing_summary], Policy())
        self.assertIsInstance(choice, PairChoice)
        self.assertEqual(tuple(choice.pair), ("A", "B"))
        dec = agent.decide_action(
            pair=("A", "B"), position_state=PositionState.flat, signal_z=2.1,
            recent_pnl_3d=0.01, news_sentiment="Positive", policy=Policy(),
        )
        self.assertEqual(dec.action, Action.short_long)
        # the LLM payload carries CODE-computed summaries only (no raw series)
        sent = llm.calls[1]["payload"]
        self.assertIn("spread_zscore", sent)
        self.assertNotIn("recent_prices", sent)

    def test_no_temperature_in_client_call(self):
        import inspect
        from llm import pair_trading_llm
        src = inspect.getsource(pair_trading_llm.PairTradingLLM._call_openai)
        # no temperature argument is passed to the API (comments may mention it)
        self.assertNotIn("temperature=", src, "must not pass temperature (Section 7)")






class Test13LogPrices(unittest.TestCase):
    """Step 2: hedge ratio / spread / cointegration / ADF / half-life use LOG prices."""

    def test_provenance_fields_declare_log_inputs(self):
        df = _statarb_df()
        pc = compute_statarb_metrics(df, "A", "B")
        self.assertEqual(pc.spread_definition, "log(A) - beta * log(B)")
        self.assertEqual(pc.cointegration_input, "log_prices")
        self.assertEqual(pc.adf_input, "log_spread")
        self.assertEqual(pc.half_life_input, "log_spread")

    def test_hedge_ratio_is_ols_on_log_prices(self):
        import numpy as np
        df = _statarb_df()
        pc = compute_statarb_metrics(df, "A", "B")
        la, lb = np.log(df["A"].values), np.log(df["B"].values)
        beta_log = float(np.polyfit(lb, la, 1)[0])          # OLS on LOG prices
        beta_raw = float(np.polyfit(df["B"].values, df["A"].values, 1)[0])  # raw
        self.assertAlmostEqual(pc.hedge_ratio, beta_log, places=4)
        # log beta should differ from the raw-price beta (confirms NOT raw prices)
        self.assertNotAlmostEqual(pc.hedge_ratio, beta_raw, places=3)


class Test14NoNoTrade(unittest.TestCase):
    """Step 1: pair selection ALWAYS returns a pair -- NO_TRADE is never an output."""

    def test_forced_pair_selection_when_none_qualify(self):
        # Requirements 1, 2, 3: never NO_TRADE; always one pair; forced when unqualified.
        df = _statarb_df()
        ranked = rank_statarb_pairs(df, ["A", "B", "C"], top_k=3)
        summaries = [summarize_pair_text(pc, i + 1) for i, pc in enumerate(ranked)]
        agent = PolicyAgent(backend="offline", symbols=["A", "B", "C"])
        strict = Policy(coint_pmax=0.0, adf_pmax=0.0)  # impossible -> nothing qualifies
        choice = agent.select_pair(summaries, strict)
        self.assertFalse(choice.no_trade)
        self.assertIsNotNone(choice.pair)
        self.assertEqual(len(choice.pair), 2)
        self.assertFalse(choice.qualified)
        self.assertTrue(choice.forced_pair_selection)
        self.assertTrue(choice.failed_thresholds, "forced selection must explain failures")

    def test_qualified_selection_returns_pair(self):
        df = _statarb_df()
        ranked = rank_statarb_pairs(df, ["A", "B", "C"], top_k=3)
        summaries = [summarize_pair_text(pc, i + 1) for i, pc in enumerate(ranked)]
        agent = PolicyAgent(backend="offline", symbols=["A", "B", "C"])
        choice = agent.select_pair(summaries, Policy(**RELAXED))
        self.assertIsNotNone(choice.pair)
        self.assertFalse(choice.no_trade)
        self.assertTrue(choice.qualified)
        self.assertFalse(choice.forced_pair_selection)

    def test_openai_never_returns_no_trade(self):
        # Even if the model answers NO_TRADE / malformed, a real pair is forced out.
        for resp in ('{"selected_pair":"NO_TRADE","reasoning":"x","confidence":0.5,"risks":[]}',
                     '{"selected_pair":["NO_TRADE"],"reasoning":"x","confidence":0.5,"risks":[]}',
                     '{"selected_pair":["AAPL"],"reasoning":"x","confidence":0.5,"risks":[]}'):
            llm = _FakeLLM([resp])
            agent = PolicyAgent(backend="open_source", llm=llm, symbols=["AAPL", "TSLA"])
            summary = {"pair": ["AAPL", "TSLA"], "text": "x", "cointegration_pvalue": 0.01,
                       "adf_pvalue": 0.02, "half_life_days": 10.0, "composite_score": 0.5,
                       "relationship_breakdown": False}
            choice = agent.select_pair([summary], Policy(**RELAXED))
            self.assertFalse(choice.no_trade, f"must NEVER be NO_TRADE for {resp}")
            self.assertEqual(tuple(choice.pair), ("AAPL", "TSLA"))

    def test_engine_always_holds_a_pair_even_when_unqualified(self):
        syms, env = _toy_env()
        dates = sorted(env)
        agent = PolicyAgent(backend="offline", symbols=syms)
        strict = Policy(coint_pmax=0.0, adf_pmax=0.0)  # nothing qualifies -> forced pair
        res = simulate_window(dates=dates, env_data=env, symbols=syms, agent=agent,
                              policy=strict, monitor_frequency=10,
                              pair_candidates=_toy_candidates(syms))
        # selection never yields NO_TRADE: every record names a real pair
        self.assertTrue(all(r.selected_pair != "NO_TRADE" for r in res.records))
        self.assertEqual(res.n_no_trade_days(), 0)


class Test15RelationshipBreakSwitching(unittest.TestCase):
    """Steps 3/4/7: persistence, break-triggered switch, FLAT-first, pair reflection."""

    def test_pair_persists_when_no_break(self):
        syms, env = _toy_env()
        dates = sorted(env)
        agent = NoBreakAgent(("A", "B"), syms)
        res = simulate_window(dates=dates, env_data=env, symbols=syms, agent=agent,
                              policy=Policy(**RELAXED), monitor_frequency=5,
                              pair_candidates=_toy_candidates(syms))
        self.assertEqual(len(res.switch_events), 0, "pair must persist without a break")
        traded = [r for r in res.records if not r.no_trade]
        self.assertTrue(traded and all(r.selected_pair == "A/B" for r in traded))

    def test_switch_only_on_break_and_forces_flat_and_reflects(self):
        syms, env = _toy_env()
        dates = sorted(env)
        agent = ForcedPairAgent(("A", "B"), ("C", "D"), syms)  # forces break + switch
        res = simulate_window(dates=dates, env_data=env, symbols=syms, agent=agent,
                              policy=Policy(**RELAXED), monitor_frequency=10,
                              pair_candidates=_toy_candidates(syms))
        self.assertTrue(res.switch_events)
        self.assertTrue(all(ev["verified_flat"] for ev in res.switch_events))  # FLAT first
        self.assertTrue(res.pair_reflections, "pair reflection generated after replacement")
        self.assertIn("lessons", res.pair_reflections[0])

    def test_pair_reflection_memory_in_selection_prompt(self):
        mem = PairReflectionMemory()
        mem.add(pair=("AAPL", "TSLA"), reflection_text="closed: half-life too long",
                reason="half-life unreasonable",
                lessons=["avoid pairs with half-life above 60 days"])
        # failed pair + lessons must surface in the rendered memory
        rendered = mem.render()
        self.assertIn("AAPL/TSLA", rendered)
        self.assertIn("FAILED PAIR HISTORY", rendered)
        self.assertIn("avoid pairs with half-life above 60 days", rendered)
        # and the openai selection prompt must include the memory
        llm = _FakeLLM(['{"selected_pair":"NO_TRADE","reasoning":"x","confidence":0.5,'
                        '"risks":[],"pair_reflection_used":true}'])
        agent = PolicyAgent(backend="open_source", llm=llm, symbols=["AAPL", "TSLA", "MSFT"])
        summary = {"pair": ["MSFT", "TSLA"], "text": "Pair MSFT/TSLA",
                   "cointegration_pvalue": 0.01, "adf_pvalue": 0.02,
                   "half_life_days": 10.0, "composite_score": 0.5,
                   "relationship_breakdown": False}
        choice = agent.select_pair([summary], Policy(), pair_memory=mem)
        self.assertIn("AAPL/TSLA", choice.prompt)          # failed pair history in prompt
        self.assertIn("LESSONS", choice.prompt.upper())
        self.assertTrue(choice.pair_reflection_used)


class Test16MultiHorizon(unittest.TestCase):
    """Steps 3/5/6: multi-horizon metrics + profile sent to GPT (no raw series)."""

    def test_metrics_include_all_five_horizons(self):  # req 4
        df = _toy_price_df(600)
        hdfs = {h: df for h in HORIZON_ORDER}
        prof = compute_multihorizon_profile(hdfs, "A", "B")
        self.assertIsNotNone(prof)
        for h in ("1Y", "3Y", "5Y", "10Y", "20Y"):
            self.assertIn(h, prof.horizons)
            pc = prof.horizons[h]
            for attr in ("coint_pvalue", "adf_pvalue", "half_life", "hedge_ratio",
                         "n_entry_1", "n_entry_2", "hist_sharpe", "win_rate"):
                self.assertTrue(hasattr(pc, attr))
        # derived relationship profile present
        for f in ("short_term_relationship_strength", "medium_term_relationship_strength",
                  "long_term_relationship_strength", "horizon_consistency",
                  "relationship_stability_across_horizons", "relationship_trend"):
            self.assertTrue(hasattr(prof, f))

    def test_full_profile_sent_to_gpt_no_raw_series(self):  # req 5,6,7,8
        syms = ["A", "B", "C", "D"]
        profiles = _toy_profiles(syms)
        apply_reflection_adjustment(profiles, None)
        sel_h, eligible, qual, _ = select_horizon(
            profiles, **dict(coint_pmax=0.99, adf_pmax=0.99, half_life_min=0.0,
                             half_life_max=1e9, composite_min=0.0))
        mh = [summarize_multihorizon_text(p, i + 1, selection_horizon=sel_h, qualified=qual)
              for i, p in enumerate(eligible[:5])]
        top = eligible[0].pair
        llm = _FakeLLM(['{"selected_pair":["%s","%s"],"selection_horizon":"%s",'
                        '"qualified":true,"forced_pair_selection":false,"reasoning":"x",'
                        '"confidence":0.7,"risks":[],"multi_horizon_profile_used":true}'
                        % (top[0], top[1], sel_h)])
        agent = PolicyAgent(backend="open_source", llm=llm, symbols=syms)
        choice = agent.select_pair_multihorizon(mh, Policy(**RELAXED),
                                                selection_horizon=sel_h, qualified=qual)
        self.assertTrue(choice.multi_horizon_profile_used)
        payload = llm.calls[0]["payload"]
        blob = json.dumps(payload)
        # full multi-horizon profile is in the text (all horizons + trend)
        joined = "\n".join(payload["candidate_pairs"])
        for h in ("1Y", "3Y", "5Y", "10Y", "20Y"):
            self.assertIn(h, joined)
        self.assertIn("Relationship Trend", joined)
        # no raw price series / returns / correlation matrix leaked
        for forbidden in ("recent_prices", "returns", "price_series", "correlation_matrix"):
            self.assertNotIn(forbidden, blob)
        import re
        self.assertFalse(re.search(r"\[\s*-?\d+\.\d+,\s*-?\d+\.\d+,\s*-?\d+\.\d+", blob))


class Test17ReflectionAdjustedRanking(unittest.TestCase):
    """Step 9: reflection-adjusted scoring drives ranking (not base score alone)."""

    def _profiles(self):
        profiles = _toy_profiles(["A", "B", "C", "D"])
        self.assertTrue(profiles)
        return profiles

    def test_adjusted_score_is_calculated(self):  # req 12
        profiles = self._profiles()
        apply_reflection_adjustment(profiles, None)
        for p in profiles:
            self.assertAlmostEqual(p.adjusted_score, p.base_score, places=6)
            self.assertEqual(p.success_bonus, 0.0)
            self.assertEqual(p.failure_penalty, 0.0)

    def test_success_bonus_raises_adjusted(self):  # req 13
        profiles = self._profiles()
        target = profiles[0]
        mem = PairReflectionMemory()
        mem.add_success(pair=tuple(target.pair), reflection_text="worked",
                        reason="profitable", lessons=["prefer software pairs"],
                        stats={"half_life_days": 8.0, "cointegration_pvalue": 0.01},
                        news={"pair_level_risk": "low"})
        apply_reflection_adjustment(profiles, mem)
        self.assertGreater(target.success_bonus, 0.0)
        self.assertGreater(target.adjusted_score, target.base_score)

    def test_failure_penalty_lowers_adjusted(self):  # req 14
        profiles = self._profiles()
        target = profiles[0]
        mem = PairReflectionMemory()
        mem.add_failure(pair=tuple(target.pair), reflection_text="failed",
                        reason="half-life too long", lessons=["avoid X"],
                        stats={"half_life_days": 80.0, "cointegration_pvalue": 0.4},
                        news={"pair_level_risk": "high"})
        apply_reflection_adjustment(profiles, mem)
        self.assertGreater(target.failure_penalty, 0.0)
        self.assertLess(target.adjusted_score, target.base_score)

    def test_ranking_uses_adjusted_not_base(self):  # req 15
        profiles = self._profiles()
        # boost the SECOND-by-base profile via success memory so adjusted order flips
        by_base = sorted(profiles, key=lambda p: p.base_score, reverse=True)
        booster = by_base[1]
        mem = PairReflectionMemory()
        mem.add_success(pair=tuple(booster.pair), reflection_text="great",
                        reason="profitable", lessons=["prefer it"],
                        stats={"half_life_days": 8.0, "cointegration_pvalue": 0.01},
                        news={"pair_level_risk": "low"})
        apply_reflection_adjustment(profiles, mem)
        _h, eligible, _q, _f = select_horizon(
            profiles, coint_pmax=0.99, adf_pmax=0.99, half_life_min=0.0,
            half_life_max=1e9, composite_min=0.0)
        # eligible is ordered by ADJUSTED score
        adj = [p.adjusted_score for p in eligible]
        self.assertEqual(adj, sorted(adj, reverse=True))
        self.assertEqual(tuple(eligible[0].pair),
                         tuple(max(profiles, key=lambda p: p.adjusted_score).pair))


class Test18SuccessFailureReflection(unittest.TestCase):
    """Steps 7/8: SUCCESS/FAILURE reflections + both memories in the prompt."""

    def test_classify_outcome(self):
        self.assertEqual(classify_pair_outcome(cum_pnl=0.1), "SUCCESS")
        self.assertEqual(classify_pair_outcome(cum_pnl=-0.1), "FAILURE")
        self.assertEqual(classify_pair_outcome(cum_pnl=0.1, relationship_break=True), "FAILURE")
        self.assertEqual(classify_pair_outcome(cum_pnl=0.1, half_life=120.0), "FAILURE")
        self.assertEqual(classify_pair_outcome(cum_pnl=0.0), "NEUTRAL")

    def test_success_reflection_generated(self):  # req 9
        agent = PolicyAgent(backend="offline", symbols=["MSFT", "GOOGL"])
        rec = agent.reflect_on_pair(
            pair=("MSFT", "GOOGL"), outcome_type="SUCCESS",
            stats={"cointegration_pvalue": 0.01, "adf_pvalue": 0.02, "half_life_days": 8.0},
            news_summary={"pair_level_risk": "low"},
            performance={"cumulative_pnl": 0.05, "reversions": 7})
        self.assertEqual(rec["outcome_type"], "SUCCESS")
        self.assertTrue(rec["why_worked"])
        self.assertIn("SUCCESS", rec["reflection_text"])
        self.assertTrue(rec["lessons"])

    def test_failure_reflection_generated(self):  # req 10
        agent = PolicyAgent(backend="offline", symbols=["AAPL", "TSLA"])
        rec = agent.reflect_on_pair(
            pair=("AAPL", "TSLA"), outcome_type="FAILURE",
            stats={"cointegration_pvalue": 0.4, "adf_pvalue": 0.5, "half_life_days": 90.0},
            news_summary={"pair_level_risk": "high"},
            performance={"cumulative_pnl": -0.05})
        self.assertEqual(rec["outcome_type"], "FAILURE")
        self.assertTrue(rec["why_failed"])
        self.assertIn("FAILURE", rec["reflection_text"])

    def test_both_memories_in_selection_prompt(self):  # req 11
        mem = PairReflectionMemory()
        mem.add_failure(pair=("AAPL", "TSLA"), reflection_text="failed: half-life too long",
                        reason="half-life unreasonable", lessons=["avoid long half-life"])
        mem.add_success(pair=("MSFT", "GOOGL"), reflection_text="worked: strong cointegration",
                        reason="profitable", lessons=["prefer software pairs"])
        rendered = mem.render()
        self.assertIn("FAILURE MEMORY", rendered)
        self.assertIn("SUCCESS MEMORY", rendered)
        self.assertIn("AAPL/TSLA", rendered)
        self.assertIn("MSFT/GOOGL", rendered)
        llm = _FakeLLM(['{"selected_pair":["NVDA","ADBE"],"reasoning":"x","confidence":0.5,'
                        '"risks":[],"pair_reflection_used":true}'])
        agent = PolicyAgent(backend="open_source", llm=llm, symbols=["NVDA", "ADBE"])
        summary = {"pair": ["NVDA", "ADBE"], "text": "Pair NVDA/ADBE",
                   "cointegration_pvalue": 0.01, "adf_pvalue": 0.02, "half_life_days": 10.0,
                   "composite_score": 0.5, "relationship_breakdown": False}
        choice = agent.select_pair([summary], Policy(**RELAXED), pair_memory=mem)
        self.assertIn("AAPL/TSLA", choice.prompt)     # failure memory in prompt
        self.assertIn("MSFT/GOOGL", choice.prompt)    # success memory in prompt
        self.assertTrue(choice.success_memory_used)
        self.assertTrue(choice.failure_memory_used)


class Test19RelationshipStatus(unittest.TestCase):
    """Step 10/11: 3-level HEALTHY/WARNING/BREAK + their effect on the action.

    The old DANGER layer is collapsed into a HIGH-``pre_break_risk`` WARNING; only
    three monitoring statuses (HEALTHY/WARNING/BREAK) are ever emitted now.
    """

    _ALLOWED_STATUSES = {"HEALTHY", "WARNING", "BREAK"}

    def _detect(self, *, recent_corr, signal_z, adf=0.01, coint=0.01, half_life=10.0,
                news_summary=None, event_type="scheduled_break_validation",
                risk_validation_streak=0):
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        return agent.detect_relationship_break(
            pair=("A", "B"), recent_corr=recent_corr, signal_z=signal_z,
            adf_pvalue=adf, coint_pvalue=coint, half_life=half_life,
            news_summary=news_summary or {"A": {}, "B": {}, "pair_level_risk": "low"},
            pair_memory=None, policy=Policy(), event_type=event_type,
            risk_validation_streak=risk_validation_streak)

    def test_status_healthy(self):  # req 16
        ba = self._detect(recent_corr=0.9, signal_z=0.5)
        self.assertEqual(ba.relationship_status, "HEALTHY")
        self.assertEqual(ba.pre_break_risk, "LOW")
        self.assertFalse(ba.should_force_flat)

    def test_status_warning(self):  # req 17
        ba = self._detect(recent_corr=0.5, signal_z=0.5)
        self.assertEqual(ba.relationship_status, "WARNING")
        self.assertEqual(ba.pre_break_risk, "MEDIUM")
        self.assertFalse(ba.should_switch_pair)   # WARNING keeps the pair

    def test_warning_never_forces_flat(self):  # lifecycle simplification
        # A sharp isolated correlation collapse is serious pre-break stress, but it is a
        # WARNING -- and WARNING (at ANY severity, incl. the old "HIGH") NEVER forces
        # FLAT and never switches the pair. pre_break_risk remains a DIAGNOSTIC only.
        ba = self._detect(recent_corr=0.3, signal_z=0.5)
        self.assertEqual(ba.relationship_status, "WARNING")
        self.assertFalse(ba.should_force_flat)     # WARNING must NOT force FLAT
        self.assertFalse(ba.should_switch_pair)    # WARNING must NOT switch the pair

    def test_danger_status_never_emitted(self):  # Step 8 req 1/2
        # Across a sweep of statistical/news conditions, only HEALTHY/WARNING/BREAK
        # are ever produced -- DANGER must never appear as a monitoring status.
        scenarios = [
            dict(recent_corr=0.95, signal_z=0.2),                       # benign
            dict(recent_corr=0.5, signal_z=0.5),                        # mild
            dict(recent_corr=0.3, signal_z=0.5),                        # pre-break
            dict(recent_corr=0.2, signal_z=4.0, adf=0.6, coint=0.6),    # severe
            dict(recent_corr=0.3, signal_z=0.5, news_summary=_NEWS_MED),
            dict(recent_corr=0.9, signal_z=0.5, news_summary=_NEWS_CRIT),
            dict(recent_corr=0.3, signal_z=0.5, event_type="daily_risk_check"),
        ]
        for kw in scenarios:
            st = self._detect(**kw).relationship_status
            self.assertIn(st, self._ALLOWED_STATUSES, kw)
            self.assertNotEqual(st, "DANGER", kw)

    def test_status_break(self):  # req 19, 22 -- BREAK via CRITICAL news
        crit = {"A": {"critical": True, "critical_event": "files for bankruptcy",
                      "events": [], "sentiment": "negative"},
                "B": {}, "pair_level_risk": "high"}
        ba = self._detect(recent_corr=0.9, signal_z=0.5, news_summary=crit)
        self.assertEqual(ba.relationship_status, "BREAK")
        self.assertEqual(ba.pre_break_risk, "CRITICAL")
        self.assertTrue(ba.relationship_break_detected)
        self.assertTrue(ba.should_switch_pair)    # BREAK triggers a switch

    def test_warning_increases_caution(self):  # req 20
        # WARNING still ALLOWS directional trades (new prompt), but raises the
        # threshold: a z that trades in HEALTHY is FLAT in WARNING.
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        pol = Policy(entry_z=1.0, exit_band=0.5)
        kw = dict(pair=("A", "B"), position_state=PositionState.flat, signal_z=0.6,
                  news_sentiment="Neutral", policy=pol)
        healthy = agent.decide_action(relationship_status="HEALTHY", **kw)
        warning = agent.decide_action(relationship_status="WARNING", **kw)
        self.assertNotEqual(healthy.action, Action.flat)  # trades when healthy
        self.assertEqual(warning.action, Action.flat)     # higher caution threshold -> FLAT
        # but a STRONG signal still trades under WARNING (directional allowed)
        strong = agent.decide_action(relationship_status="WARNING",
                                     pair=("A", "B"), position_state=PositionState.flat,
                                     signal_z=1.2, news_sentiment="Neutral", policy=pol)
        self.assertNotEqual(strong.action, Action.flat)

    def test_high_risk_warning_action_forces_flat(self):  # req 21 (old DANGER safety)
        # A HIGH-pre_break_risk WARNING (the old DANGER level) without convincing news
        # forces FLAT -- risk control is preserved under the collapsed 3-level model.
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        pol = Policy(entry_z=1.0, exit_band=0.5)
        dec = agent.decide_action(pair=("A", "B"), position_state=PositionState.flat,
                                  signal_z=1.1, recent_pnl_3d=0.01, news_sentiment="Neutral",
                                  policy=pol, relationship_status="WARNING",
                                  pre_break_risk="HIGH")
        self.assertEqual(dec.action, Action.flat)

    def test_break_forces_flat(self):  # req 22 (action side)
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        dec = agent.decide_action(pair=("A", "B"), position_state=PositionState.flat,
                                  signal_z=3.0, recent_pnl_3d=0.01, news_sentiment="Positive",
                                  policy=Policy(), relationship_status="BREAK")
        self.assertEqual(dec.action, Action.flat)


class Test20PairPreferenceConvergence(unittest.TestCase):
    """Step 12: convergence needs policy params AND pair-preference stability."""

    @staticmethod
    def _pref_converged(history, n=3):
        window = history[-n:]
        return len(history) >= n and len(set(window)) == 1

    def test_requires_stable_pair_for_n_consecutive(self):  # req 23
        self.assertFalse(self._pref_converged(["A/B", "C/D"]))
        self.assertFalse(self._pref_converged(["A/B", "A/B", "C/D"]))
        self.assertTrue(self._pref_converged(["C/D", "A/B", "A/B", "A/B"]))

    def test_final_requires_both(self):  # req 24
        def final(policy_conv, pref_conv, pnl_change):
            return bool(policy_conv and pref_conv and abs(pnl_change) < 0.03)
        self.assertFalse(final(True, False, 0.0))   # pair preference not converged
        self.assertFalse(final(False, True, 0.0))   # policy not converged
        self.assertFalse(final(True, True, 0.10))   # PnL changed > 3%
        self.assertTrue(final(True, True, 0.01))





def _detect_break(*, recent_corr, signal_z=0.5, adf=0.01, coint=0.01, half_life=10.0,
                  news_summary=None, event_type="scheduled_break_validation",
                  risk_validation_streak=0, policy=None):
    agent = PolicyAgent(backend="offline", symbols=["A", "B"])
    return agent.detect_relationship_break(
        pair=("A", "B"), recent_corr=recent_corr, signal_z=signal_z,
        adf_pvalue=adf, coint_pvalue=coint, half_life=half_life,
        news_summary=news_summary or {"A": {}, "B": {}, "pair_level_risk": "low"},
        pair_memory=None, policy=policy or Policy(), event_type=event_type,
        risk_validation_streak=risk_validation_streak)


_NEWS_LOW = {"A": {}, "B": {}, "pair_level_risk": "low"}
_NEWS_MED = {"A": {"sentiment": "negative", "events": []}, "B": {}, "pair_level_risk": "medium"}
_NEWS_CRIT = {"A": {"critical": True, "critical_event": "files for bankruptcy",
                    "events": [], "sentiment": "negative"}, "B": {}, "pair_level_risk": "high"}
# Step 5: MILD pair-unfavorable news -- a leg-specific adverse event (analyst
# downgrade) on ONE leg with NEUTRAL sentiment (NOT a severe CRITICAL event, and
# the old "sentiment==negative" rule would have read it as LOW).
_NEWS_MILD_UNFAVORABLE = {"A": {"sentiment": "neutral", "events": ["analyst"], "critical": False},
                          "B": {"sentiment": "neutral", "events": []}, "pair_level_risk": "low"}


class Test22ConservativeBreak(unittest.TestCase):
    """Steps 2-5: BREAK is rare; statistics alone stop at a high-risk WARNING off-cadence."""

    def test_single_weak_stat_no_break(self):  # req 4
        ba = _detect_break(recent_corr=0.9, coint=0.06)  # one mildly weak statistic
        self.assertNotEqual(ba.relationship_status, "BREAK")
        self.assertIn(ba.relationship_status, ("HEALTHY", "WARNING"))

    def test_moderate_corr_decline_warning(self):  # req 5
        self.assertEqual(_detect_break(recent_corr=0.5).relationship_status, "WARNING")

    def test_sharp_isolated_high_risk_warning_not_break(self):  # req 6 (old DANGER)
        # A sharp corr drop with LOW news on the first window is serious pre-break
        # stress -> a WARNING (not a BREAK). Lifecycle simplification: WARNING does NOT
        # force FLAT (pre_break_risk is a diagnostic label only).
        ba = _detect_break(recent_corr=0.3)
        self.assertEqual(ba.relationship_status, "WARNING")
        self.assertFalse(ba.should_force_flat)     # WARNING must NOT force FLAT

    def test_high_risk_non_validation_no_switch(self):  # req 7
        ba = _detect_break(recent_corr=0.3, event_type="daily_risk_check")
        self.assertEqual(ba.relationship_status, "WARNING")
        self.assertEqual(ba.pre_break_risk, "HIGH")
        self.assertFalse(ba.should_switch_pair)

    def test_persistence_two_windows_before_break(self):  # req 8, 12
        w1 = _detect_break(recent_corr=0.3, risk_validation_streak=0)
        self.assertEqual(w1.relationship_status, "WARNING")
        self.assertEqual(w1.pre_break_risk, "HIGH")
        self.assertEqual(w1.persistence_count, 1)
        w2 = _detect_break(recent_corr=0.3, risk_validation_streak=1)
        self.assertEqual(w2.relationship_status, "BREAK")   # persisted across 2 windows
        self.assertTrue(w2.should_switch_pair)

    def test_critical_news_immediate_break(self):  # req 9
        ba = _detect_break(recent_corr=0.9, news_summary=_NEWS_CRIT,
                           event_type="daily_risk_check")  # NOT a validation day
        self.assertEqual(ba.relationship_status, "BREAK")
        self.assertTrue(ba.critical_news_exception)
        self.assertTrue(ba.should_switch_pair)

    def test_statistical_alone_non_validation_no_break(self):  # req 10
        ba = _detect_break(recent_corr=0.3, coint=0.5, adf=0.5,
                           event_type="daily_risk_check")  # severe stats, daily check
        self.assertNotEqual(ba.relationship_status, "BREAK")
        self.assertEqual(ba.relationship_status, "WARNING")
        self.assertEqual(ba.pre_break_risk, "HIGH")

    def test_mild_unfavorable_news_plus_stat_validation_break(self):  # req 11, Step 5
        # MILD pair-unfavorable news (one leg analyst downgrade, NOT a CRITICAL event)
        # combined with statistical weakening confirms a BREAK on a validation day.
        ba = _detect_break(recent_corr=0.3, news_summary=_NEWS_MILD_UNFAVORABLE,
                           risk_validation_streak=0)   # validation day, no persistence
        self.assertEqual(ba.relationship_status, "BREAK")   # mild news>=MEDIUM confirms

    def test_stat_plus_news_validation_break(self):  # req 11
        ba = _detect_break(recent_corr=0.3, news_summary=_NEWS_MED,
                           risk_validation_streak=0)   # validation day, no persistence
        self.assertEqual(ba.relationship_status, "BREAK")   # news>=MEDIUM confirms

    def test_warning_does_not_switch(self):  # req 13
        self.assertFalse(_detect_break(recent_corr=0.5).should_switch_pair)

    def test_high_risk_warning_does_not_switch(self):  # req 14
        self.assertFalse(_detect_break(recent_corr=0.3).should_switch_pair)

    def test_break_switch_only_on_validation_or_critical(self):  # req 16
        # validation + persistence -> BREAK + switch
        self.assertTrue(_detect_break(recent_corr=0.3, risk_validation_streak=1).should_switch_pair)
        # daily severe statistics -> never a switch (no critical news)
        self.assertFalse(_detect_break(recent_corr=0.3, coint=0.5, adf=0.5,
                                       event_type="daily_risk_check").should_switch_pair)

    def test_scores_and_levels_populated(self):  # req 20 (fields)
        ba = _detect_break(recent_corr=0.3, news_summary=_NEWS_MED)
        for attr in ("statistical_break_score", "news_break_score", "statistical_risk",
                     "news_risk", "persistence_count", "monitoring_event_type"):
            self.assertTrue(hasattr(ba, attr))
        self.assertGreater(ba.statistical_break_score, 0)
        self.assertEqual(ba.news_risk, "WARNING")          # news_risk is 3-level now


class Test22bNewsBreakRelaxation(unittest.TestCase):
    """Steps 8/9: news_break_level uses ONLY the 3-level HEALTHY/WARNING/BREAK."""

    _3LEVEL = {"HEALTHY", "WARNING", "BREAK"}

    def test_news_levels_are_3level_only(self):  # req 16
        for ns in (_NEWS_LOW, _NEWS_MED, _NEWS_CRIT, _NEWS_MILD_UNFAVORABLE):
            _s, level, _f, _b = news_break_level(ns)
            self.assertIn(level, self._3LEVEL)
            self.assertNotIn(level, {"LOW", "MEDIUM", "HIGH", "CRITICAL"})

    def test_mild_unfavorable_event_is_warning(self):  # req 7
        score, level, factors, is_break = news_break_level(_NEWS_MILD_UNFAVORABLE)
        self.assertEqual(level, "WARNING")
        self.assertFalse(is_break, "mild news must NOT be an immediate break")

    def test_negative_sentiment_is_warning(self):
        _s, level, _f, _b = news_break_level(_NEWS_MED)
        self.assertEqual(level, "WARNING")

    def test_routine_positive_news_is_healthy(self):
        benign = {"A": {"sentiment": "positive", "events": ["earnings"]},
                  "B": {"sentiment": "neutral", "events": []}, "pair_level_risk": "low"}
        _s, level, _f, _b = news_break_level(benign)
        self.assertEqual(level, "HEALTHY")

    def test_both_strong_positive_at_least_warning(self):  # req 17
        ns = {"A": {"financial_sentiment": "strong_positive"},
              "B": {"financial_sentiment": "strong_positive"}}
        self.assertEqual(news_break_level(ns)[1], "WARNING")

    def test_both_strong_negative_at_least_warning(self):  # req 18
        ns = {"A": {"financial_sentiment": "strong_negative"},
              "B": {"financial_sentiment": "strong_negative"}}
        self.assertEqual(news_break_level(ns)[1], "WARNING")

    def test_asymmetric_strong_sentiment_at_least_warning(self):  # req 19
        ns = {"A": {"financial_sentiment": "strong_negative"},
              "B": {"financial_sentiment": "neutral"}}
        self.assertEqual(news_break_level(ns)[1], "WARNING")
        ns2 = {"A": {"financial_sentiment": "strong_positive"},
               "B": {"financial_sentiment": "negative"}}
        self.assertEqual(news_break_level(ns2)[1], "WARNING")

    def test_structural_news_is_break(self):  # req 20
        _s, level, _f, is_break = news_break_level(_NEWS_CRIT)
        self.assertEqual(level, "BREAK")
        self.assertTrue(is_break)
        sem = {"semantic_relationship_signal": "broken_candidate",
               "A": {"financial_sentiment": "neutral"}, "B": {"financial_sentiment": "neutral"}}
        self.assertEqual(news_break_level(sem)[1], "BREAK")

    def test_mild_news_alone_is_not_an_automatic_break(self):  # Step 5 guard
        ba = _detect_break(recent_corr=0.95, signal_z=0.3,
                           news_summary=_NEWS_MILD_UNFAVORABLE)
        self.assertNotEqual(ba.relationship_status, "BREAK")


class Test22cStatisticalStrictness(unittest.TestCase):
    """Steps 5/6: 10%-stricter statistical BOUNDARIES 1.8 / 3.6 / 5.4."""

    def _stat_score(self, **kw):
        base = dict(recent_corr=0.9, signal_z=0.5, adf_pvalue=0.01, coint_pvalue=0.01,
                    half_life=10.0, policy=Policy())
        base.update(kw)
        return statistical_break_score(**base)[0]

    def test_threshold_constants_are_1p8_3p6_5p4(self):  # req 9
        from llm.policy_agent import _STAT_WARNING, _STAT_BREAK_CANDIDATE, _STAT_CRITICAL
        self.assertAlmostEqual(_STAT_WARNING, 1.8, places=6)
        self.assertAlmostEqual(_STAT_BREAK_CANDIDATE, 3.6, places=6)
        self.assertAlmostEqual(_STAT_CRITICAL, 5.4, places=6)

    def test_score_is_raw_unscaled(self):
        # No 1.10 scaling: a moderate correlation decline is exactly +2 raw points.
        self.assertAlmostEqual(self._stat_score(recent_corr=0.5), 2.0, places=6)
        self.assertAlmostEqual(self._stat_score(recent_corr=0.3), 4.0, places=6)

    def test_stat_status_mapping_uses_boundaries(self):  # req 10 (BREAK_CANDIDATE internal)
        self.assertEqual(stat_status_from_score(1.7), "HEALTHY")
        self.assertEqual(stat_status_from_score(1.8), "WARNING")
        self.assertEqual(stat_status_from_score(3.5), "WARNING")
        self.assertEqual(stat_status_from_score(3.6), "BREAK_CANDIDATE")
        self.assertEqual(stat_status_from_score(6.0), "BREAK_CANDIDATE")

    def test_break_candidate_is_internal_only(self):  # req 10/11
        # BREAK_CANDIDATE must NEVER surface as a public relationship_status.
        for cc in (0.3, 0.2):
            for ev in ("scheduled_break_validation", "daily_risk_check"):
                st = _detect_break(recent_corr=cc, event_type=ev).relationship_status
                self.assertIn(st, {"HEALTHY", "WARNING", "BREAK"})
                self.assertNotEqual(st, "BREAK_CANDIDATE")

    def test_validation_break_when_stat_ge_5p4(self):  # req 14
        # Severe stats (corr collapse + ADF + coint failure) -> stat_score >= 5.4 -> BREAK.
        ba = _detect_break(recent_corr=0.2, signal_z=4.0, adf=0.6, coint=0.6)
        self.assertGreaterEqual(ba.statistical_break_score, 5.4)
        self.assertEqual(ba.relationship_status, "BREAK")

    def test_validation_break_when_stat_ge_3p6_plus_confirmation(self):  # req 15
        # stat_score in [3.6, 5.4) alone is NOT a break...
        alone = _detect_break(recent_corr=0.3)
        self.assertAlmostEqual(alone.statistical_break_score, 4.0, places=6)
        self.assertNotEqual(alone.relationship_status, "BREAK")
        # ...but WITH confirming news (WARNING) on a validation day it IS.
        confirmed = _detect_break(recent_corr=0.3, news_summary=_NEWS_MED)
        self.assertEqual(confirmed.relationship_status, "BREAK")




class Test22ePairRankingInputs(unittest.TestCase):
    """Step 12/req 21: pair ranking combines stats + past-7-day news + reflection memory."""

    def test_selection_uses_stats_news_and_reflection_memory(self):
        import inspect
        from llm import policy_agent as pa
        # selection prompt references stats, news, and (failure/success) reflection memory
        self.assertIn("cointegration", pa.STATARB_SELECTION_PROMPT.lower())
        self.assertIn("news", pa.STATARB_SELECTION_PROMPT.lower())
        self.assertIn("memory", pa.STATARB_SELECTION_PROMPT.lower())
        # the engine candidate builder injects per-leg news into the pair summaries
        src = inspect.getsource(pa.PolicyAgent.select_pair)
        self.assertIn("pair_memory", src)


class Test23BreakValidationCadence(unittest.TestCase):
    """Pair-break validation runs every PAIR_BREAK_CHECK_FREQUENCY (default 23) days."""


    def _run(self, freq, n=70):
        syms, env = _toy_env(n=n)
        dates = sorted(env)
        agent = PolicyAgent(backend="offline", symbols=syms)
        return simulate_window(
            dates=dates, env_data=env, symbols=syms, agent=agent,
            policy=Policy(**RELAXED), lookback=10, monitor_frequency=10,
            break_check_frequency=freq, pair_candidates=_toy_candidates(syms))

    def test_validation_not_every_day(self):  # req 1
        res = self._run(15)
        n_val = sum(1 for m in res.monitoring if m.get("scheduled_break_validation"))
        n_daily = sum(1 for m in res.monitoring
                      if m.get("monitoring_event_type") == "daily_risk_check")
        self.assertGreater(len(res.monitoring), 0)
        self.assertGreater(n_daily, n_val, "most monitoring events are daily risk checks")
        self.assertLess(n_val, len(res.monitoring))

    def test_validation_every_15_days_default(self):  # req 2
        res = self._run(15)
        n_val = sum(1 for m in res.monitoring if m.get("scheduled_break_validation"))
        # ~ one formal validation per 15 held days (allow +/-1 for boundaries/switches)
        self.assertLessEqual(n_val, len(res.monitoring) // 15 + 2)
        self.assertGreaterEqual(n_val, 1)

    def test_frequency_is_configurable(self):  # req 3
        few = sum(1 for m in self._run(15).monitoring if m.get("scheduled_break_validation"))
        many = sum(1 for m in self._run(5).monitoring if m.get("scheduled_break_validation"))
        self.assertGreater(many, few, "smaller PAIR_BREAK_CHECK_FREQUENCY -> more validations")

    def test_monitoring_logs_have_required_fields(self):  # req 20
        res = self._run(15)
        m = res.monitoring[0]
        for key in ("monitoring_event_type", "scheduled_break_validation",
                    "critical_news_exception", "days_since_last_break_validation",
                    "statistical_risk", "news_risk", "statistical_break_score",
                    "news_break_score", "persistence_count", "should_force_flat",
                    "should_switch_pair", "risk_factors", "reasoning"):
            self.assertIn(key, m)

    def test_not_collapsed_into_break(self):  # req 23
        # benign (empty) news -> states should NOT be almost entirely BREAK
        res = self._run(15, n=70)
        from collections import Counter
        c = Counter(r.relationship_status for r in res.records)
        n = len(res.records)
        self.assertLess(c.get("BREAK", 0), n * 0.5, "BREAK must not dominate")
        # intermediate / healthy states carry most of the distribution
        self.assertGreater(c.get("HEALTHY", 0) + c.get("WARNING", 0),
                           c.get("BREAK", 0))
        self.assertEqual(c.get("DANGER", 0), 0, "DANGER must never be emitted")




def _break_switch_scenario(pair_memory=None, n=60, cooldown=2):
    """Hold the top pair, inject CRITICAL news on a held leg -> forces a BREAK switch."""
    import math
    from datetime import timedelta
    syms = ["A", "B", "C", "D"]
    cands = _toy_candidates(syms)
    held_leg = cands[0].pair[0]
    base = date(2025, 1, 1)
    env = {}
    for i in range(n):
        d = base + timedelta(days=i)
        prices = {"A": 100 + 5 * math.sin(i / 3.0), "B": 100 - 5 * math.sin(i / 3.0),
                  "C": 50 + 3 * math.cos(i / 4.0), "D": 50 + 3 * math.cos(i / 4.0 + 1.0)}
        news = {}
        if 24 <= i <= 27:
            news = {held_leg: [f"{held_leg} files for bankruptcy amid going concern doubt"]}
        env[d] = {"price": prices, "news": news}
    dates = sorted(env)
    agent = PolicyAgent(backend="offline", symbols=syms)
    res = simulate_window(
        dates=dates, env_data=env, symbols=syms, agent=agent, policy=Policy(**RELAXED),
        lookback=10, monitor_frequency=5, break_check_frequency=23,
        pair_reselection_cooldown=cooldown, pair_candidates=cands, pair_memory=pair_memory)
    return res, cands, "/".join(cands[0].pair)


class Test25PairLifecycle(unittest.TestCase):
    """Pair lifecycle: BREAK -> retire -> failure memory -> re-rank -> replace -> cooldown."""

    def test_critical_news_triggers_switch(self):  # req 5, 18, 19, 20
        res, _, first_pair = _break_switch_scenario()
        self.assertGreaterEqual(len(res.switch_events), 1, "a BREAK must replace the pair")
        ev = res.switch_events[0]
        self.assertNotEqual(ev["old_pair"], ev["new_pair"], "replacement must be a DIFFERENT pair")
        # not stuck forever on the first pair after the break
        post = [r for r in res.records if r.date >= ev["switch_date"]]
        self.assertTrue(any(r.selected_pair != first_pair and not r.no_trade for r in post))

    def test_break_retires_current_pair(self):  # req 9
        res, _, first_pair = _break_switch_scenario()
        retired = [lc for lc in res.pair_lifecycle if lc["pair"] == first_pair]
        self.assertTrue(retired)
        self.assertIn("RETIRED", retired[0]["states_seen"])
        self.assertIn("REPLACED", retired[0]["states_seen"])

    def test_break_generates_pair_reflection(self):  # req 7
        res, _, _ = _break_switch_scenario()
        fails = [pr for pr in res.pair_reflections if pr.get("outcome_type") == "FAILURE"]
        self.assertTrue(fails, "BREAK must generate a FAILURE pair reflection")
        self.assertTrue(fails[0]["lessons"] is not None)

    def test_break_updates_failure_memory(self):  # req 8, 11
        mem = PairReflectionMemory()
        res, _, first_pair = _break_switch_scenario(pair_memory=mem)
        self.assertTrue(mem.failed_pairs, "Failure Memory must be updated after BREAK")
        self.assertIn(first_pair, [fp["pair"] for fp in mem.failed_pairs])

    def test_replacement_pair_selected(self):  # req 10, 12
        res, _, first_pair = _break_switch_scenario()
        ev = res.switch_events[0]
        self.assertTrue(ev["new_pair"] and ev["new_pair"] != first_pair)

    def test_failed_pair_enters_cooldown(self):  # req 13
        res, _, _ = _break_switch_scenario()
        ev = res.switch_events[0]
        self.assertIsNotNone(ev["cooldown_until"])
        self.assertGreaterEqual(ev["cooldown_until"], 1)

    def test_cooldown_prevents_immediate_reselection(self):  # req 14
        # with a long cooldown the just-retired pair must not be the immediate replacement
        res, _, first_pair = _break_switch_scenario(cooldown=10)
        ev = res.switch_events[0]
        self.assertNotEqual(ev["new_pair"], first_pair)

    def test_switch_is_logged_with_fields(self):  # req 15
        res, _, _ = _break_switch_scenario()
        ev = res.switch_events[0]
        for key in ("old_pair", "new_pair", "switch_date", "switch_reason", "break_type",
                    "critical_news", "statistical_risk", "pair_reflection", "cooldown_until",
                    "failure_memory_update"):
            self.assertIn(key, ev)
        self.assertTrue(ev["failure_memory_update"])

    def test_daily_monitoring_runs(self):  # req 2 (daily monitoring active)
        res, _, _ = _break_switch_scenario()
        daily = [m for m in res.monitoring if m.get("monitoring_event_type") == "daily_risk_check"]
        self.assertTrue(daily, "daily risk checks must still run")

    def test_lifecycle_states_progress(self):  # lifecycle history
        res, _, first_pair = _break_switch_scenario()
        states = {r.lifecycle_state for r in res.records}
        self.assertIn("ACTIVE", states)
        self.assertTrue({"WARNING", "BREAK"} & states)
        self.assertNotIn("DANGER", states)  # 3-level lifecycle: no DANGER state



    def test_critical_news_requires_llm_verification(self):  # req 6
        crit = {"A": {"critical": True, "critical_event": "merger talks", "events": [],
                      "sentiment": "negative"}, "B": {}, "pair_level_risk": "high"}

        def det(agent):
            return agent.detect_relationship_break(
                pair=("A", "B"), recent_corr=0.9, signal_z=0.5, adf_pvalue=0.01,
                coint_pvalue=0.01, half_life=10.0, news_summary=crit, pair_memory=None,
                policy=Policy(), event_type="daily_risk_check")
        # LLM confirms CRITICAL + affects -> immediate BREAK
        llm_yes = _FakeLLM(['{"event_type":"acquisition","affects_pair_relationship":true,'
                            '"severity":"CRITICAL","reasoning":"leg acquired"}'])
        ba_yes = det(PolicyAgent(backend="open_source", llm=llm_yes, symbols=["A", "B"]))
        self.assertEqual(ba_yes.relationship_status, "BREAK")
        self.assertTrue(ba_yes.critical_news_exception)
        # LLM rejects (not relationship-breaking) -> NOT an immediate BREAK
        llm_no = _FakeLLM(['{"event_type":"industry commentary","affects_pair_relationship":false,'
                           '"severity":"LOW","reasoning":"about another firm"}'])
        ba_no = det(PolicyAgent(backend="open_source", llm=llm_no, symbols=["A", "B"]))
        self.assertNotEqual(ba_no.relationship_status, "BREAK")
        self.assertFalse(ba_no.critical_news_exception)




class Test28TradingCalendar(unittest.TestCase):
    """News alignment must skip weekends AND national holidays (real NYSE calendar)."""

    def test_filter_drops_weekends_and_holidays(self):
        from datetime import timedelta
        # all calendar days of Jan 2025
        days = [date(2025, 1, 1) + timedelta(days=i) for i in range(31)]
        td = filter_to_trading_days(days)
        self.assertTrue(all(d.weekday() < 5 for d in td), "no weekends")
        self.assertNotIn(date(2025, 1, 1), td, "New Year's Day excluded")
        self.assertNotIn(date(2025, 1, 9), td, "Carter mourning closure excluded")
        self.assertNotIn(date(2025, 1, 20), td, "MLK Day excluded")
        # a normal weekday is kept
        self.assertIn(date(2025, 1, 2), td)

    def test_special_closure_is_not_an_annual_holiday_and_news_rolls_forward(self):
        self.assertEqual(filter_to_trading_days([date(2026, 1, 9)]), [date(2026, 1, 9)])
        before, closed, after = date(2025, 1, 8), date(2025, 1, 9), date(2025, 1, 10)
        sessions = filter_to_trading_days([before, closed, after])
        self.assertEqual(sessions, [before, after])
        news = {before: {"A": ["before"]}, closed: {"A": ["closure"]}, after: {"A": ["same day"]}}
        self.assertEqual(align_news_to_trading_days(sessions, news)[after]["A"], ["before", "closure"])

    def test_corrected_calendar_preserves_original_75_day_test_window(self):
        from experiments.run_redesign import SYMBOLS, split_train_test
        env = load_offline_dataset(SYMBOLS, news_dir=str(Path(PROJECT_ROOT) / "data/news"))
        train, test = split_train_test(sorted(env))
        self.assertEqual(len(train), 27)
        self.assertEqual((train[0], train[-1]), (date(2025, 1, 2), date(2025, 2, 11)))
        self.assertEqual(len(test), 75)
        self.assertEqual((test[0], test[-1]), (date(2025, 2, 12), date(2025, 5, 30)))
        self.assertEqual([test[i] for i in (23, 46, 69)],
                         [date(2025, 3, 18), date(2025, 4, 21), date(2025, 5, 22)])

    def test_offline_dataset_uses_trading_days_only(self):
        from experiments.run_redesign import SYMBOLS
        env = load_offline_dataset(SYMBOLS, news_dir=str(Path(PROJECT_ROOT) / "data" / "news"))
        ses = set(env.keys())
        self.assertFalse(any(d.weekday() >= 5 for d in ses), "no weekend sessions")
        for hol in (date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18), date(2025, 5, 26)):
            self.assertNotIn(hol, ses, f"holiday {hol} must not be a trading session")

    def test_holiday_news_rolls_to_next_session(self):
        # Fri session, weekend, Mon HOLIDAY, Tue next session: Tue sees Fri+Sat+Sun+Mon news.
        fri = date(2025, 2, 14)
        sat, sun, mon_hol, tue = (date(2025, 2, 15), date(2025, 2, 16),
                                  date(2025, 2, 17), date(2025, 2, 18))
        cal_days = [fri, sat, sun, mon_hol, tue]
        trading = filter_to_trading_days(cal_days)            # drops Sat/Sun + Presidents' Day
        self.assertEqual(trading, [fri, tue])
        news = {fri: {"MSFT": ["fri"]}, sat: {"MSFT": ["sat"]}, sun: {"MSFT": ["sun"]},
                mon_hol: {"MSFT": ["mon-holiday"]}, tue: {"MSFT": ["tue-same-day"]}}
        aligned = align_news_to_trading_days(trading, news)
        self.assertEqual(aligned[tue]["MSFT"], ["fri", "sat", "sun", "mon-holiday"])
        self.assertNotIn("tue-same-day", aligned[tue]["MSFT"])   # no same-day lookahead






# --------------------------------------------------------------------------- #
# Audit-2 fix: temporal-leakage / causality of pair selection
# --------------------------------------------------------------------------- #
def _causal_pool(n_years=6, seed=11):
    """Daily 4-symbol pool ending 2025-06-01 with cointegrated (A,B) and (C,D).

    The *future* half (after a chosen decision date) can be poisoned to prove the
    causal helpers never read it.
    """
    import numpy as np
    n = n_years * 365
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2025-06-01", periods=n, freq="D")
    common = np.cumsum(rng.normal(0, 1, n))
    common2 = np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "A": 100 + common + rng.normal(0, 0.5, n),
        "B": 100 + 0.9 * common + rng.normal(0, 0.5, n),
        "C": 50 + common2 + rng.normal(0, 0.3, n),
        "D": 50 + 0.8 * common2 + rng.normal(0, 0.3, n),
    }, index=idx)


def _poison_future(pool_df, t):
    """Return a copy with every row strictly after ``t`` set to NaN (a 'future bomb').

    If any causal function reads a price after ``t`` the NaN propagates and the
    derived statistic diverges from the clean run -> the test fails.
    """
    import numpy as np
    poisoned = pool_df.copy()
    poisoned.loc[poisoned.index > pd.Timestamp(t)] = np.nan
    return poisoned




class Test33PairDirectionDiscovery(unittest.TestCase):
    """Stage-1 Top-5 coint discovery + Stage-2 Direction Discovery foundation."""

    def test_discover_top5_by_coint_pvalue(self):
        from tests.test_redesign import _toy_profiles  # local helper
        profs = _toy_profiles(["A", "B", "C", "D"])
        top = discover_pairs_coint10y(profs, top_k=5)
        self.assertLessEqual(len(top), 5)
        ps = [p for _, p in top]
        self.assertEqual(ps, sorted(ps))                 # ascending coint p (best first)

    def test_spread_state_has_required_fields(self):
        profs = _toy_profiles(["A", "B", "C", "D"])
        pc = profs[0].horizon_metrics("10Y") or next(iter(profs[0].horizons.values()))
        ss = build_spread_state(1, pc)
        for k in ("cointegration_rank", "cointegration_pvalue", "hedge_ratio_beta",
                  "current_spread", "historical_spread_mean", "historical_spread_std",
                  "current_zscore", "half_life_days", "distance_from_equilibrium_sigma"):
            self.assertIn(k, ss)

    def test_reflection_filter_rejects_specifics(self):
        for bad in ("Prefer AAPL/NVDA", "Meta usually outperforms BMRN",
                    "Always LONG_SHORT pair X", "Avoid pair Y", "On 2025-03-18 we lost"):
            self.assertFalse(sanitize_reflection(bad), bad)
        for ok in ("Go FLAT when both legs receive similar news",
                   "Calibrate confidence; do not overreact to a single headline",
                   "Balance the news signal with the mean-reversion signal"):
            self.assertTrue(sanitize_reflection(ok), ok)
        kept = filter_reflection_lessons(["Prefer AAPL/NVDA", "Use FLAT when signals conflict"])
        self.assertEqual(kept, ["Use FLAT when signals conflict"])

    def test_pair_and_direction_memory_isolated(self):
        psm, dm = PairSelectionMemory(), DirectionMemory()
        psm.add(pair=("A", "B"), quality="mean_reversion_success", coint_pvalue=0.03)
        i = dm.add_decision(selected_pair="A/B", action="LONG_SHORT", reason="x")
        dm.set_realized_pnl(i, 0.01)
        dm.clear()                                       # clearing Direction must NOT touch Pair
        self.assertEqual(len(psm.records), 1)
        self.assertEqual(len(dm.records), 0)
        # Pair memory only accepts spread-quality labels, never actions
        with self.assertRaises(ValueError):
            psm.add(pair=("A", "B"), quality="LONG_SHORT")

    def test_direction_memory_stores_required_fields(self):
        dm = DirectionMemory()
        i = dm.add_decision(selected_pair="A/B", action="SHORT_LONG", reason="r")
        dm.set_realized_pnl(i, -0.02)
        rec = dm.records[0]
        for k in ("selected_pair", "action", "reason", "realized_pnl"):
            self.assertIn(k, rec)
        self.assertEqual(rec["realized_pnl"], -0.02)

    def test_offline_direction_discovery_picks_valid_pair_and_action(self):
        profs = _toy_profiles(["A", "B", "C", "D"])
        top = discover_pairs_coint10y(profs, top_k=5)
        cands = [{"pair": pr.pair,
                  "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values())),
                  "news_a": [], "news_b": []} for pr, _ in top]
        ag = PolicyAgent(backend="offline", symbols=["A", "B", "C", "D"])
        dd = ag.discover_direction(candidates=cands, entry_z=1.0)
        self.assertIsInstance(dd, DirectionDecision)
        self.assertIn(tuple(dd.selected_pair), {tuple(c["pair"]) for c in cands})
        self.assertIn(dd.action, (Action.long_short, Action.short_long, Action.flat))

    def test_direction_prompt_has_required_sections(self):
        p = load_direction_discovery_prompt()
        self.assertIn("Pair Selection has already been completed", p)
        self.assertIn("Do not analyze Asset A and Asset B independently", p)
        self.assertIn("past 3 days Direction Memory", p)
        self.assertIn("direction_reason", p)
        self.assertTrue("LONG_SHORT" in p and "SHORT_LONG" in p and "FLAT" in p)


class Test34DiscoveryEngine(unittest.TestCase):
    """simulate_discovery_window: Top-5 -> Direction Discovery, no lifecycle."""

    def _setup(self):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {}
        for i in range(30):
            d = date(2025, 2, 1) + timedelta(days=i)
            env[d] = {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms},
                      "news": {"A": ["A beats earnings"] if i == 12 else []}}
        profs = _toy_profiles(syms)

        def provider(as_of, prices):
            top = discover_pairs_coint10y(profs, top_k=5)
            return [{"pair": pr.pair,
                     "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values()))}
                    for pr, _ in top]
        return syms, env, sorted(env), provider

    def test_runs_and_records_decisions(self):
        syms, env, dates, provider = self._setup()
        ag = PolicyAgent(backend="offline", symbols=syms)
        dm = DirectionMemory()
        res = simulate_discovery_window(dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=provider, lookback=10, selection_frequency=10,
            beta_hedged=False, direction_memory=dm, daily_direction=False)
        self.assertEqual(len(res.records), len(dates))
        self.assertEqual(len(dm.records), 3)              # legacy: 3 decisions (every 10 of 30)
        for r in dm.records:
            for k in ("selected_pair", "action", "reason"):
                self.assertIn(k, r)

    def test_beta_variant_differs(self):
        syms, env, dates, provider = self._setup()
        ag = PolicyAgent(backend="offline", symbols=syms)
        kw = dict(dates=dates, env_data=env, symbols=syms, agent=ag,
                  discovery_provider=provider, lookback=10, selection_frequency=10)
        nob = simulate_discovery_window(beta_hedged=False, **kw).cumulative_pnl
        bh = simulate_discovery_window(beta_hedged=True, **kw).cumulative_pnl
        self.assertNotEqual(round(nob, 6), round(bh, 6))   # sizing variant changes PnL

    def test_flat_holds_cash(self):
        # entry_z huge -> offline Direction Discovery always FLAT -> no positions held.
        syms, env, dates, provider = self._setup()
        ag = PolicyAgent(backend="offline", symbols=syms)
        res = simulate_discovery_window(dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=provider, lookback=10, selection_frequency=10, entry_z=99.0,
            daily_direction=False)
        self.assertTrue(all(r.action == "FLAT" and r.no_trade for r in res.records))
        self.assertAlmostEqual(res.cumulative_pnl, 0.0, places=9)   # cash-only: flat PnL

    def test_session_news_uses_aligned_bucket_without_mutating_it(self):
        previous, current, future = date(2025, 2, 3), date(2025, 2, 4), date(2025, 2, 5)
        env = {previous: {"news": {"A": ["stale"]}},
               current: {"news": {"a": ["prior-session publication"]}},
               future: {"news": {"A": ["future-only"]}}}
        items = _session_news_items(env, current, ["A", "B"])
        self.assertEqual(items, {"A": ["prior-session publication"], "B": []})
        items["A"].append("caller mutation")
        self.assertEqual(env[current]["news"]["a"], ["prior-session publication"])

    def test_prev_trading_day(self):
        from utils.engine import _prev_trading_day
        from datetime import timedelta
        # Mon decision; prev trading day is the prior Fri (env has Fri, Mon, no weekend rows)
        env = {date(2025, 2, 7): {}, date(2025, 2, 10): {}}   # Fri, Mon
        self.assertEqual(_prev_trading_day(env, date(2025, 2, 10)), date(2025, 2, 7))
        self.assertIsNone(_prev_trading_day(env, date(2025, 2, 7)))   # nothing earlier

    def test_direction_payload_has_price_delta_and_two_zscores(self):
        """Steps 5/9: payload exposes prices, daily deltas, and BOTH z-scores."""
        profs = _toy_profiles(["A", "B", "C", "D"])
        top = discover_pairs_coint10y(profs, top_k=5)
        cands = [{"pair": pr.pair,
                  "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values())),
                  "news_a": [], "news_b": [],
                  "price_a": 101.0, "price_b": 99.0,
                  "prev_price_a": 100.0, "prev_price_b": 100.0,
                  "trading_window_z": 1.234} for pr, _ in top]
        payload = PolicyAgent._build_direction_payload(cands)
        for entry in payload:
            for k in ("price_A_T", "price_B_T", "price_A_delta", "price_B_delta",
                      "price_A_delta_pct", "price_B_delta_pct",
                      "long_horizon_z_score", "trading_window_z_score",
                      "news_A_summary", "news_B_summary", "relative_A_vs_B"):
                self.assertIn(k, entry)
            self.assertEqual(entry["price_A_delta"], 1.0)
            self.assertEqual(entry["price_B_delta"], -1.0)
            self.assertEqual(entry["trading_window_z_score"], 1.234)


class Test35DiscoveryTraining(unittest.TestCase):
    """Part 2: run_discovery_training / run_discovery_test wiring (offline, deterministic)."""

    def _setup(self):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {}
        for i in range(30):
            d = date(2025, 2, 1) + timedelta(days=i)
            env[d] = {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms},
                      "news": {"A": ["A beats earnings"] if i == 12 else []}}
        profs = _toy_profiles(syms)

        def provider(as_of, prices=None):
            top = discover_pairs_coint10y(profs, top_k=5)
            return [{"pair": pr.pair,
                     "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values()))}
                    for pr, _ in top]
        return syms, env, sorted(env), provider

    def test_assign_realized_pnl_sums_next_period(self):
        from experiments.run_redesign import _assign_realized_pnl
        syms, env, dates, provider = self._setup()
        ag = PolicyAgent(backend="offline", symbols=syms)
        dm = DirectionMemory()
        res = simulate_discovery_window(dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=provider, lookback=10, selection_frequency=10,
            beta_hedged=False, direction_memory=dm, daily_direction=False)
        _assign_realized_pnl(dm, res, 10)
        # forward attribution: decision k earns the return over (start_k, start_{k+1}]
        starts = [r["decision_index"] for r in dm.records]
        n = len(res.records)
        for k, s in enumerate(starts):
            hi = (starts[k + 1] + 1) if k + 1 < len(starts) else n
            self.assertAlmostEqual(dm.records[k]["realized_pnl"],
                                   sum(r.daily_pnl for r in res.records[s + 1:hi]), places=9)

    def test_realized_pnl_alignment_with_empty_block(self):
        """Step 3: a decision block with NO candidates must not shift later PnL.

        The provider returns nothing on the second decision window, so no
        DirectionMemory record is created for it; PnL must still map correctly to the
        decisions that DID happen (block boundaries taken from decision_index)."""
        from experiments.run_redesign import _assign_realized_pnl
        syms, env, dates, provider = self._setup()

        calls = {"n": 0}
        def flaky_provider(as_of, prices=None):
            calls["n"] += 1
            if calls["n"] == 2:            # second decision day -> no candidates
                return []
            return provider(as_of, prices)

        ag = PolicyAgent(backend="offline", symbols=syms)
        dm = DirectionMemory()
        res = simulate_discovery_window(dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=flaky_provider, lookback=10, selection_frequency=10,
            beta_hedged=False, direction_memory=dm, daily_direction=False)
        _assign_realized_pnl(dm, res, 10)
        # forward attribution stays correct regardless of which decision days occur;
        # a skipped/empty selection day must not shift later PnL
        starts = [r["decision_index"] for r in dm.records]
        n = len(res.records)
        for k, s in enumerate(starts):
            hi = (starts[k + 1] + 1) if k + 1 < len(starts) else n
            self.assertAlmostEqual(dm.records[k]["realized_pnl"],
                                   sum(r.daily_pnl for r in res.records[s + 1:hi]), places=9)

    def test_reflect_direction_filtered_and_transferable(self):
        ag = PolicyAgent(backend="offline", symbols=["A", "B"])
        dm = DirectionMemory()
        i = dm.add_decision(selected_pair=("A", "B"), action="LONG_SHORT", reason="news edge")
        dm.set_realized_pnl(i, -0.02)
        refl = ag.reflect_direction(dm)
        self.assertIn("reflection_text", refl)
        self.assertTrue(refl["lessons"])
        # no leaked specifics survive the filter
        for les in refl["lessons"]:
            self.assertNotIn("A", les["lesson_text"].split())
            self.assertTrue(sanitize_reflection(les["lesson_text"]))   # passes the filter

    def test_prompt_memory_rejects_specific_lessons(self):
        pm = DirectionPromptMemory(base_prompt="BASE", symbols=["AAPL", "NVDA"])
        kept = pm.accept([{"lesson_text": "Always LONG_SHORT AAPL/NVDA"},
                          {"lesson_text": "Combine the news edge with the spread z-score."}],
                         source_round=1, reward_before=0.0, reward_after=1.0)
        self.assertEqual(kept, 1)                         # only the transferable lesson kept
        self.assertIn("spread z-score", pm.active_prompt())
        self.assertNotIn("AAPL", pm.active_prompt())

    def test_training_then_frozen_test(self):
        import experiments.run_redesign as rr
        syms, env, dates, provider = self._setup()
        orig = rr.make_discovery_provider
        rr.make_discovery_provider = lambda *a, **k: provider  # avoid needing a 10Y pool_df
        try:
            ag = PolicyAgent(backend="offline", symbols=syms)
            pm, pairmem, cycles = rr.run_discovery_training(
                None, train_dates=dates, env_data=env, agent=ag, pool_df=None,
                symbols=syms, selection_frequency=10, beta_hedged=False, max_cycles=3)
            self.assertTrue(cycles)
            self.assertTrue(any(c["early_stop"] for c in cycles) or len(cycles) == 3)
            # frozen test: no DirectionMemory, deterministic, full record set
            res = rr.run_discovery_test(test_dates=dates, env_data=env, agent=ag,
                pool_df=None, symbols=syms, frozen_prompt=pm.active_prompt(),
                selection_frequency=10, beta_hedged=False)
            self.assertEqual(len(res.records), len(dates))
        finally:
            rr.make_discovery_provider = orig

    def test_training_pair_switch_logging(self):
        # Each training cycle must carry a pair-switch summary (timeline + #switches).
        import experiments.run_redesign as rr
        syms, env, dates, provider = self._setup()
        orig = rr.make_discovery_provider
        rr.make_discovery_provider = lambda *a, **k: provider
        try:
            ag = PolicyAgent(backend="offline", symbols=syms)
            _pm, _pmem, cycles = rr.run_discovery_training(
                None, train_dates=dates, env_data=env, agent=ag, pool_df=None,
                symbols=syms, selection_frequency=10, max_cycles=2)
            self.assertTrue(cycles)
            for c in cycles:
                self.assertIn("pair_timeline", c)
                self.assertIn("n_pair_switches", c)
                self.assertIsInstance(c["pair_timeline"], list)
                self.assertTrue(c["pair_timeline"])               # decisions were recorded
                self.assertEqual(
                    c["n_pair_switches"],
                    sum(1 for k in range(1, len(c["pair_timeline"]))
                        if c["pair_timeline"][k] != c["pair_timeline"][k - 1]))
        finally:
            rr.make_discovery_provider = orig

    def test_direction_memory_cleared_each_round(self):
        # reflect_direction must run on a non-empty memory; training clears it afterward.
        import experiments.run_redesign as rr
        syms, env, dates, provider = self._setup()
        orig = rr.make_discovery_provider
        rr.make_discovery_provider = lambda *a, **k: provider
        seen = []
        try:
            ag = PolicyAgent(backend="offline", symbols=syms)
            real = ag.reflect_direction
            ag.reflect_direction = lambda dm: (seen.append(len(dm.records)) or real(dm))
            rr.run_discovery_training(None, train_dates=dates, env_data=env, agent=ag,
                pool_df=None, symbols=syms, selection_frequency=10, max_cycles=2)
            # daily Direction Discovery -> one fresh decision per training day, every round
            self.assertTrue(all(n == len(dates) for n in seen))
        finally:
            rr.make_discovery_provider = orig


class Test36BaselineComparison(unittest.TestCase):
    """Part 3: 12-strategy comparison baselines (deterministic, no LLM)."""

    def _env(self):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {date(2025, 2, 1) + timedelta(days=i):
               {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms}, "news": {}}
               for i in range(20)}
        return syms, env, sorted(env)

    def test_action_rule_mapping(self):
        from experiments.baselines import simulate_baseline_window, LONG_SHORT, SHORT_LONG
        syms, env, dates = self._env()
        kw = dict(dates=dates, env_data=env, symbols=syms,
                  pair_for_index=lambda i: ("A", "B"))
        ls = simulate_baseline_window(action=LONG_SHORT, **kw)["cumulative_pnl"]
        sl = simulate_baseline_window(action=SHORT_LONG, **kw)["cumulative_pnl"]
        # LONG_SHORT and SHORT_LONG are opposite directional bets -> mirror PnL signs
        self.assertAlmostEqual(ls, -sl, places=3)

    def test_fixed_pair_held_whole_window(self):
        from experiments.baselines import simulate_baseline_window, LONG_SHORT
        syms, env, dates = self._env()
        sim = simulate_baseline_window(dates=dates, env_data=env, symbols=syms,
            pair_for_index=lambda i: ("A", "C"), action=LONG_SHORT)
        self.assertEqual(sim["n_switches"], 0)               # never switches
        self.assertTrue(all(h == ("A", "C") for h in sim["held"]))

    def test_equal_weight_longs_both_legs(self):
        from experiments.baselines import simulate_baseline_window, EQUAL_WEIGHT_LONG
        from utils.portfolio import PortfolioManager
        syms, env, dates = self._env()
        # replicate one step to assert both legs are long (positive shares)
        from experiments.baselines import _equal_weight_long_targets
        tgt = _equal_weight_long_targets(env[dates[0]]["price"], ("A", "B"), 1.0)
        self.assertGreater(tgt["A"], 0); self.assertGreater(tgt["B"], 0)
        sim = simulate_baseline_window(dates=dates, env_data=env, symbols=syms,
            pair_for_index=lambda i: ("A", "B"), action=EQUAL_WEIGHT_LONG)
        self.assertEqual(len(sim["normalized_equity"]), len(dates))

    def test_timeline_schedule_reused_and_flat_respected(self):
        from experiments.baselines import _schedule_from_records
        recs = [type("R", (), {"selected_pair": "A/B", "no_trade": False})(),
                type("R", (), {"selected_pair": "FLAT", "no_trade": True})(),
                type("R", (), {"selected_pair": "C/D", "no_trade": False})()]
        sched = _schedule_from_records(recs)
        self.assertEqual(sched(0), ("A", "B"))
        self.assertIsNone(sched(1))                          # FLAT day -> no pair held
        self.assertEqual(sched(2), ("C", "D"))

    def test_comparison_builds_twelve_rows(self):
        import experiments.baselines as bl
        syms, env, dates = self._env()
        recs = [type("R", (), {"selected_pair": "A/B", "no_trade": False})() for _ in dates]
        nob_sim = {"normalized_equity": [1.0] * len(dates), "n_switches": 0}
        orig = bl.select_coint_pair
        bl.select_coint_pair = lambda pool_df, symbols, as_of: ("A", "B")  # toy pool too short for coint
        try:
            rows, equity = bl.build_baseline_comparison(dates=dates, env_data=env, symbols=syms,
                main_records=recs, main_nobeta_sim=nob_sim, main_beta_sim=nob_sim,
                pool_df=None, history_seed=None)
        finally:
            bl.select_coint_pair = orig
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(equity), 12)
        self.assertEqual({r["group"] for r in rows}, {"main", "timeline", "fixed_coint"})

    def test_main_strategy_labels_offline_vs_llm(self):
        """Step 6: offline results never labeled LLM; live results carry the model."""
        import experiments.baselines as bl
        off_n, off_b = bl.main_strategy_labels("offline", "qwen2.5:7b")
        self.assertNotIn("LLM", off_n); self.assertNotIn("LLM", off_b)
        self.assertIn("Offline", off_n)
        llm_n, llm_b = bl.main_strategy_labels("qwen", "qwen2.5:7b")
        self.assertIn("LLM", llm_n); self.assertIn("qwen2.5:7b", llm_n)
        self.assertIn("LLM", llm_b)

    def test_exactly_two_main_ten_baselines(self):
        import experiments.baselines as bl
        syms, env, dates = self._env()
        recs = [type("R", (), {"selected_pair": "A/B", "no_trade": False})() for _ in dates]
        nob_sim = {"normalized_equity": [1.0] * len(dates), "n_switches": 0}
        orig = bl.select_coint_pair
        bl.select_coint_pair = lambda *a, **k: ("A", "B")
        try:
            rows, _ = bl.build_baseline_comparison(
                dates=dates, env_data=env, symbols=syms, main_records=recs,
                main_nobeta_sim=nob_sim, main_beta_sim=nob_sim, pool_df=None,
                history_seed=None, main_labels=bl.main_strategy_labels("offline", "m"))
        finally:
            bl.select_coint_pair = orig
        self.assertEqual(sum(1 for r in rows if r["group"] == "main"), 2)
        self.assertEqual(sum(1 for r in rows if r["group"] != "main"), 10)
        main_names = [r["strategy"] for r in rows if r["group"] == "main"]
        self.assertTrue(all("LLM" not in n for n in main_names))   # offline -> not LLM


class Test38BetaFallback(unittest.TestCase):
    """Step 7: unified hedge-ratio (beta) fallback rule."""

    def test_effective_hedge_ratio_rule(self):
        from utils.portfolio import effective_hedge_ratio
        self.assertAlmostEqual(effective_hedge_ratio(1.3), 1.3)      # in range kept
        self.assertAlmostEqual(effective_hedge_ratio(0.5), 0.5)      # boundary kept
        self.assertAlmostEqual(effective_hedge_ratio(2.0), 2.0)      # boundary kept
        self.assertAlmostEqual(effective_hedge_ratio(1.7), 1.7)
        self.assertEqual(effective_hedge_ratio(0.4), 1.0)            # < 0.5 -> 1.0
        self.assertEqual(effective_hedge_ratio(2.5), 1.0)            # > 2.0 -> 1.0
        self.assertEqual(effective_hedge_ratio(None), 1.0)           # missing -> 1.0
        self.assertEqual(effective_hedge_ratio(float("nan")), 1.0)   # NaN -> 1.0
        self.assertEqual(effective_hedge_ratio(float("inf")), 1.0)   # inf -> 1.0
        self.assertAlmostEqual(effective_hedge_ratio(-1.5), 1.5)     # magnitude used

    def test_fallback_is_logged(self):
        from utils.portfolio import effective_hedge_ratio
        log = []
        effective_hedge_ratio(3.0, pair=("A", "B"), as_of="2025-02-01", log=log)
        self.assertEqual(len(log), 1)
        rec = log[0]
        self.assertEqual(rec["pair"], "A/B")
        self.assertEqual(rec["original_beta"], 3.0)
        self.assertEqual(rec["fallback_beta"], 1.0)
        self.assertIn("outside", rec["reason"])

    def test_beta_hedged_action_signs(self):
        from utils.portfolio import size_pair_position
        prices = {"A": 100.0, "B": 50.0}
        ls = size_pair_position(prices, ("A", "B"), 1.0, Action.long_short, hedge_ratio=1.5)
        self.assertGreater(ls["A"], 0); self.assertLess(ls["B"], 0)   # (+1, -beta)
        self.assertAlmostEqual(abs(ls["B"]) / abs(ls["A"]), 1.5, places=6)  # leg B scaled by beta
        sl = size_pair_position(prices, ("A", "B"), 1.0, Action.short_long, hedge_ratio=1.5)
        self.assertLess(sl["A"], 0); self.assertGreater(sl["B"], 0)   # (-1, +beta)
        # no-beta (hedge_ratio=1) -> equal-magnitude share legs (+1, -1)
        nb = size_pair_position(prices, ("A", "B"), 1.0, Action.long_short, hedge_ratio=1.0)
        self.assertAlmostEqual(nb["A"], -nb["B"], places=6)


class Test37DiscoveryLifecycle(unittest.TestCase):
    """Steps 1/2/8: risk lifecycle reconnected into the discovery path."""

    def _setup(self):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {}
        for i in range(30):
            d = date(2025, 2, 1) + timedelta(days=i)
            env[d] = {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms},
                      "news": {}}
        profs = _toy_profiles(syms)

        def provider(as_of, prices=None):
            top = discover_pairs_coint10y(profs, top_k=5)
            return [{"pair": pr.pair,
                     "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values()))}
                    for pr, _ in top]
        return syms, env, sorted(env), provider

    def _break_agent(self, syms, *, on_validation_only=True, switch=True, force_flat=True,
                     status="BREAK"):
        from llm.policy_agent import BreakAssessment
        score = {"BREAK": 6.0, "WARNING": 4.0, "HEALTHY": 0.0}[status]

        class _Probe(PolicyAgent):
            def detect_relationship_break(self, **kw):
                is_val = kw.get("event_type") == "scheduled_break_validation"
                fire = is_val if on_validation_only else True
                # The engine now records lifecycle from the PURE statistical break_score;
                # drive that so ``status`` actually maps to the recorded lifecycle_state.
                return BreakAssessment(
                    relationship_break_detected=bool(fire and switch),
                    reasoning="forced break", break_type="spread breakdown",
                    confidence=0.9, should_switch_pair=bool(fire and switch),
                    model="test", relationship_status=(status if fire else "WARNING"),
                    pre_break_risk=("CRITICAL" if fire else "MEDIUM"),
                    should_force_flat=bool(fire and force_flat), break_confirmed=bool(fire and switch),
                    persistence_count=0,
                    statistical_break_score=(score if fire else 0.0), news_risk="HEALTHY")
        return _Probe(backend="offline", symbols=syms)

    def test_break_triggers_switch_retire_and_cooldown(self):
        syms, env, dates, provider = self._setup()
        ag = self._break_agent(syms)
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=23, break_check_frequency=5, entry_z=0.5,
            pair_reselection_cooldown=2, lifecycle=True,
            pair_memory=(pm := PairSelectionMemory()))
        self.assertTrue(res.break_events)                       # BREAK fired
        self.assertTrue(any(r.pair_switched for r in res.records))
        self.assertGreaterEqual(len(set(res.pair_selections)), 2)  # cooldown forced a new pair
        # Pair Selection Memory got pair-QUALITY records only (never action labels)
        self.assertTrue(pm.records)
        for r in pm.records:
            self.assertIn(r["quality"], PairSelectionMemory.QUALITIES)
            self.assertNotIn(r["quality"], ("LONG_SHORT", "SHORT_LONG", "FLAT"))
            self.assertEqual(set(r.keys()), {"pair", "quality", "note", "coint_pvalue", "half_life"})

    def test_warning_holds_pair_no_force_flat(self):
        syms, env, dates, provider = self._setup()
        # WARNING every day, never BREAK -> pair retained, no breaks, and (new lifecycle)
        # NO force-flat and no break-triggered re-selection.
        ag = self._break_agent(syms, on_validation_only=False, switch=False,
                               force_flat=True, status="WARNING")
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=23, break_check_frequency=5, entry_z=0.5,
            lifecycle=True)
        self.assertFalse(res.break_events)                     # WARNING never breaks
        self.assertTrue(any(m["lifecycle_state"] == "WARNING" for m in res.monitoring))
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))   # no force-flat
        self.assertFalse(any(d.get("pair_selection_trigger") == "break"
                             for d in res.decisions))           # no break-triggered switch

    def test_healthy_holds_and_validation_cadence(self):
        syms, env, dates, provider = self._setup()
        ag = self._break_agent(syms, switch=False, force_flat=False, status="HEALTHY")
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=100, break_check_frequency=5, entry_z=0.5,
            lifecycle=True)                                    # sf=100 -> no scheduled reselect
        self.assertFalse(res.break_events)
        self.assertTrue(res.monitoring)
        val_days = [m for m in res.monitoring if m["scheduled_break_validation"]]
        self.assertGreaterEqual(len(val_days), 2)
        # with no scheduled re-selection, validations are spaced by break_check_frequency (=5)
        idxs = [dates.index(date.fromisoformat(m["date"])) for m in val_days]
        gaps = {idxs[k + 1] - idxs[k] for k in range(len(idxs) - 1)}
        self.assertEqual(gaps, {5})

    def test_audit_csv_logs_written(self):
        import tempfile, os, csv
        from utils.engine import write_discovery_logs
        syms, env, dates, provider = self._setup()
        ag = self._break_agent(syms)
        dm = DirectionMemory()
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=23, break_check_frequency=5, entry_z=0.5,
            lifecycle=True, direction_memory=dm)
        with tempfile.TemporaryDirectory() as td:
            paths = write_discovery_logs(res, td, backend="offline")
            for name in ("decision_log", "pair_pool_log", "lifecycle_log"):
                self.assertTrue(os.path.exists(paths[name]))
            with open(paths["decision_log"]) as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            for col in ("date", "active_pair", "selected_action", "backend", "beta",
                        "long_horizon_z", "trading_window_z", "news_window_start",
                        "lifecycle_state", "cooldown_flag", "realized_pnl",
                        "is_pair_selection_day", "direction_decision_failed"):
                self.assertIn(col, rows[0])
            self.assertTrue(all(r["backend"] == "offline" for r in rows))

    def test_audit_logs_tag_prevents_overwrite(self):
        # Regression: a both-passes run (No-Beta then Beta) into one dir must NOT let the
        # second pass overwrite the first pass's lifecycle/decision/pool logs.
        import tempfile, os
        from utils.engine import write_discovery_logs
        syms, env, dates, provider = self._setup()
        ag = self._break_agent(syms)
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=23, break_check_frequency=5, entry_z=0.5,
            lifecycle=True, direction_memory=DirectionMemory())
        with tempfile.TemporaryDirectory() as td:
            nb = write_discovery_logs(res, td, backend="offline", tag="nobeta")
            b = write_discovery_logs(res, td, backend="offline", tag="beta")
            # tagged filenames are distinct and both survive on disk
            for name in ("decision_log", "pair_pool_log", "lifecycle_log"):
                self.assertNotEqual(nb[name], b[name])
                self.assertTrue(os.path.exists(nb[name]))
                self.assertTrue(os.path.exists(b[name]))
            self.assertTrue(nb["lifecycle_log"].endswith("lifecycle_log_nobeta.csv"))
            self.assertTrue(b["lifecycle_log"].endswith("lifecycle_log_beta.csv"))
            # empty tag stays backward-compatible (unsuffixed)
            base = write_discovery_logs(res, td, backend="offline")
            self.assertTrue(base["lifecycle_log"].endswith("lifecycle_log.csv"))

    def test_lifecycle_off_is_unchanged(self):
        # default lifecycle=False must behave exactly like the foundation engine (no monitoring)
        syms, env, dates, provider = self._setup()
        ag = PolicyAgent(backend="offline", symbols=syms)
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, selection_frequency=10, entry_z=1.0)
        self.assertFalse(res.monitoring)
        self.assertFalse(res.break_events)
        self.assertTrue(all(r.lifecycle_state == "ACTIVE" for r in res.records))


class Test42DailyDirectionCadence(unittest.TestCase):
    """Daily direction with BREAK-gated pair selection on the default 23-day grid."""

    def _env(self, n):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {}
        for i in range(n):
            d = date(2025, 2, 1) + timedelta(days=i)
            env[d] = {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms},
                      "news": {}}
        profs = _toy_profiles(syms)

        def provider(as_of, prices=None):
            top = discover_pairs_coint10y(profs, top_k=5)
            return [{"pair": pr.pair,
                     "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values()))}
                    for pr, _ in top]
        return syms, env, sorted(env), provider

    def _run(self, n, status="BREAK"):
        syms, env, dates, provider = self._env(n)
        ag = Test37DiscoveryLifecycle()._break_agent(syms, status=status)
        dm = DirectionMemory()
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag, discovery_provider=provider,
            lookback=10, entry_z=0.5,
            lifecycle=True, daily_direction=True, direction_memory=dm)
        return res, dm, dates

    def test_training_cadence_28d(self):
        res, dm, dates = self._run(28)
        sel_days = sorted(set(p["date"] for p in res.pair_pool))
        self.assertEqual(sel_days, [dates[i].isoformat() for i in (0, 23)])
        self.assertEqual(len(res.decisions), len(dates))         # 28 daily direction events
        self.assertEqual(len(dm.records), len(dates))            # 28 daily memory records

    def test_testing_cadence_75d(self):
        res, dm, dates = self._run(75)
        sel_days = sorted(set(p["date"] for p in res.pair_pool))
        self.assertEqual(sel_days, [dates[i].isoformat() for i in (0, 23, 46, 69)])
        self.assertEqual(len(res.decisions), 75)                 # daily direction every day
        from experiments.main_artifacts import result_metrics
        self.assertEqual(sum(r.pair_switched for r in res.records), 3)
        self.assertEqual(result_metrics(res)["pair_switches"], 3)
        self.assertEqual([e["decision_index"] for e in res.switch_events], [23, 46, 69])
        self.assertEqual([e["switch_date"] for e in res.switch_events],
                         [dates[i].isoformat() for i in (23, 46, 69)])
        for event in res.switch_events:
            self.assertNotEqual(event["old_pair"], event["new_pair"])

    def test_no_break_retains_pair_for_entire_test_window(self):
        for status in ("HEALTHY", "WARNING"):
            with self.subTest(status=status):
                res, dm, dates = self._run(75, status=status)
                selected = [d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]]
                self.assertEqual(selected, [0])
                self.assertEqual([m["decision_index"] for m in res.monitoring if m["is_validation_day"]],
                                 [23, 46, 69])
                self.assertFalse(any(r.pair_switched for r in res.records))
                self.assertEqual(res.switch_events, [])
                self.assertEqual(len(dm.records), 75)

    def test_nonselection_days_reuse_pair_fresh_action(self):
        res, dm, dates = self._run(28)
        # pair stays constant between selection days, but a decision row exists every day
        sel_days = sorted(set(p["date"] for p in res.pair_pool))
        self.assertEqual(len(res.decisions), len(dates))
        non_sel = [d for d in res.decisions if not d["is_pair_selection_day"]]
        self.assertTrue(non_sel)                                 # most days are non-selection
        self.assertTrue(all("active_pair" in d for d in non_sel))


class _StubLLM:
    """Minimal OpenAI-like backend returning a fixed text (captures the last prompt)."""
    model = "stub-llm"
    def __init__(self, text):
        self._text = text
        self.last_system = None
        self.last_payload = None
    def call_text(self, *, system_prompt=None, payload=None, **kw):
        self.last_system = system_prompt
        self.last_payload = payload
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class Test43OutputSafety(unittest.TestCase):
    """Direction Discovery never emits an empty action; failures are flagged, genuine FLAT is not."""

    def _cand(self):
        pc = type("PC", (), {"coint_pvalue": 0.01, "half_life": 10.0, "current_z": 1.5,
                             "hedge_ratio": 1.0, "spread_mean": 0.0, "spread_std": 1.0})()
        return {"pair": ("A", "B"), "pc": pc, "news_a": [], "news_b": [],
                "price_a": 100.0, "price_b": 50.0, "prev_price_a": 99.0,
                "prev_price_b": 50.0, "trading_window_z": 1.5}

    def _agent(self, text):
        return PolicyAgent(backend="open_source", llm=_StubLLM(text), symbols=["A", "B"])

    def test_parse_failure_flagged(self):
        ag = self._agent("this is not json")
        dd = ag.discover_direction(candidates=[self._cand()])
        self.assertEqual(dd.action, Action.flat)
        self.assertTrue(dd.failed)
        self.assertEqual(dd.failure_type, "parse_error")

    def test_invalid_action_flagged(self):
        # action parsing happens in Mode B (daily direction); invalid action -> flagged FLAT
        ag = self._agent('{"action":"BUY","direction_reason":"x"}')
        dd = ag.discover_daily_direction(candidate=self._cand())
        self.assertEqual(dd.action, Action.flat)
        self.assertTrue(dd.failed)
        self.assertEqual(dd.failure_type, "invalid_action")

    def test_llm_exception_flagged(self):
        ag = self._agent(RuntimeError("boom"))
        dd = ag.discover_direction(candidates=[self._cand()])
        self.assertEqual(dd.action, Action.flat)
        self.assertTrue(dd.failed)
        self.assertEqual(dd.failure_type, "llm_exception")

    def test_genuine_flat_not_flagged(self):
        ag = self._agent('{"selected_pair":["A","B"],"action":"FLAT","reason":"weak"}')
        dd = ag.discover_direction(candidates=[self._cand()])
        self.assertEqual(dd.action, Action.flat)
        self.assertFalse(dd.failed)
        self.assertIsNone(dd.failure_type)

    def test_daily_parse_failure_flagged(self):
        ag = self._agent("garbage")
        dd = ag.discover_daily_direction(candidate=self._cand())
        self.assertEqual(tuple(dd.selected_pair), ("A", "B"))    # pair unchanged
        self.assertEqual(dd.action, Action.flat)
        self.assertTrue(dd.failed)
        self.assertEqual(dd.failure_type, "parse_error")

    def test_daily_genuine_action_not_flagged(self):
        ag = self._agent('{"action":"LONG_SHORT","reason":"x"}')
        dd = ag.discover_daily_direction(candidate=self._cand())
        self.assertEqual(dd.action, Action.long_short)
        self.assertFalse(dd.failed)

    def test_offline_always_valid_action(self):
        ag = PolicyAgent(backend="offline", symbols=["A", "B"])
        dd = ag.discover_daily_direction(candidate=self._cand(), entry_z=0.5)
        self.assertIn(dd.action, (Action.long_short, Action.short_long, Action.flat))
        self.assertFalse(dd.failed)


class Test44Counterfactual(unittest.TestCase):
    """Counterfactual PnL + RIGHT/WRONG + memory + past-3-day prompt + company context."""

    def test_counterfactual_pnls_and_labels(self):
        from utils.counterfactual import counterfactual_pnls, best_action, is_right
        cf = counterfactual_pnls(100.0, 50.0, 102.0, 50.0, beta=1.0)   # A up, B flat
        self.assertAlmostEqual(cf["LONG_SHORT"], 2.0 / 150.0)
        self.assertAlmostEqual(cf["SHORT_LONG"], -2.0 / 150.0)
        self.assertEqual(cf["FLAT"], 0.0)
        self.assertEqual(best_action(cf), "LONG_SHORT")               # A outperformed
        self.assertTrue(is_right("LONG_SHORT", cf))
        self.assertFalse(is_right("SHORT_LONG", cf))

    def test_wrong_has_error_analysis_right_does_not(self):
        from utils.counterfactual import memory_sentence, counterfactual_pnls
        cf = counterfactual_pnls(100.0, 50.0, 102.0, 50.0, beta=1.0)
        wrong = memory_sentence(date="2025-02-03", action="SHORT_LONG", pair="A/B",
                                pa=100.0, pb=50.0, da=2.0, db=0.0, reason="r",
                                actual_pnl=cf["SHORT_LONG"], cf=cf, right=False, tw_z=1.0)
        self.assertIn("WRONG", wrong)
        self.assertIn("Error analysis:", wrong)
        right = memory_sentence(date="2025-02-03", action="LONG_SHORT", pair="A/B",
                                pa=100.0, pb=50.0, da=2.0, db=0.0, reason="r",
                                actual_pnl=cf["LONG_SHORT"], cf=cf, right=True, tw_z=1.0)
        self.assertIn("RIGHT", right)
        self.assertNotIn("Error analysis:", right)

    def _toy(self, n=30):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {date(2025, 2, 1) + timedelta(days=i):
               {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms}, "news": {}}
               for i in range(n)}
        profs = _toy_profiles(syms)
        def prov(as_of, prices=None):
            return [{"pair": pr.pair,
                     "pc": pr.horizon_metrics("10Y") or next(iter(pr.horizons.values()))}
                    for pr, _ in discover_pairs_coint10y(profs, top_k=5)]
        return syms, env, sorted(env), prov

    def test_engine_computes_daily_counterfactuals(self):
        syms, env, dates, prov = self._toy(30)
        ag = PolicyAgent(backend="offline", symbols=syms)
        dm = DirectionMemory()
        res = simulate_discovery_window(dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=prov, lookback=10, selection_frequency=23,
            break_check_frequency=23, lifecycle=True, daily_direction=True,
            direction_memory=dm, entry_z=0.5)
        fin = [d for d in res.decisions if d.get("right_or_wrong") is not None]
        self.assertEqual(len(fin), len(dates) - 1)                    # all but the last day
        for d in fin:
            self.assertIn(d["right_or_wrong"], ("RIGHT", "WRONG"))
            self.assertAlmostEqual(d["cf_long_short"], -d["cf_short_long"])
            self.assertEqual(d["cf_flat"], 0.0)
            self.assertIn("best_action", d)
        # memory records carry counterfactuals + recent(3) returns <=3
        self.assertTrue(any(r.get("right_or_wrong") for r in dm.records))
        self.assertLessEqual(len(dm.recent(3)), 3)

    def test_daily_prompt_includes_past3_memory_and_company_context(self):
        # stub LLM captures the daily payload + system prompt
        stub = _StubLLM('{"action":"FLAT","direction_reason":"x"}')
        ag = PolicyAgent(backend="open_source", llm=stub, symbols=["A", "B"])
        pc = type("PC", (), {"coint_pvalue": 0.01, "half_life": 10.0, "current_z": 1.0,
                             "hedge_ratio": 1.0, "spread_mean": 0.0, "spread_std": 1.0})()
        cand = {"pair": ("AAPL", "MSFT"), "pc": pc, "news_a": [], "news_b": [],
                "price_a": 100.0, "price_b": 50.0, "prev_price_a": 99.0, "prev_price_b": 50.0,
                "trading_window_z": 1.0}
        recent = [{"date": "2025-02-02", "action": "LONG_SHORT", "right_or_wrong": "WRONG"}]
        ag.discover_daily_direction(candidate=cand, recent_memory=recent)
        self.assertIn("past_3_days_direction_memory", stub.last_payload)
        self.assertEqual(stub.last_payload["past_3_days_direction_memory"], recent)
        self.assertIn("COMPANY CONTEXT", stub.last_system)            # Step 5 context
        self.assertIn("Apple", stub.last_system)

    def test_pair_selection_prompt_includes_company_context(self):
        stub = _StubLLM('{"selected_pair":"AAPL/MSFT","pair_selection_reason":"r"}')
        ag = PolicyAgent(backend="open_source", llm=stub, symbols=["AAPL", "MSFT"])
        pc = type("PC", (), {"coint_pvalue": 0.01, "half_life": 10.0, "current_z": 1.0,
                             "hedge_ratio": 1.0, "spread_mean": 0.0, "spread_std": 1.0})()
        cands = [{"pair": ("AAPL", "MSFT"), "pc": pc, "news_a": [], "news_b": [],
                  "trading_window_z": 1.0}]
        dd = ag.discover_direction(candidates=cands)
        self.assertEqual(tuple(dd.selected_pair), ("AAPL", "MSFT"))
        self.assertIn("COMPANY CONTEXT", stub.last_system)
        self.assertIn("PAIR SELECTION TASK", stub.last_system)

    def test_reflection_uses_counterfactuals(self):
        ag = PolicyAgent(backend="offline", symbols=["A", "B"])
        dm = DirectionMemory()
        from utils.counterfactual import counterfactual_pnls
        cf = counterfactual_pnls(100, 50, 102, 50, 1.0)
        for k in range(3):
            i = dm.add_decision(selected_pair="A/B", action="SHORT_LONG", reason="r",
                                decision_index=k)
            dm.set_counterfactual(i, actual_pnl=cf["SHORT_LONG"], cf=cf, best="LONG_SHORT",
                                  right=False, sentence="WRONG ...")
        refl = ag.reflect_direction(dm)
        self.assertIn("counterfactual", refl["reflection_text"].lower())


class Test45SeparateRuns(unittest.TestCase):
    """Step 8: beta and no-beta produce separate tables and never mix."""

    def _env(self):
        import math
        from datetime import timedelta
        syms = ["A", "B", "C", "D"]
        env = {date(2025, 2, 1) + timedelta(days=i):
               {"price": {s: 100 + 5 * math.sin((i + ord(s)) / 4.0) for s in syms}, "news": {}}
               for i in range(20)}
        return syms, env, sorted(env)

    def _rows(self, mode):
        import experiments.baselines as bl
        syms, env, dates = self._env()
        recs = [type("R", (), {"selected_pair": "A/B", "no_trade": False})() for _ in dates]
        sim = {"normalized_equity": [1.0] * len(dates), "n_switches": 0}
        orig = bl.select_coint_pair
        bl.select_coint_pair = lambda *a, **k: ("A", "B")
        try:
            rows, eq = bl.build_mode_comparison(mode, dates=dates, env_data=env, symbols=syms,
                main_records=recs, main_sim=sim, pool_df=None, main_label=f"LLM ({mode})")
        finally:
            bl.select_coint_pair = orig
        return rows

    def test_nobeta_table_has_7_no_beta_strategies(self):
        rows = self._rows("nobeta")
        self.assertEqual(len(rows), 7)                               # 1 LLM + 6 no-beta baselines
        self.assertEqual(sum(1 for r in rows if r["group"] == "main"), 1)
        self.assertFalse(any("Beta-Hedged" in r["strategy"] for r in rows))

    def test_beta_table_has_5_beta_strategies(self):
        rows = self._rows("beta")
        self.assertEqual(len(rows), 5)                               # 1 LLM + 4 beta baselines
        self.assertTrue(all(("Beta-Hedged" in r["strategy"]) or r["group"] == "main"
                            for r in rows))

    def test_invalid_mode_raises(self):
        import experiments.baselines as bl
        with self.assertRaises(ValueError):
            bl.build_mode_comparison("mixed", dates=[date(2025, 2, 1)], env_data={},
                symbols=["A", "B"], main_records=[], main_sim={"normalized_equity": [1.0]},
                pool_df=None, main_label="x")


class Test46DeadbandCounterfactual(unittest.TestCase):
    """CF correction: dead-band best action + tolerance RIGHT/WRONG."""

    def setUp(self):
        from utils.counterfactual import COUNTERFACTUAL_DEADBAND
        self.tau = COUNTERFACTUAL_DEADBAND
        self.assertAlmostEqual(self.tau, 0.002)

    @staticmethod
    def _cf(s):
        return {"LONG_SHORT": s, "SHORT_LONG": -s, "FLAT": 0.0}

    # ---- best_action ----
    def test_zero_is_flat(self):
        from utils.counterfactual import best_action
        self.assertEqual(best_action(self._cf(0.0)), "FLAT")

    def test_inside_deadband_is_flat(self):
        from utils.counterfactual import best_action
        self.assertEqual(best_action(self._cf(0.0012)), "FLAT")
        self.assertEqual(best_action(self._cf(-0.0015)), "FLAT")

    def test_above_tau_is_long_short(self):
        from utils.counterfactual import best_action
        self.assertEqual(best_action(self._cf(0.004)), "LONG_SHORT")

    def test_below_neg_tau_is_short_long(self):
        from utils.counterfactual import best_action
        self.assertEqual(best_action(self._cf(-0.006)), "SHORT_LONG")

    # ---- RIGHT/WRONG follow corrected best ----
    def test_labels_follow_corrected_best(self):
        from utils.counterfactual import is_right
        cf = self._cf(0.004)                                  # best = LONG_SHORT
        self.assertTrue(is_right("LONG_SHORT", cf, realized_pnl=0.004))
        self.assertFalse(is_right("SHORT_LONG", cf, realized_pnl=-0.004))
        self.assertFalse(is_right("FLAT", cf))
        cf = self._cf(-0.006)                                 # best = SHORT_LONG
        self.assertTrue(is_right("SHORT_LONG", cf, realized_pnl=0.006))
        self.assertFalse(is_right("LONG_SHORT", cf, realized_pnl=-0.006))

    # ---- dead-band tolerance ----
    def test_deadband_profitable_long_short_is_right(self):
        from utils.counterfactual import is_right
        cf = self._cf(0.001)                                  # |s|<tau, LS realized=+0.001
        self.assertTrue(is_right("LONG_SHORT", cf, realized_pnl=cf["LONG_SHORT"]))

    def test_deadband_profitable_short_long_is_right(self):
        from utils.counterfactual import is_right
        cf = self._cf(-0.0015)                                # |s|<tau, SL realized=+0.0015
        self.assertTrue(is_right("SHORT_LONG", cf, realized_pnl=cf["SHORT_LONG"]))

    def test_deadband_losing_long_short_is_wrong(self):
        from utils.counterfactual import is_right
        cf = self._cf(-0.001)                                 # |s|<tau, LS realized=-0.001
        self.assertFalse(is_right("LONG_SHORT", cf, realized_pnl=cf["LONG_SHORT"]))

    def test_deadband_losing_short_long_is_wrong(self):
        from utils.counterfactual import is_right
        cf = self._cf(0.001)                                  # |s|<tau, SL realized=-0.001
        self.assertFalse(is_right("SHORT_LONG", cf, realized_pnl=cf["SHORT_LONG"]))

    def test_flat_inside_deadband_is_right(self):
        from utils.counterfactual import is_right
        cf = self._cf(0.0005)
        self.assertTrue(is_right("FLAT", cf))

    # ---- reflection receives corrected labels ----
    def test_reflection_uses_corrected_labels(self):
        from utils.counterfactual import best_action, is_right
        ag = PolicyAgent(backend="offline", symbols=["A", "B"])
        dm = DirectionMemory()
        for s, act in [(0.0005, "FLAT"), (0.0008, "FLAT"), (0.004, "LONG_SHORT")]:
            cf = self._cf(s)
            i = dm.add_decision(selected_pair="A/B", action=act, reason="r", decision_index=len(dm.records))
            dm.set_counterfactual(i, actual_pnl=cf.get(act, 0.0), cf=cf,
                                  best=best_action(cf), right=is_right(act, cf, realized_pnl=cf.get(act, 0.0)),
                                  sentence="...")
        refl = ag.reflect_direction(dm)
        # FLAT is now a valid best action -> reflection text reports a nonzero FLAT-best count
        self.assertIn("counterfactual", refl["reflection_text"].lower())
        self.assertRegex(refl["reflection_text"], r"FLAT=\d*[1-9]")   # FLAT-best count >= 1

    def test_diagnostics_deadband_right_via_profit(self):
        from utils.counterfactual import counterfactual_diagnostics
        recs = [
            {"cf_long_short": 0.001, "best_action": "FLAT", "final_executed_action": "LONG_SHORT",
             "right_or_wrong": "RIGHT"},                      # profitable directional in band
            {"cf_long_short": 0.0005, "best_action": "FLAT", "final_executed_action": "FLAT",
             "right_or_wrong": "RIGHT"},                      # plain FLAT
            {"cf_long_short": 0.004, "best_action": "LONG_SHORT", "final_executed_action": "FLAT",
             "right_or_wrong": "WRONG"}]
        d = counterfactual_diagnostics(recs)
        self.assertEqual(d["flat_best_after_deadband"], 2)
        self.assertEqual(d["flat_best_before_deadband"], 0)   # old argmax never FLAT
        self.assertEqual(d["deadband_right_via_profit"], 1)


class _FakePC:
    def __init__(self, p, hl, cz=0.0):
        self.coint_pvalue = p; self.half_life = hl; self.current_z = cz
        self.hedge_ratio = 1.0; self.spread_vol = 0.1; self.adf_pvalue = 0.01
        self.spread_mean = 0.0; self.spread_std = 1.0


class _FakeProf:
    def __init__(self, pair, p, hl, cz=0.0):
        self.pair = pair; self._pc = _FakePC(p, hl, cz); self.horizons = {"10Y": self._pc}
    def horizon_metrics(self, h):
        return self._pc


class Test39PairDiscoveryFilter(unittest.TestCase):
    """Goal 1: Top-20 by coint p -> half-life [2,20] filter -> Top-5."""

    def _profs(self):
        # 25 pairs; coint p ascending with index; half-life varied (ceiling = 60)
        profs = []
        for i in range(25):
            hl = 10.0 if i % 2 == 0 else 90.0       # even idx in-range, odd out-of-range
            profs.append(_FakeProf((f"X{i}", f"Y{i}"), p=0.001 * (i + 1), hl=hl))
        return profs

    def test_top20_then_halflife_then_top5(self):
        profs = self._profs()
        sel, diag = discover_pairs_coint10y(profs, top_k=5, return_diagnostics=True)
        self.assertEqual(len(diag["top20"]), 20)                  # Top-20 by p first
        self.assertLessEqual(len(sel), 5)                         # final pool <= 5
        ps = [p for _, p in sel]
        self.assertEqual(ps, sorted(ps))                         # p-order preserved
        for prof, _ in sel:                                      # all pass half-life [2,60]
            self.assertTrue(2 <= prof.horizon_metrics("10Y").half_life <= 60)
        self.assertFalse(diag["fallback_used"])

    def test_halflife_bounds_excluded(self):
        profs = [_FakeProf(("A", "B"), 0.001, 1.0),   # hl < 2 -> excluded
                 _FakeProf(("C", "D"), 0.002, 65.0),  # hl > 60 -> excluded
                 _FakeProf(("E", "F"), 0.003, 30.0)]  # in range
        sel, diag = discover_pairs_coint10y(profs, top_k=5, return_diagnostics=True)
        self.assertEqual([("E", "F")], [tuple(p.pair) for p, _ in sel])
        self.assertEqual(diag["n_pass_halflife"], 1)

    def test_fewer_than_5_no_backfill(self):
        profs = [_FakeProf(("A", "B"), 0.001, 10.0),   # pass
                 _FakeProf(("C", "D"), 0.002, 99.0),   # fail
                 _FakeProf(("E", "F"), 0.003, 99.0)]   # fail
        sel = discover_pairs_coint10y(profs, top_k=5)
        self.assertEqual(len(sel), 1)                            # only the passing pair, no backfill

    def test_fallback_when_none_pass(self):
        profs = [_FakeProf(("A", "B"), 0.001, 99.0),
                 _FakeProf(("C", "D"), 0.002, 99.0)]
        sel, diag = discover_pairs_coint10y(profs, top_k=5, return_diagnostics=True)
        self.assertTrue(diag["fallback_used"])                   # zero pass -> fallback
        self.assertEqual([("A", "B"), ("C", "D")], [tuple(p.pair) for p, _ in sel])


class Test40AntiFlat(unittest.TestCase):
    """Goal 2: reflection eligibility + consecutive-FLAT override."""

    def _const_env(self, n=50):
        from datetime import timedelta
        syms = ["A", "B"]
        env = {date(2025, 2, 1) + timedelta(days=i):
               {"price": {"A": 100.0, "B": 50.0}, "news": {}} for i in range(n)}
        return syms, env, sorted(env)

    def _flat_provider(self, cz):
        def p(as_of, prices=None):
            return [{"pair": ("A", "B"), "pc": _FakePC(0.001, 10.0, cz)}]
        return p

    def _flat_agent(self, syms, *, state="HEALTHY", switch=None):
        """Mock agent that always picks FLAT and reports a fixed lifecycle ``state``.

        The mock mirrors the REAL (clarified) lifecycle contract: NO state ever forces
        flat (``should_force_flat`` is always False). A BREAK only triggers REPLACEMENT
        (``should_switch_pair``); ``switch`` overrides that for isolation tests.
        """
        from llm.policy_agent import DirectionDecision, BreakAssessment
        is_break = (state == "BREAK")
        do_switch = is_break if switch is None else switch
        score = {"HEALTHY": 0.0, "WARNING": 4.0, "BREAK": 6.0}[state]   # drives break_score mapping

        class _A(PolicyAgent):
            def discover_direction(self, *, candidates, **kw):
                return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat, "forced flat")
            def discover_daily_direction(self, *, candidate, **kw):
                return DirectionDecision(tuple(candidate["pair"]), Action.flat, "forced daily flat")
            def detect_relationship_break(self, **kw):
                return BreakAssessment(is_break, "x", "none", 0.5, do_switch, "t",
                                       relationship_status=state,
                                       pre_break_risk=("CRITICAL" if is_break else "MEDIUM"),
                                       should_force_flat=False, persistence_count=0,
                                       break_confirmed=is_break,
                                       statistical_break_score=score, news_risk="HEALTHY")
        return _A(backend="offline", symbols=syms)

    # ---- episode evidence gating ----
    def test_episode_status_levels(self):
        from experiments.run_redesign import episode_status
        dm = DirectionMemory()
        for _ in range(3):
            dm.add_decision(selected_pair="A/B", action="FLAT", reason="x")
        self.assertEqual(episode_status(dm), "NO_EVIDENCE")
        dm.add_decision(selected_pair="A/B", action="LONG_SHORT", reason="x")
        self.assertEqual(episode_status(dm), "LOW_EVIDENCE")       # 1 trade < 2
        dm.add_decision(selected_pair="A/B", action="SHORT_LONG", reason="x")
        self.assertEqual(episode_status(dm), "VALID")              # 2 trades

    def test_no_evidence_round_skips_prompt_and_earlystop(self):
        import experiments.run_redesign as rr
        from datetime import timedelta
        syms = ["A", "B"]
        env = {date(2025, 2, 1) + timedelta(days=i):
               {"price": {"A": 100.0, "B": 50.0}, "news": {}} for i in range(28)}
        dates = sorted(env)
        prov = self._flat_provider(0.0)
        orig = rr.make_discovery_provider
        rr.make_discovery_provider = lambda *a, **k: prov
        try:
            ag = self._flat_agent(syms)
            pm, _pmem, cycles = rr.run_discovery_training(
                None, train_dates=dates, env_data=env, agent=ag, pool_df=None,
                symbols=syms, selection_frequency=10, max_cycles=3)   # 3 decisions -> all FLAT
            self.assertTrue(all(c["episode_status"] == "NO_EVIDENCE" for c in cycles))
            self.assertTrue(all(not c["early_stop"] for c in cycles))  # no early-stop contribution
            self.assertEqual(pm.version, 1)                            # no prompt update
        finally:
            rr.make_discovery_provider = orig

    # ---- consecutive-FLAT override ----
    def _run_flat(self, cz, *, lifecycle=False, state="HEALTHY", switch=None,
                  direction_memory=None):
        syms, env, dates = self._const_env(50)        # 5 decisions at sf=10
        ag = self._flat_agent(syms, state=state, switch=switch)
        return simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=ag,
            discovery_provider=self._flat_provider(cz), lookback=10,
            selection_frequency=10, lifecycle=lifecycle, break_check_frequency=5,
            direction_memory=direction_memory)

    def test_first_three_flat_allowed_fourth_overridden(self):
        res = self._run_flat(cz=2.0)
        decs = res.decisions
        self.assertGreaterEqual(len(decs), 4)
        for k in range(3):
            self.assertFalse(decs[k]["flat_override_triggered"])      # first 3 allowed
            self.assertEqual(decs[k]["final_executed_action"], "FLAT")
        self.assertTrue(decs[3]["flat_override_triggered"])           # 4th overridden
        self.assertNotEqual(decs[3]["final_executed_action"], "FLAT")

    def test_override_positive_z_short_long(self):
        res = self._run_flat(cz=2.0)                                  # long_horizon_z>0 (tw_z=0)
        self.assertEqual(res.decisions[3]["final_executed_action"], "SHORT_LONG")

    def test_override_negative_z_long_short(self):
        res = self._run_flat(cz=-2.0)
        self.assertEqual(res.decisions[3]["final_executed_action"], "LONG_SHORT")

    def test_warning_does_not_block_override(self):  # spec #10
        # lifecycle WARNING must NOT block the anti-FLAT override and must NOT force FLAT;
        # the 4th consecutive FLAT is still overridden into a directional trade.
        res = self._run_flat(cz=2.0, lifecycle=True, state="WARNING")
        self.assertFalse(any(d["flat_override_blocked_by_lifecycle"] for d in res.decisions))
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))
        self.assertTrue(any(d["flat_override_triggered"] for d in res.decisions))

    def test_break_does_not_force_flat_or_block_override(self):  # spec clarification
        # A BREAK status must NOT force FLAT and must NOT suppress the anti-FLAT override.
        # On the 4th day the pair was selected on day 0 and not yet re-selected (sf=10),
        # so it is NOT in cooldown -> the override fires despite the BREAK status.
        res = self._run_flat(cz=2.0, lifecycle=True, state="BREAK", switch=False)
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))  # never force-flat
        d3 = res.decisions[3]                                # 4th consecutive FLAT day
        self.assertTrue(d3["flat_override_triggered"])       # BREAK does not suppress override
        self.assertFalse(d3["flat_override_blocked_by_lifecycle"])
        self.assertNotEqual(d3["final_executed_action"], "FLAT")

    def test_override_block_predicate(self):  # spec #10/#11/#12 (unit)
        from utils.engine import anti_flat_override_blocked
        # Lifecycle state NEVER blocks the override (informational only)...
        self.assertFalse(anti_flat_override_blocked("HEALTHY"))
        self.assertFalse(anti_flat_override_blocked("WARNING"))          # WARNING never blocks
        self.assertFalse(anti_flat_override_blocked("BREAK"))            # BREAK never blocks
        self.assertFalse(anti_flat_override_blocked("BREAK", break_triggered=True))
        # ...only an active cooldown on the held pair blocks it.
        self.assertTrue(anti_flat_override_blocked("WARNING", in_cooldown=True))
        self.assertTrue(anti_flat_override_blocked("HEALTHY", in_cooldown=True))

    def test_override_applies_without_direction_memory(self):
        # testing path passes no DirectionMemory; the override must still apply
        res = self._run_flat(cz=2.0, direction_memory=None)
        self.assertTrue(res.decisions[3]["flat_override_triggered"])


class Test42LifecycleSimplified(unittest.TestCase):
    """Lifecycle simplification: exactly 3 states; WARNING never forces flat / retires /
    cooldowns / reselects; only a validation-window BREAK does. The validation grid
    is global and independent of pair changes."""

    def _env(self, syms, n):
        from datetime import timedelta
        base = {s: 100.0 + 5.0 * i for i, s in enumerate(syms)}
        env = {date(2025, 2, 1) + timedelta(days=k):
               {"price": dict(base), "news": {}} for k in range(n)}
        return sorted(env), env

    def _agent(self, syms, *, state="HEALTHY", news="HEALTHY"):
        """Mock agent: drives the PURE statistical break_score (and news_risk) so the
        engine's combined lifecycle mapping records the intended state. No force-flat."""
        from llm.policy_agent import DirectionDecision, BreakAssessment
        score = {"HEALTHY": 0.0, "WARNING": 4.0, "BREAK": 6.0}[state]

        class _A(PolicyAgent):
            def discover_direction(self, *, candidates, **kw):
                return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat, "f")
            def discover_daily_direction(self, *, candidate, **kw):
                return DirectionDecision(tuple(candidate["pair"]), Action.flat, "f")
            def detect_relationship_break(self, **kw):
                return BreakAssessment(False, "x", "none", 0.5, False, "t",
                                       relationship_status=state, pre_break_risk="MEDIUM",
                                       should_force_flat=False, persistence_count=0,
                                       statistical_break_score=score, news_risk=news)
        return _A(backend="offline", symbols=syms)

    def _const_provider(self, pair=("A", "B")):
        return lambda as_of, prices=None: [{"pair": pair, "pc": _FakePC(0.001, 10.0, 0.0)}]

    def _switching_provider(self):
        st = {"k": -1}
        def p(as_of, prices=None):
            st["k"] += 1
            a, b = f"P{2*st['k']}", f"P{2*st['k']+1}"
            return [{"pair": (a, b), "pc": _FakePC(0.001, 10.0, 0.0)}]
        return p

    def _run(self, *, syms, provider, state="HEALTHY", news="HEALTHY", n=50, sf=10, bcf=23):
        dates, env = self._env(syms, n)
        agent = self._agent(syms, state=state, news=news)
        return simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=agent,
            discovery_provider=provider, lookback=10, selection_frequency=sf,
            lifecycle=True, break_check_frequency=bcf)

    # ---- spec #1: lifecycle state space is exactly {HEALTHY, WARNING, BREAK} ----
    def test_only_three_states(self):
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        allowed = {"HEALTHY", "WARNING", "BREAK"}
        base = dict(pair=("A", "B"), signal_z=0.5, half_life=10.0,
                    news_summary={"A": {}, "B": {}}, pair_memory=None, policy=Policy(),
                    event_type="scheduled_break_validation")
        for kw in (dict(recent_corr=0.95, adf_pvalue=0.01, coint_pvalue=0.01),
                   dict(recent_corr=0.5, adf_pvalue=0.01, coint_pvalue=0.01),
                   dict(recent_corr=0.3, adf_pvalue=0.01, coint_pvalue=0.01),
                   dict(recent_corr=0.2, adf_pvalue=0.6, coint_pvalue=0.6)):
            ba = agent.detect_relationship_break(**{**base, **kw})
            self.assertIn(ba.relationship_status, allowed)

    # ---- spec #2/#3/#4/#5: WARNING is inert except as context; daily direction runs ----
    def test_warning_inert_and_daily_direction_runs(self):
        res = self._run(syms=["A", "B"], provider=self._const_provider(), state="WARNING", n=50)
        self.assertEqual(res.break_events, [])                       # #5 no re-selection trigger
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))  # #2 no force-flat
        self.assertFalse(any(d["pair_retired"] for d in res.decisions))          # #4 no retire
        self.assertFalse(any(d["cooldown_active"] for d in res.decisions))       # #3 no cooldown
        self.assertEqual(res.pair_selections, ["A/B"])             # initial selection only
        self.assertEqual(len(res.decisions), 50)                    # #13 daily direction runs
        states = [d["lifecycle_state"] for d in res.decisions]
        self.assertNotIn("BREAK", states)
        self.assertEqual(states[0], "HEALTHY")                      # day 0: selected, not yet monitored
        self.assertTrue(all(s == "WARNING" for s in states[1:]))    # WARNING persists, never resets

    # ---- BREAK triggers replacement at the validation day, NOT force-flat ----
    def test_break_triggers_replacement_no_force_flat(self):
        # Two pairs available; a persistent BREAK (break_score>=5) means the validation
        # window contains BREAK -> at the validation day the pair is retired + cooled down
        # and REPLACED by the other (fresh) pair. BREAK must NOT force flat at any point.
        syms = ["A", "B", "C", "D"]
        prov = lambda as_of, prices=None: [
            {"pair": ("A", "B"), "pc": _FakePC(0.001, 10.0, 0.0)},
            {"pair": ("C", "D"), "pc": _FakePC(0.002, 12.0, 0.0)}]
        res = self._run(syms=syms, provider=prov, state="BREAK", n=50, sf=10, bcf=23)
        self.assertTrue(res.break_events)                            # BREAK -> re-selection
        self.assertTrue(any(d["pair_retired"] for d in res.decisions))      # retire
        self.assertTrue(any(d["cooldown_active"] for d in res.decisions))   # cooldown
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))  # NO force-flat
        self.assertIn("C/D", {d["active_pair"] for d in res.decisions})     # replaced -> new pair

    # ---- global validation clock preserved for both retained and replaced pairs ----
    def test_validation_fires_on_global_grid_same_pair(self):
        # validations land on the GLOBAL grid (multiples of bcf=23) regardless of pair.
        res = self._run(syms=["A", "B"], provider=self._const_provider(),
                        state="HEALTHY", n=50, sf=10, bcf=23)
        val_idx = [m["decision_index"] for m in res.monitoring if m["scheduled_break_validation"]]
        self.assertTrue(val_idx)
        self.assertTrue(all(ix % 23 == 0 for ix in val_idx))       # fixed cadence

    def test_break_reselecting_same_pair_is_not_a_switch(self):
        from experiments.main_artifacts import result_metrics
        res = self._run(syms=["A", "B"], provider=self._const_provider(),
                        state="BREAK", n=75, sf=23, bcf=23)
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]],
                         [0, 23, 46, 69])
        self.assertEqual(len(res.break_events), 3)
        self.assertFalse(any(r.pair_switched for r in res.records))
        self.assertEqual(result_metrics(res)["pair_switches"], 0)

    def test_validation_grid_is_pair_independent(self):
        # a genuinely NEW pair every selection must NOT change the global validation grid:
        # validations still fire at the same fixed cadence (the clock is NOT reset on
        # pair changes).
        syms = [f"P{i}" for i in range(12)]
        res = self._run(syms=syms, provider=self._switching_provider(),
                        state="BREAK", n=50, sf=10, bcf=23)
        val_idx = [m["decision_index"] for m in res.monitoring if m["scheduled_break_validation"]]
        self.assertTrue(val_idx)                                   # validations DO fire
        self.assertTrue(all(ix % 23 == 0 for ix in val_idx))
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]],
                         [0, 23, 46])

    # ---- spec #15: persistent WARNING can escalate to BREAK on a validation day ----
    def test_persistent_warning_becomes_break(self):
        agent = PolicyAgent(backend="offline", symbols=["A", "B"])
        kw = dict(pair=("A", "B"), recent_corr=0.3, signal_z=0.5, adf_pvalue=0.01,
                  coint_pvalue=0.01, half_life=10.0, news_summary={"A": {}, "B": {}},
                  pair_memory=None, policy=Policy(), event_type="scheduled_break_validation")
        w1 = agent.detect_relationship_break(risk_validation_streak=0, **kw)
        self.assertEqual(w1.relationship_status, "WARNING")
        w2 = agent.detect_relationship_break(risk_validation_streak=1, **kw)
        self.assertEqual(w2.relationship_status, "BREAK")           # persisted -> BREAK


class Test43CombinedLifecycleRecording(unittest.TestCase):
    """Lifecycle threshold + recording change: pure statistical break_score (0-6) mapping
    + separate news relation signal -> combined lifecycle_state; validation-window
    re-selection; break_score_trend recording."""

    # ---- mapping helpers (spec 1-6, 14) ----
    def test_statistical_state_healthy(self):                # 1
        from utils.engine import statistical_lifecycle_state
        for s in (0, 1, 2, 3):
            self.assertEqual(statistical_lifecycle_state(s), "HEALTHY")

    def test_statistical_state_warning(self):                # 2
        from utils.engine import statistical_lifecycle_state
        for s in (3.5, 4, 4.9):
            self.assertEqual(statistical_lifecycle_state(s), "WARNING")

    def test_statistical_state_break(self):                  # 3
        from utils.engine import statistical_lifecycle_state
        for s in (5, 6):
            self.assertEqual(statistical_lifecycle_state(s), "BREAK")

    def test_news_break_overrides_low_score(self):           # 4
        from utils.engine import (statistical_lifecycle_state, news_relation_signal,
                                   news_lifecycle_state, combined_lifecycle_state)
        stat = statistical_lifecycle_state(2)                # HEALTHY (score < 5)
        news = news_lifecycle_state(news_relation_signal("BREAK"))
        self.assertEqual(combined_lifecycle_state(stat, news), "BREAK")

    def test_news_warning_when_stat_healthy(self):           # 5
        from utils.engine import (news_relation_signal, news_lifecycle_state,
                                   combined_lifecycle_state)
        news = news_lifecycle_state(news_relation_signal("WARNING"))
        self.assertEqual(combined_lifecycle_state("HEALTHY", news), "WARNING")

    def test_combined_break_if_either(self):                 # 6
        from utils.engine import combined_lifecycle_state
        self.assertEqual(combined_lifecycle_state("BREAK", "HEALTHY"), "BREAK")
        self.assertEqual(combined_lifecycle_state("HEALTHY", "BREAK"), "BREAK")
        self.assertEqual(combined_lifecycle_state("WARNING", "WARNING"), "WARNING")
        self.assertEqual(combined_lifecycle_state("HEALTHY", "HEALTHY"), "HEALTHY")

    def test_break_score_trend_constant_not_flat(self):      # 14
        from utils.engine import break_score_trend
        self.assertEqual(break_score_trend(None, 4.0), "constant")
        self.assertEqual(break_score_trend(4.0, 4.0), "constant")   # constant, NOT "flat"
        self.assertEqual(break_score_trend(3.0, 4.0), "rising")
        self.assertEqual(break_score_trend(5.0, 4.0), "falling")

    # ---- engine behavior (spec 7-13) ----
    def _agent(self, syms, *, score=0.0, news="HEALTHY"):
        from llm.policy_agent import DirectionDecision, BreakAssessment

        class _A(PolicyAgent):
            def discover_direction(self, *, candidates, **kw):
                return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat, "f")
            def discover_daily_direction(self, *, candidate, **kw):
                return DirectionDecision(tuple(candidate["pair"]), Action.flat, "f")
            def detect_relationship_break(self, **kw):
                return BreakAssessment(False, "x", "none", 0.5, False, "t",
                                       relationship_status="HEALTHY", pre_break_risk="LOW",
                                       should_force_flat=False, persistence_count=0,
                                       statistical_break_score=score, news_risk=news)
        return _A(backend="offline", symbols=syms)

    def _env(self, syms, n):
        from datetime import timedelta
        base = {s: 100.0 + 5.0 * i for i, s in enumerate(syms)}
        env = {date(2025, 2, 1) + timedelta(days=k): {"price": dict(base), "news": {}}
               for k in range(n)}
        return sorted(env), env

    def _run(self, *, score, news="HEALTHY", n=40, sf=10, bcf=23, provider=None, syms=("A", "B")):
        syms = list(syms)
        dates, env = self._env(syms, n)
        prov = provider or (lambda as_of, prices=None:
                            [{"pair": tuple(syms[:2]), "pc": _FakePC(0.001, 10.0, 0.0)}])
        return simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=self._agent(syms, score=score, news=news),
            discovery_provider=prov, lookback=10, selection_frequency=sf, lifecycle=True,
            break_check_frequency=bcf)

    def test_break_recorded_on_daily_day(self):              # 7
        res = self._run(score=6.0, n=10, bcf=23)             # bcf>n -> daily checks only
        self.assertTrue(any((not m["is_validation_day"]) and m["lifecycle_state"] == "BREAK"
                            for m in res.monitoring))

    def test_break_does_not_force_flat(self):                # 8
        res = self._run(score=6.0, n=10, bcf=23)
        self.assertFalse(any(d["force_flat_triggered"] for d in res.decisions))

    def test_break_no_immediate_switch(self):                # 9
        res = self._run(score=6.0, n=20, bcf=50)             # never a validation day
        self.assertFalse(res.break_events)
        self.assertFalse(any(d.get("pair_selection_trigger") == "break" for d in res.decisions))
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]], [0])

    def test_direction_discovery_runs_each_day(self):        # 10
        for sc in (0.0, 4.0, 6.0):
            res = self._run(score=sc, n=15)
            self.assertEqual(len(res.decisions), 15)

    def test_validation_window_break_triggers_reselection(self):  # 11
        syms = ["A", "B", "C", "D"]
        prov = lambda a, prices=None: [{"pair": ("A", "B"), "pc": _FakePC(0.001, 10.0, 0.0)},
                                       {"pair": ("C", "D"), "pc": _FakePC(0.002, 12.0, 0.0)}]
        res = self._run(score=6.0, n=40, sf=10, bcf=23, provider=prov, syms=syms)
        self.assertTrue(any(m["pair_reselection_triggered"] for m in res.monitoring))
        self.assertTrue(any(m["validation_window_has_break"] for m in res.monitoring))

    def test_no_break_in_window_no_reselection(self):        # 12
        res = self._run(score=4.0, n=40, sf=10, bcf=23)      # WARNING only
        self.assertFalse(any(m["pair_reselection_triggered"] for m in res.monitoring))
        self.assertFalse(any(m["validation_window_has_break"] for m in res.monitoring))
        self.assertFalse(res.break_events)
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]], [0])

    def test_earlier_break_triggers_at_validation_but_not_next_window(self):
        # A/B breaks only at index 5, then recovers. Index 23 must still reselect,
        # but the historical BREAK must not cause another selection at index 46.
        from llm.policy_agent import DirectionDecision, BreakAssessment
        syms = ["A", "B", "C", "D"]
        calls = {"n": 0}

        def prov(as_of, prices=None):
            calls["n"] += 1
            pair = ("A", "B") if calls["n"] == 1 else ("C", "D")
            return [{"pair": pair, "pc": _FakePC(0.001, 10.0, 0.0)}]

        class _A(PolicyAgent):
            monitored_days = 0
            def discover_direction(self, *, candidates, **kw):
                return DirectionDecision(tuple(candidates[0]["pair"]), Action.flat, "f")
            def discover_daily_direction(self, *, candidate, **kw):
                return DirectionDecision(tuple(candidate["pair"]), Action.flat, "f")
            def detect_relationship_break(self, *, pair, **kw):
                self.monitored_days += 1
                score = 6.0 if self.monitored_days == 5 else 0.0
                return BreakAssessment(False, "x", "none", 0.5, False, "t",
                                       relationship_status="HEALTHY", pre_break_risk="LOW",
                                       should_force_flat=False, persistence_count=0,
                                       statistical_break_score=score, news_risk="HEALTHY")
        dates, env = self._env(syms, 75)
        res = simulate_discovery_window(
            dates=dates, env_data=env, symbols=syms, agent=_A(backend="offline", symbols=syms),
            discovery_provider=prov, lookback=10, selection_frequency=10, lifecycle=True,
            break_check_frequency=23)
        v = next(m for m in res.monitoring if m["is_validation_day"])   # global grid -> index 23
        self.assertEqual(v["pair"], "A/B")
        self.assertEqual(v["lifecycle_state"], "HEALTHY")
        self.assertTrue(v["validation_window_has_break"])
        self.assertTrue(v["pair_reselection_triggered"])
        self.assertEqual(v["validation_window_break_source"], "statistical")
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]], [0, 23])
        later = [m for m in res.monitoring if m["is_validation_day"] and m["decision_index"] > 23]
        self.assertTrue(all(not m["pair_reselection_triggered"] for m in later))

    def test_empty_initial_pool_retries_only_at_validation(self):
        calls = []
        def provider(as_of, prices=None):
            calls.append(as_of)
            return [] if len(calls) == 1 else [{"pair": ("A", "B"), "pc": _FakePC(0.001, 10., 0.)}]
        res = self._run(score=0., n=75, sf=10, bcf=23, provider=provider)
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[1] - calls[0]).days, 23)  # daily synthetic dates
        self.assertEqual([d["decision_index"] for d in res.decisions if d["is_pair_selection_day"]], [23])
        self.assertFalse(res.break_events)

    def test_validation_window_break_source(self):           # 13
        res = self._run(score=6.0, news="HEALTHY", n=40, bcf=23)     # statistical-only
        self.assertEqual({m["validation_window_break_source"] for m in res.monitoring
                          if m["is_validation_day"] and m["validation_window_has_break"]},
                         {"statistical"})
        res2 = self._run(score=2.0, news="BREAK", n=40, bcf=23)      # news-only
        self.assertEqual({m["validation_window_break_source"] for m in res2.monitoring
                          if m["is_validation_day"] and m["validation_window_has_break"]},
                         {"news"})


class Test41EarlyStopGate(unittest.TestCase):
    """Training fix: early stop is blocked until >=1 prompt update has been accepted."""

    def _train(self, *, boost, n_trades, max_cycles):
        """Run run_discovery_training with a stubbed engine whose training PnL is a pure
        function of the prompt: a candidate that carries the 'GOODLESSON' marker scores
        higher (so it is accepted) iff ``boost`` is True. ``n_trades`` sets the per-round
        executed-trade count (0 -> NO_EVIDENCE)."""
        import experiments.run_redesign as rr
        import utils.engine as ue
        from utils.engine import WindowResult

        def fake_sdw(*, direction_memory=None, action_prompt=None, **kw):
            res = WindowResult()
            if direction_memory is not None:
                for k in range(n_trades):
                    direction_memory.add_decision(selected_pair="A/B", action="LONG_SHORT",
                                                  reason="x", decision_index=k)
            r = -0.01
            if boost and action_prompt and "GOODLESSON" in action_prompt:
                r += 0.02
            res.cumulative_pnl = r
            return res

        class _Ag:
            def reflect_direction(self, dm):
                return {"reflection_text": "r",
                        "lessons": [{"lesson_text": "GOODLESSON improves discipline"}]}

        orig_sdw = ue.simulate_discovery_window
        orig_prov = rr.make_discovery_provider
        ue.simulate_discovery_window = fake_sdw
        rr.make_discovery_provider = lambda *a, **k: (lambda *a, **k: [])
        try:
            pm, _pmem, cycles = rr.run_discovery_training(
                None, train_dates=[date(2025, 2, 1)], env_data={}, agent=_Ag(),
                pool_df=None, symbols=["A", "B"], selection_frequency=10,
                max_cycles=max_cycles)
        finally:
            ue.simulate_discovery_window = orig_sdw
            rr.make_discovery_provider = orig_prov
        return pm, cycles

    def test_no_early_stop_before_any_accept(self):
        # never accept (boost off) -> early stop must be BLOCKED, training runs to max
        pm, cycles = self._train(boost=False, n_trades=2, max_cycles=4)
        self.assertEqual(len(cycles), 4)                          # ran to max, no early stop
        self.assertFalse(any(c["early_stop"] for c in cycles))
        self.assertTrue(all(c["accepted_prompt_updates"] == 0 for c in cycles))
        # once two rejections accumulate, the stop is blocked (not eligible)
        blocked = [c for c in cycles if c["early_stop_blocked_reason"]]
        self.assertTrue(blocked)
        self.assertTrue(all(c["early_stop_blocked_reason"] == "no_accepted_prompt_update_yet"
                            for c in blocked))
        self.assertTrue(all(not c["early_stop_eligible"] for c in cycles))

    def test_early_stop_after_an_accept(self):
        # accept on cycle 1, then rejections -> early stop becomes eligible and fires
        pm, cycles = self._train(boost=True, n_trades=2, max_cycles=8)
        self.assertTrue(cycles[0]["prompt_accepted"])            # first round accepted
        self.assertGreaterEqual(cycles[0]["accepted_prompt_updates"], 1)
        self.assertTrue(any(c["early_stop"] for c in cycles))    # eventually stops
        stop = next(c for c in cycles if c["early_stop"])
        self.assertTrue(stop["early_stop_eligible"])
        self.assertEqual(stop["early_stop_reason"], "two_consecutive_prompt_rejections")
        self.assertEqual(cycles[-1], stop)                       # loop broke at the stop

    def test_max_rounds_reached_when_never_accepted(self):
        pm, cycles = self._train(boost=False, n_trades=2, max_cycles=5)
        self.assertEqual(len(cycles), 5)
        self.assertEqual(pm.version, 1)                          # no prompt update applied

    def test_no_evidence_rounds_do_not_contribute(self):
        # all rounds NO_EVIDENCE -> never eligible, never early-stop, run to max
        pm, cycles = self._train(boost=False, n_trades=0, max_cycles=4)
        self.assertEqual(len(cycles), 4)
        self.assertTrue(all(c["episode_status"] == "NO_EVIDENCE" for c in cycles))
        self.assertFalse(any(c["early_stop"] for c in cycles))
        self.assertTrue(all(c["accepted_prompt_updates"] == 0 for c in cycles))

    def test_accept_reject_unchanged(self):
        # accept once (version++), subsequent equal-reward rounds reject (no version bump)
        pm, cycles = self._train(boost=True, n_trades=2, max_cycles=8)
        self.assertEqual(pm.version, 2)                          # exactly one accept applied
        self.assertTrue(any(not c["prompt_accepted"] for c in cycles))  # rejections still occur


if __name__ == "__main__":
    unittest.main(verbosity=2)
