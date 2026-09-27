"""Final settlement is a price-only valuation, never a 76th test decision."""
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.baselines import simulate_baseline_window
from experiments.main_artifacts import result_metrics
from experiments.run_redesign import SYMBOLS, split_train_test, _assign_realized_pnl
from llm.policy_agent import DirectionMemory
from utils.data_utils import (apply_cached_prices, filter_to_trading_days,
                             load_offline_dataset, load_pair_selection_price_history,
                             next_trading_day)
from utils.engine import simulate_discovery_window, write_discovery_logs
import test_redesign as fixtures


class TestTerminalSettlement(unittest.TestCase):
    def setup_run(self):
        symbols, synthetic, _, provider = fixtures.Test42DailyDirectionCadence()._env(75)
        dates = filter_to_trading_days(pd.date_range("2025-02-12", "2025-05-30").date)
        env = dict(zip(dates, synthetic.values()))
        settlement = date(2025, 6, 2)
        env[settlement] = {"price": {s: p * (1.10 if s in ("A", "C") else .93)
                                    for s, p in env[dates[-1]]["price"].items()}}
        return symbols, env, dates, provider, settlement

    def test_75_decisions_settle_without_news_or_additional_calls(self):
        symbols, env, dates, provider, settlement = self.setup_run()
        for beta in (False, True):
            with self.subTest(beta=beta):
                agent = fixtures.Test37DiscoveryLifecycle()._break_agent(symbols, status="HEALTHY")
                dm = DirectionMemory()
                kwargs = dict(dates=dates, env_data=env, symbols=symbols, agent=agent,
                              discovery_provider=provider, lifecycle=True, beta_hedged=beta,
                              lookback=10, direction_memory=dm)
                with patch.object(agent, "discover_daily_direction", wraps=agent.discover_daily_direction) as calls:
                    result = simulate_discovery_window(**kwargs, settlement_date=settlement)
                    self.assertEqual(calls.call_count, 75)
                self.assertEqual(len(result.records), 75)
                self.assertEqual(len(result.decisions), 75)
                self.assertEqual(len(result.normalized_equity), 76)
                self.assertEqual(result_metrics(result)["finalized_transitions"], 75)
                self.assertEqual(len(result.monitoring), 74)
                self.assertNotIn("news", env[settlement])
                self.assertNotIn(settlement.isoformat(), result.action_by_date)
                self.assertAlmostEqual(np.prod([1 + d["actual_pnl"] for d in result.decisions]),
                                       result.normalized_equity[-1], places=8)
                self.assertAlmostEqual(result.decisions[-1]["actual_pnl"],
                                       result.settlement["daily_pnl"], places=8)
                _assign_realized_pnl(dm, result, 23)
                with TemporaryDirectory() as tmp:
                    write_discovery_logs(result, tmp)
                    log = pd.read_csv(Path(tmp) / "decision_log.csv")
                    self.assertEqual(len(log), 75)
                    self.assertAlmostEqual(log.realized_pnl.iloc[-1],
                                           result.settlement["daily_pnl"], places=7)

    def test_terminal_prices_cannot_change_decisions(self):
        symbols, env, dates, provider, settlement = self.setup_run()
        agent = fixtures.Test37DiscoveryLifecycle()._break_agent(symbols, status="HEALTHY")
        kwargs = dict(dates=dates, env_data=env, symbols=symbols, agent=agent,
                      discovery_provider=provider, lifecycle=True)
        before = simulate_discovery_window(**kwargs)
        after = simulate_discovery_window(**kwargs, settlement_date=settlement)
        self.assertEqual(before.records, after.records)
        self.assertEqual(before.pair_lifecycle, after.pair_lifecycle)
        self.assertEqual(before.normalized_equity, after.normalized_equity[:-1])
        self.assertIsNone(before.decisions[-1].get("actual_pnl"))
        self.assertIsNotNone(after.decisions[-1].get("actual_pnl"))

    def test_baseline_values_last_position_without_reselecting(self):
        symbols, env, dates, _, settlement = self.setup_run()
        for d in dates:
            env[d]["price"] = {s: 100.0 for s in symbols}
        env[settlement]["price"] = {s: (110.0 if s in ("A", "C") else 93.0) for s in symbols}
        pair_calls = []
        def pair_for_index(i):
            pair_calls.append(i)
            return ("A", "B")
        sim = simulate_baseline_window(dates=dates, env_data=env, symbols=symbols,
                pair_for_index=pair_for_index, action="LONG_SHORT", settlement_date=settlement)
        self.assertEqual(pair_calls, list(range(75)))
        self.assertEqual(len(sim["normalized_equity"]), 76)
        self.assertEqual(len(sim["held"]), 75)
        self.assertAlmostEqual(sim["daily_pnls"][-1], .5 * (.10 - (-.07)))

    def test_missing_or_wrong_settlement_fails_before_model_call(self):
        symbols, env, dates, provider, settlement = self.setup_run()
        agent = fixtures.Test37DiscoveryLifecycle()._break_agent(symbols)
        kwargs = dict(dates=dates, env_data=env, symbols=symbols, agent=agent,
                      discovery_provider=provider)
        with patch.object(agent, "discover_direction") as calls:
            with self.assertRaises(ValueError):
                simulate_discovery_window(**kwargs, settlement_date=date(2025, 6, 3))
            del env[settlement]
            with self.assertRaises(ValueError):
                simulate_discovery_window(**kwargs, settlement_date=settlement)
            calls.assert_not_called()

    def test_bundled_prices_cover_27_train_75_test_and_june2(self):
        root = Path(__file__).resolve().parents[1]
        data = load_offline_dataset(SYMBOLS, news_dir=str(root / "data/news"))
        train, test = split_train_test(sorted(data))
        self.assertEqual((len(train), len(test)), (27, 75))
        self.assertEqual((test[0], test[-1]), (date(2025, 2, 12), date(2025, 5, 30)))
        terminal = next_trading_day(test[-1])
        self.assertEqual(terminal, date(2025, 6, 2))
        with patch("yfinance.download") as download:
            pool, _ = load_pair_selection_price_history(SYMBOLS,
                decision_dates=[*sorted(data), terminal], cache_dir=root / "data/cache")
            download.assert_not_called()
        aligned = apply_cached_prices(data, pool, SYMBOLS, terminal)
        self.assertEqual(len(aligned), 103)
        self.assertNotIn("news", aligned[terminal])
        for d in [test[0], test[-1], terminal]:
            self.assertEqual(aligned[d]["price"], pool.loc[pd.Timestamp(d)].to_dict())
        with self.assertRaises(ValueError):
            apply_cached_prices(data, pool.drop(pd.Timestamp(terminal)), SYMBOLS, terminal)


if __name__ == "__main__":
    unittest.main()
