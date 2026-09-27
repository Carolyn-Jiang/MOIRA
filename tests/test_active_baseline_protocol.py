import unittest
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.baselines import simulate_engle_granger_zscore
from experiments.eg_ddqn import EGDoubleDQN, _action_return, build_market_series
from experiments.prepare_finmem import pair_records, make_config
from strategies.types import Action
from utils.data_utils import filter_to_trading_days
from utils.portfolio import PortfolioManager, size_pair_position_fractional, effective_hedge_ratio


class TestActiveBaselineProtocol(unittest.TestCase):
    def setUp(self):
        self.dates = filter_to_trading_days(list(pd.date_range("2025-02-12", "2025-05-30").date))
        self.terminal = date(2025, 6, 2)
        self.env = {d: {"price": {"A": 100., "B": 200.}, "news": {"A": ["known news"]}}
                    for d in [*self.dates, self.terminal]}
        self.env[self.terminal]["price"]["A"] = 130.

    def test_eg_z_settles_last_of_75_decisions(self):
        with patch("experiments.baselines.compute_spread_z", return_value=(-2., 1.)):
            result = simulate_engle_granger_zscore(dates=self.dates, env_data=self.env,
                symbols=["A", "B"], pair=("A", "B"), settlement_date=self.terminal)
        self.assertEqual(len(result["actions"]), 75)
        self.assertEqual(len(result["normalized_equity"]), 76)
        self.assertAlmostEqual(result["normalized_equity"][-1], 1.1)

    def test_ddqn_settles_without_learning_or_terminal_action(self):
        agent = EGDoubleDQN(1103)
        before = {k: v.copy() for k, v in agent.online.params.items()}
        with patch.object(agent.online, "predict", return_value=np.array([[1., 0., 0.]])) as predict:
            result = agent.evaluate(self.dates, self.env, ["A", "B"], ("A", "B"),
                beta_hedged=False, settlement_date=self.terminal)
        self.assertEqual(predict.call_count, 75)
        self.assertEqual(len(result["normalized_equity"]), 76)
        self.assertAlmostEqual(result["cum_return"], .1)
        for key in before:
            np.testing.assert_array_equal(before[key], agent.online.params[key])

    def test_training_reward_matches_shared_engine_and_beta_fallback(self):
        prices, following = {"A": 100., "B": 300.}, {"A": 110., "B": 290.}
        for beta in (.25, 1., 1.7, -1.7, 8.):
            for index, action in enumerate((Action.long_short, Action.flat, Action.short_long)):
                pm = PortfolioManager(["A", "B"], initial_cash=1.)
                target = size_pair_position_fractional(prices, ("A", "B"), 1., action,
                                                      hedge_ratio=effective_hedge_ratio(beta))
                pm.step(prices, target)
                pm.step(following, {"A": 0., "B": 0.})
                self.assertAlmostEqual(_action_return(prices, following, ("A", "B"), index, beta),
                                       pm.normalized_equity_history[-1] - 1.)

    def test_future_price_change_cannot_change_earlier_features(self):
        before, _ = build_market_series(self.dates, self.env, ("A", "B"))
        self.env[self.dates[-1]]["price"]["A"] = 999.
        after, _ = build_market_series(self.dates, self.env, ("A", "B"))
        np.testing.assert_array_equal(before[:-1], after[:-1])

    def test_invalid_terminal_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_engle_granger_zscore(dates=self.dates, env_data=self.env,
                symbols=["A", "B"], pair=("A", "B"), settlement_date=self.dates[-1])

    def test_finmem_adapter_preserves_aligned_news_and_terminal(self):
        spec = dict(env_data=self.env, train_dates=self.dates[:2], test_dates=self.dates[2:],
                    settlement_date=self.terminal)
        records = pair_records(spec, ("A", "B"))
        self.assertEqual(records[self.dates[2].isoformat()]["news"], ["[A] known news"])
        self.assertEqual(records[self.terminal.isoformat()]["news"], [])
        self.assertAlmostEqual(records[self.terminal.isoformat()]["prices"], 110.)
        config = make_config(spec, ("A", "B"), "runs/test", 3301)
        self.assertEqual(config["chat_config"]["chat_model"], "gpt-4.1-mini")
        self.assertEqual(config["env_config"]["test_end_time"], "2025-06-02")
        self.assertEqual(config["meta_config"]["decision_end_time"], "2025-05-30")
