"""Offline tests for the Herculean ablation protocol."""

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from experiments.run_ablations import (
    ABLATIONS,
    ABLATION_DATASETS,
    SETTINGS,
    _cross_regime_dates,
    _split_dates,
)
from experiments.main_artifacts import result_metrics as summarize_result
from llm.policy_agent import PairSelectionMemory
from utils.engine import WindowResult
from utils.herculean import HERCULEAN_SYMBOLS, build_herculean_env


class TestHerculeanLoader(unittest.TestCase):
    def test_complete_panel_and_prior_day_news(self):
        dates = pd.bdate_range("2025-01-02", periods=30)
        prices = pd.DataFrame([
            {"symbol": symbol, "date": d, "adj_close": 100 + i}
            for i, d in enumerate(dates) for symbol in HERCULEAN_SYMBOLS
        ])
        news = pd.DataFrame([
            {"symbol": "AAPL", "date": dates[0], "highlights": "day-zero summary"},
            {"symbol": "AAPL", "date": dates[1], "highlights": "same-day unavailable"},
        ])
        env = build_herculean_env(prices, news)
        ds = sorted(env)
        self.assertEqual(len(ds), 30)
        self.assertNotIn("AAPL", env[ds[0]]["news"])
        self.assertIn("day-zero summary", env[ds[1]]["news"]["AAPL"])
        self.assertNotIn("same-day unavailable", env[ds[1]]["news"]["AAPL"])

    def test_split_is_28_plus_remainder(self):
        base = date(2025, 1, 1)
        env = {base + timedelta(days=i): {} for i in range(332)}
        train, test = _split_dates(env)
        self.assertEqual((len(train), len(test)), (28, 304))

    def test_cross_regime_window_is_fixed_and_complete(self):
        dates = pd.bdate_range("2025-05-20", periods=120).date
        selected = _cross_regime_dates({day: {} for day in dates})
        self.assertEqual(len(selected), 103)
        self.assertGreaterEqual(selected[0], date(2025, 6, 2))

    def test_cross_regime_window_rejects_truncated_data(self):
        dates = pd.bdate_range("2025-06-02", periods=102).date
        with self.assertRaisesRegex(ValueError, "A2 needs 103 sessions"):
            _cross_regime_dates({day: {} for day in dates})


class TestAblationContract(unittest.TestCase):
    def test_requested_settings_and_conditions_are_exact(self):
        self.assertEqual(SETTINGS["gpt41_beta"].model, "gpt-4.1-mini")
        self.assertTrue(SETTINGS["gpt41_beta"].beta_hedged)
        self.assertEqual(SETTINGS["gpt54_nobeta"].model, "gpt-5.4-mini")
        self.assertFalse(SETTINGS["gpt54_nobeta"].beta_hedged)
        self.assertEqual(set(ABLATIONS), {"a1", "a2", "a3", "a4"})
        self.assertEqual(
            ABLATION_DATASETS,
            {
                "a1": "herculean",
                "a2": "herculean_regime",
                "a3": "main",
                "a4": "main",
            },
        )

    def test_result_metrics_include_counterfactual_diagnostics(self):
        result = WindowResult(normalized_equity=[1.0, 1.1])
        result.decisions = [{
            "actual_pnl": 0.1,
            "final_executed_action": "LONG_SHORT",
            "cf_long_short": 0.1,
            "cf_short_long": -0.1,
            "cf_flat": 0.0,
        }]
        summary = summarize_result(result)
        self.assertAlmostEqual(summary["cumulative_return"], 0.1)
        self.assertEqual(summary["counterfactual_accuracy"], 1.0)
        self.assertEqual(summary["counterfactual_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
