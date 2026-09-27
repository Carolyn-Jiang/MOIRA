"""Ten-year pair-selection windows, cache coverage and acquisition boundaries."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.data_utils import (load_pair_selection_price_history,
                              pair_history_bounds, validate_pair_selection_history)
from utils.statarb import discover_pairs_coint10y
from experiments.run_redesign import causal_horizon_dfs, causal_profiles
from experiments.baselines import select_coint_pair


class TestTenYearHistory(unittest.TestCase):
    dates = [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-05-30")]

    def prices(self, start="2015-01-02", end="2025-05-30", symbol="A"):
        index = pd.bdate_range(start, end)
        return pd.DataFrame({symbol: 100. + np.arange(len(index))*.01}, index=index)

    def test_bounds_are_anchored_to_experiment_not_wall_clock(self):
        start, end = pair_history_bounds(reversed(self.dates))
        self.assertEqual(start, pd.Timestamp("2015-01-02"))
        self.assertEqual(end, pd.Timestamp("2025-05-30"))

    def test_only_ten_year_window_and_no_future_or_older_prices(self):
        pool = self.prices("2010-01-01", "2026-01-01")
        windows = causal_horizon_dfs(pool, self.dates[0])
        self.assertEqual(set(windows), {"10Y"})
        pd.testing.assert_frame_equal(windows["10Y"], pool.loc["2015-01-02":"2025-01-01"])
        changed = pool.copy()
        changed.loc[changed.index < "2015-01-02", "A"] = -1e6
        changed.loc[changed.index >= "2025-01-02", "A"] = 1e6
        pd.testing.assert_frame_equal(causal_horizon_dfs(changed, self.dates[0])["10Y"], windows["10Y"])

    def test_ten_year_window_rolls_with_every_selection_date(self):
        pool = self.prices("2010-01-01", "2026-09-18")
        for day in ("2025-01-02", "2025-02-12", "2025-03-18", "2025-04-21", "2025-05-22"):
            ts = pd.Timestamp(day)
            expected = pool[(pool.index >= ts - pd.DateOffset(years=10)) & (pool.index < ts)]
            actual = causal_horizon_dfs(pool, ts)["10Y"]
            pd.testing.assert_frame_equal(actual, expected)
            self.assertLess(actual.index.max(), ts)

    def test_main_and_fixed_baseline_share_the_ten_year_slice(self):
        pool = self.prices("2010-01-01", "2026-01-01")
        with patch("experiments.run_redesign.rank_multihorizon_profiles", return_value=[]) as main_rank:
            causal_profiles(pool, ["A"], self.dates[0])
        with patch("experiments.baselines.rank_multihorizon_profiles", return_value=[]) as baseline_rank, \
                patch("experiments.baselines.discover_pairs_coint10y", return_value=[(SimpleNamespace(pair=("A", "B")), .01)]):
            self.assertEqual(select_coint_pair(pool, ["A"], self.dates[0]), ("A", "B"))
        self.assertEqual(set(main_rank.call_args.args[0]), {"10Y"})
        self.assertEqual(set(baseline_rank.call_args.args[0]), {"10Y"})
        pd.testing.assert_frame_equal(main_rank.call_args.args[0]["10Y"], baseline_rank.call_args.args[0]["10Y"])

    def test_no_other_horizon_substitutes_for_missing_ten_year_metrics(self):
        profile = SimpleNamespace(horizons={"1Y": SimpleNamespace(coint_pvalue=.001)},
                                  horizon_metrics=lambda name: None)
        selected, diagnostics = discover_pairs_coint10y([profile], return_diagnostics=True)
        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["total_candidates"], 0)

    def test_valid_cache_needs_no_download_and_excludes_future(self):
        with TemporaryDirectory() as tmp:
            self.prices("2010-01-01", "2026-06-01").to_csv(Path(tmp) / "pool_A_10y.csv", index_label="date")
            with patch("yfinance.download") as download:
                result, source = load_pair_selection_price_history(["A"], decision_dates=self.dates, cache_dir=tmp)
            download.assert_not_called()
            self.assertEqual(source, "cache_10y")
            self.assertEqual(result.index.min(), pd.Timestamp("2015-01-02"))
            self.assertEqual(result.index.max(), pd.Timestamp("2025-05-30"))

    def test_current_date_ten_year_cache_is_refreshed_for_historical_experiment(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool_A_10y.csv"
            self.prices("2016-06-02", "2026-06-01").to_csv(path, index_label="date")
            raw = self.prices().rename(columns={"A": "Close"})
            with patch("yfinance.download", return_value=raw) as download:
                frame, source = load_pair_selection_price_history(["A"], decision_dates=self.dates, cache_dir=tmp)
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["start"], "2015-01-02")
            self.assertEqual(kwargs["end"], "2025-05-31")
            self.assertNotIn("period", kwargs)
            self.assertEqual(source, "yfinance_10y_refreshed")
            self.assertEqual(frame.index.min(), pd.Timestamp("2015-01-02"))
            self.assertEqual(pd.read_csv(path).date.iloc[0], "2015-01-02")

    def test_download_failure_does_not_use_short_cache_or_overwrite_it(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool_A_10y.csv"
            self.prices("2016-06-02", "2026-06-01").to_csv(path, index_label="date")
            original = path.read_bytes()
            with patch("yfinance.download", return_value=pd.DataFrame()), self.assertRaisesRegex(RuntimeError, "No shorter horizon"):
                load_pair_selection_price_history(["A"], decision_dates=self.dates, cache_dir=tmp)
            self.assertEqual(path.read_bytes(), original)

    def test_twenty_year_cache_is_not_a_substitute(self):
        with TemporaryDirectory() as tmp:
            self.prices("2005-01-01", "2026-01-01").to_csv(Path(tmp) / "pool_A_20y.csv", index_label="date")
            with patch("yfinance.download", return_value=self.prices().rename(columns={"A": "Close"})) as download:
                load_pair_selection_price_history(["A"], decision_dates=self.dates, cache_dir=tmp)
            download.assert_called_once()
            self.assertTrue((Path(tmp) / "pool_A_10y.csv").exists())

    def test_snapshots_reject_short_missing_invalid_and_duplicate_data(self):
        complete = self.prices()
        missing = complete.drop(self.dates[0])
        invalid = complete.copy()
        invalid.loc[self.dates[1], "A"] = np.inf
        for frame in [self.prices("2016-06-02"), missing, invalid,
                      pd.concat([complete, complete.iloc[:1]])]:
            with self.subTest(rows=len(frame)), self.assertRaises(ValueError):
                validate_pair_selection_history(frame, ["A"], iter(self.dates))

    def test_market_holiday_at_start_is_allowed_without_years_of_missing_data(self):
        frame = self.prices("2015-01-05")
        result = validate_pair_selection_history(frame, ["A"], self.dates)
        self.assertEqual(result.index.min(), pd.Timestamp("2015-01-05"))
        with self.assertRaises(ValueError):
            validate_pair_selection_history(self.prices("2015-02-02"), ["A"], self.dates)

    def test_multiindex_yfinance_columns_are_supported(self):
        raw = self.prices().rename(columns={"A": "Close"})
        raw.columns = pd.MultiIndex.from_tuples([("Close", "A")])
        with TemporaryDirectory() as tmp, patch("yfinance.download", return_value=raw):
            frame, _ = load_pair_selection_price_history(["A"], decision_dates=self.dates, cache_dir=tmp)
        self.assertEqual(list(frame), ["A"])


if __name__ == "__main__":
    unittest.main()
