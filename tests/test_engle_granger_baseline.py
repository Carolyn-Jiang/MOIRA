import math
import unittest
from datetime import date, timedelta

from experiments.baselines import (
    LONG_SHORT,
    SHORT_LONG,
    engle_granger_zscore_action,
    simulate_engle_granger_zscore,
)


class TestEngleGrangerZscoreBaseline(unittest.TestCase):
    def test_state_machine_enters_holds_and_exits(self):
        self.assertEqual(engle_granger_zscore_action(-1.1, None), LONG_SHORT)
        self.assertEqual(engle_granger_zscore_action(-0.4, LONG_SHORT), LONG_SHORT)
        self.assertIsNone(engle_granger_zscore_action(-0.2, LONG_SHORT))
        self.assertEqual(engle_granger_zscore_action(1.1, None), SHORT_LONG)
        self.assertEqual(engle_granger_zscore_action(0.4, SHORT_LONG), SHORT_LONG)
        self.assertIsNone(engle_granger_zscore_action(0.2, SHORT_LONG))

    def test_invalid_thresholds_fail(self):
        with self.assertRaises(ValueError):
            engle_granger_zscore_action(0.0, None, entry_z=0.0)
        with self.assertRaises(ValueError):
            engle_granger_zscore_action(0.0, None, entry_z=1.0, exit_band=1.0)

    def test_simulation_is_deterministic_and_emits_daily_actions(self):
        dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(40)]
        env = {}
        for i, d in enumerate(dates):
            base = 100.0 + i * 0.2
            env[d] = {
                "price": {
                    "A": base + 3.0 * math.sin(i / 3.0),
                    "B": base,
                },
                "news": {},
            }
        kwargs = dict(
            dates=dates[20:],
            env_data=env,
            symbols=["A", "B"],
            pair=("A", "B"),
            history_seed={d: env[d] for d in dates[:20]},
        )
        first = simulate_engle_granger_zscore(**kwargs)
        second = simulate_engle_granger_zscore(**kwargs)
        self.assertEqual(first["actions"], second["actions"])
        self.assertEqual(first["normalized_equity"], second["normalized_equity"])
        self.assertEqual(len(first["actions"]), len(dates[20:]))
        self.assertTrue(set(first["actions"]) <= {"FLAT", LONG_SHORT, SHORT_LONG})


if __name__ == "__main__":
    unittest.main()
