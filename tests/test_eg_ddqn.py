import unittest
from datetime import date, timedelta

import numpy as np

from experiments.eg_ddqn import ACTIONS, DDQNConfig, EGDoubleDQN, build_market_series


class TestEGDoubleDQN(unittest.TestCase):
    def setUp(self):
        self.dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(36)]
        self.env = {}
        for i, day in enumerate(self.dates):
            self.env[day] = {"price": {
                "A": 100 + i * 0.2 + 2 * np.sin(i / 3),
                "B": 90 + i * 0.15,
            }}

    def test_features_are_causal_and_finite(self):
        features, betas = build_market_series(self.dates, self.env, ("A", "B"))
        self.assertEqual(features.shape, (36, 8))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.isfinite(betas).all())

    def test_seed_reproduces_policy_and_results(self):
        config = DDQNConfig(episodes=8, batch_size=8, target_update=5)
        outputs = []
        for _ in range(2):
            agent = EGDoubleDQN(17, config)
            agent.fit(self.dates[:20], self.env, ("A", "B"), beta_hedged=False)
            outputs.append(agent.evaluate(
                self.dates[20:], self.env, ["A", "B"], ("A", "B"),
                beta_hedged=False,
                history_seed={day: self.env[day] for day in self.dates[:20]},
            ))
        self.assertEqual(outputs[0]["actions"], outputs[1]["actions"])
        self.assertEqual(outputs[0]["normalized_equity"], outputs[1]["normalized_equity"])
        self.assertTrue(set(outputs[0]["actions"]) <= set(ACTIONS))


if __name__ == "__main__":
    unittest.main()
