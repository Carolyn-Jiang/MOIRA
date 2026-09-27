"""Exercise news delivery from publication-date inputs to actual agent calls."""
import copy
import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from llm.policy_agent import PolicyAgent
from utils.data_utils import filter_to_trading_days, load_offline_dataset
from utils.engine import simulate_discovery_window
from utils.herculean import build_herculean_env


SYMBOLS = ["A", "B"]
CALENDAR = list(pd.date_range("2025-01-02", "2025-03-07").date)
SESSIONS = filter_to_trading_days(CALENDAR)


def load_fixture(loader, publications):
    if loader == "main":
        with TemporaryDirectory() as directory:
            for symbol in SYMBOLS:
                rows = {d.isoformat(): {
                    "prices": 100. + i,
                    "news": publications.get(d, {}).get(symbol, []),
                } for i, d in enumerate(CALENDAR)}
                (Path(directory) / f"{symbol}_market_data.json").write_text(json.dumps(rows))
            return load_offline_dataset(SYMBOLS, news_dir=directory)
    prices = pd.DataFrame([
        {"symbol": s, "date": d, "adj_close": 100. + i}
        for i, d in enumerate(SESSIONS) for s in SYMBOLS
    ])
    news = pd.DataFrame([
        {"symbol": s, "date": d, "highlights": item}
        for d, by_symbol in publications.items()
        for s, items in by_symbol.items() for item in items
    ], columns=["symbol", "date", "highlights"])
    return build_herculean_env(prices, news, SYMBOLS)


def capture_agent_calls(env, decisions):
    llm = Mock(model="news-delivery-fixture")
    requests = []

    def reply(*, system_prompt, payload, task_type):
        requests.append((task_type, copy.deepcopy(payload)))
        if task_type == "pair_selection":
            return '{"selected_pair":["A","B"],"pair_selection_reason":"fixture"}'
        if task_type == "daily_direction":
            return '{"action":"FLAT","direction_reason":"fixture"}'
        return '{"news_break_level":"HEALTHY","reason":"fixture"}'

    llm.call_text.side_effect = reply
    agent = PolicyAgent(backend="open_source", llm=llm, symbols=SYMBOLS)
    pc = SimpleNamespace(coint_pvalue=.001, half_life=10., current_z=0.,
                         hedge_ratio=1., spread_vol=.1, adf_pvalue=.01,
                         spread_mean=0., spread_std=1.)
    with patch.object(agent, "detect_relationship_break",
                      wraps=agent.detect_relationship_break) as monitor:
        simulate_discovery_window(
            dates=decisions, env_data=env, symbols=SYMBOLS, agent=agent,
            discovery_provider=lambda *args: [{"pair": ("A", "B"), "pc": pc}],
            lifecycle=True, anti_flat_override=False,
        )
        monitoring = [copy.deepcopy(c.kwargs["news_summary"])
                      for c in monitor.call_args_list]
    return requests, monitoring


class TestNewsDelivery(unittest.TestCase):
    def test_prior_session_weekend_and_holiday_news_reach_model(self):
        # Ordinary weekday, Monday, and Tuesday after the MLK holiday.
        for loader in ("main", "herculean"):
            for decision, previous, count in (
                (date(2025, 1, 16), date(2025, 1, 15), 1),
                (date(2025, 2, 10), date(2025, 2, 7), 3),
                (date(2025, 1, 21), date(2025, 1, 17), 4),
            ):
                with self.subTest(loader=loader, decision=decision):
                    publications = {
                        d: {"A": [f"profit growth item {d}"]}
                        for d in CALENDAR if previous <= d < decision
                    }
                    # Same-day/future negative evidence must not cancel past positives.
                    publications[decision] = {"A": ["loss decline " * 20]}
                    publications[decision + timedelta(days=1)] = {"A": ["bankruptcy"]}
                    env = load_fixture(loader, publications)
                    self.assertEqual(len(env[decision]["news"]["A"]), count)
                    original = copy.deepcopy(env)
                    next_session = min(d for d in env if d > decision)
                    requests, monitoring = capture_agent_calls(env, [previous, decision, next_session])
                    daily = [p for task, p in requests if task == "daily_direction"]
                    self.assertEqual(daily[1]["active_candidate"]["news_A_summary"]["sentiment"],
                                     "positive")
                    self.assertEqual(monitoring[0]["A"]["n_items"], count)
                    self.assertFalse(monitoring[0]["A"]["critical"])
                    self.assertEqual(daily[2]["active_candidate"]["news_A_summary"]["sentiment"],
                                     "negative")
                    self.assertEqual(monitoring[1]["A"]["n_items"], 1)
                    self.assertEqual(env, original)
                    # A test slice must retain prior training/gap news on initial selection.
                    initial, _ = capture_agent_calls(env, [decision])
                    selection = next(p for task, p in initial if task == "pair_selection")
                    self.assertEqual(selection["top5_candidates"][0]["news_A_summary"]["sentiment"],
                                     "positive")

    def test_changing_current_and_future_publications_cannot_change_prior_requests(self):
        decision = date(2025, 1, 21)
        publications = {date(2025, 1, 17): {"A": ["profit growth"]}}
        for loader in ("main", "herculean"):
            with self.subTest(loader=loader):
                before = capture_agent_calls(load_fixture(loader, publications), [decision])
                changed = copy.deepcopy(publications)
                for d in CALENDAR:
                    if d >= decision:
                        changed[d] = {"A": ["bankruptcy loss decline"], "B": ["profit growth"]}
                after = capture_agent_calls(load_fixture(loader, changed), [decision])
                self.assertEqual(before, after)

    def test_empty_session_does_not_repeat_stale_news(self):
        for loader in ("main", "herculean"):
            with self.subTest(loader=loader):
                env = load_fixture(loader, {date(2025, 1, 17): {"A": ["profit growth"]}})
                requests, monitoring = capture_agent_calls(
                    env, [date(2025, 1, 21), date(2025, 1, 22)])
                daily = [p for task, p in requests if task == "daily_direction"]
                self.assertEqual(daily[0]["active_candidate"]["news_A_summary"]["sentiment"],
                                 "positive")
                self.assertEqual(daily[1]["active_candidate"]["news_A_summary"]["sentiment"],
                                 "neutral")
                self.assertEqual(monitoring[0]["A"]["n_items"], 0)


if __name__ == "__main__":
    unittest.main()
