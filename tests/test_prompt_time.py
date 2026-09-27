"""Calendar masking at the actual model boundary, without paid inference."""
import copy
from datetime import date
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from llm.backbone import build_llm
from llm.pair_trading_llm import PairTradingLLM
from llm.policy_agent import sanitize_reflection
from llm.prompt_time import PromptTime, TIME_INSTRUCTION
from utils.company_profiles import COMPANY_PROFILES, profiles_block
from test_reasoning_contract import response


DAYS = [date(2025, 2, 12), date(2025, 2, 13), date(2025, 2, 14)]


class TestPromptTime(unittest.TestCase):
    def test_nested_inputs_mask_dates_without_mutating_data_or_numbers(self):
        clock = PromptTime(DAYS)
        clock.set_step(DAYS[1])
        payload = {"date": DAYS[1], "price": 2024, "return": .0306,
                   "memory": [{"decision_date": "2025-02-12", "reason":
                               "February 12, 2025 news; Sep 2024 revenue"}],
                   "news": "2025/02/13 and 14 February 2025; 2024-12-31T12:35:00Z"}
        original = copy.deepcopy(payload)
        system, body = clock.prepare("Observation on 2025-02-13", payload)
        self.assertEqual(body["current_step"], "step 002")
        self.assertEqual(body["step"], "step 002")
        self.assertEqual(body["memory"][0]["decision_step"], "step 001")
        self.assertEqual(body["news"], "step 002 and step 003; step XXX")
        self.assertEqual(body["price"], 2024)
        self.assertEqual(body["return"], .0306)
        self.assertEqual(payload, original)
        self.assertNotRegex(body["memory"][0]["reason"], r"202[45]")
        self.assertEqual(clock.prepare(system, body), (system, body))

    def test_unmapped_history_does_not_collapse_records(self):
        masked = PromptTime(DAYS).value({"2024-12-30": 100, "2024-12-31": 110})
        self.assertEqual(masked, [{"step": "step XXX", "value": 100},
                                  {"step": "step XXX", "value": 110}])

    def test_company_context_keeps_revenue_values_but_not_fiscal_dates(self):
        original = copy.deepcopy(COMPANY_PROFILES)
        text = profiles_block(list(COMPANY_PROFILES))
        self.assertNotRegex(text, r"202[45]|FY202")
        self.assertIn("~$60.9B (reported fiscal period 1)", text)
        self.assertIn("~$130.5B (reported fiscal period 2)", text)
        self.assertEqual(COMPANY_PROFILES, original)

    def test_step_specific_lessons_are_not_transferable(self):
        self.assertFalse(sanitize_reflection("Repeat the winning action at step 003."))
        self.assertFalse(sanitize_reflection("At step XXX use the same action."))
        self.assertTrue(sanitize_reflection("Compare relative news with spread pressure."))

    def test_all_transports_and_prompt_tasks_send_and_log_the_masked_view(self):
        models = [("llama", "meta-llama/Llama-3.1-8B-Instruct", None),
                  ("qwen", "Qwen/Qwen3-4B", "off"),
                  ("qwen", "Qwen/Qwen3-4B", "on"),
                  ("openai", "gpt-4.1-mini", None),
                  ("deepseek", "deepseek-v4.1-flash", None),
                  ("gemini", "gemini-3.8-flash", None)]
        for provider, model, thinking in models:
            with self.subTest(model=model, thinking=thinking), TemporaryDirectory() as tmp:
                with patch.dict(os.environ, {f"{provider.upper()}_API_KEY": "test-placeholder"}, clear=True):
                    llm = build_llm(provider, model, log_dir=tmp, thinking_mode=thinking)
                llm._client = Mock()
                fields = {"reasoning_content": "separate test reasoning"} if thinking == "on" else {}
                llm._client.chat.completions.create.return_value = response(**fields)
                llm.set_prompt_calendar(DAYS)
                llm.set_prompt_step(DAYS[1])
                for task in ("pair_selection", "daily_direction", "policy_reflection",
                             "relationship_break", "critical_news_verify"):
                    llm.call_text(system_prompt="News from Feb 12, 2025", task_type=task,
                                  payload={"date": "2025-02-13", "memory": {"date": "2025-02-12"}})
                for call in llm._client.chat.completions.create.call_args_list:
                    messages = call.kwargs["messages"]
                    self.assertNotIn("2025", json.dumps(messages))
                    self.assertEqual(messages[0]["content"].count(TIME_INSTRUCTION), 1)
                    body = json.loads(messages[1]["content"])
                    self.assertEqual(body["current_step"], "step 002")
                    self.assertEqual(body["memory"]["step"], "step 001")
                for record in (Path(tmp) / "llm_calls.jsonl").read_text().splitlines():
                    record = json.loads(record)
                    self.assertNotIn("2025", json.dumps([record["prompt"], record["payload"]]))
                    self.assertEqual(record["payload"]["step"], "step 002")
                # Retries and repeated calls do not increment the step; replay resets it.
                llm.set_prompt_calendar(DAYS)
                llm.set_prompt_step(DAYS[0])
                self.assertEqual(llm.prompt_time.current_step, "step 001")

    def test_legacy_transport_entrypoint_also_masks_price_history(self):
        llm = PairTradingLLM(model="test")
        llm._client = Mock()
        llm._client.chat.completions.create.return_value = response('{"pair":["A","B"]}')
        llm.set_prompt_calendar(DAYS)
        llm.set_prompt_step(DAYS[1])
        llm.select_pair(symbols=["A", "B"], recent_prices={DAYS[0]: {"A": 10, "B": 20}})
        body = json.loads(llm._client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(body["recent_prices"], [{"step": "step 001", "prices": {"A": 10., "B": 20.}}])

    def test_engine_uses_one_step_per_decision_and_preserves_results(self):
        from test_redesign import Test42LifecycleSimplified
        from utils.engine import simulate_discovery_window
        f = Test42LifecycleSimplified()
        symbols = [f"P{i}" for i in range(10)]
        dates, env = f._env(symbols, 75)
        agent = f._agent(symbols, state="BREAK")
        original = copy.deepcopy(env)
        observed = []
        llm = PairTradingLLM(model="offline-fixture")
        agent.llm = llm
        original_direction = agent.discover_daily_direction
        def decision(**kwargs):
            observed.append(llm.prompt_time.current_step)
            return original_direction(**kwargs)
        agent.discover_daily_direction = decision
        kwargs = dict(dates=dates, env_data=env, symbols=symbols, lifecycle=True, anti_flat_override=False)
        masked = simulate_discovery_window(agent=agent, discovery_provider=f._switching_provider(), **kwargs)
        control = simulate_discovery_window(agent=f._agent(symbols, state="BREAK"),
                                            discovery_provider=f._switching_provider(), **kwargs)
        self.assertEqual(observed, [f"step {i:03d}" for i in range(1, 76)])
        self.assertEqual(masked.decisions, control.decisions)
        self.assertEqual(masked.switch_events, control.switch_events)
        self.assertEqual(env, original)
        self.assertEqual(masked.decisions[0]["date"], dates[0].isoformat())


if __name__ == "__main__":
    unittest.main()
