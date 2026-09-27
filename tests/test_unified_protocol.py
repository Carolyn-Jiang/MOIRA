"""Offline integration checks for hosted transport and the ablation interventions."""
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx
from openai import OpenAI
from llm.backbone import build_llm, infer_family
from llm.errors import InferenceContractError
from utils.engine import simulate_discovery_window
from experiments.run_main import parse_args

class HostedTransport(unittest.TestCase):
    def test_real_sdk_requests_and_final_content_separation(self):
        for provider, model in [("openai", "gpt-4.1-mini"), ("openai", "o4-mini"),
                                ("deepseek", "deepseek-v4.1-flash"), ("gemini", "gemini-3.8-flash")]:
            with self.subTest(model=model), patch.dict(os.environ, {f"{provider.upper()}_API_KEY": "test-placeholder"}, clear=True):
                self.assertEqual(infer_family(model), provider)
                client = build_llm(provider, model, seed=1103)
                requests = []
                def reply(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(200, json={"id":"fixture", "object":"chat.completion", "created":0,
                        "model":model, "choices":[{"index":0,"finish_reason":"stop","message":{
                            "role":"assistant", "content":'{"answer":323}', "reasoning_content":"separate"}}],
                        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
                client._client = OpenAI(api_key="test-placeholder", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(reply)))
                proof = client.validate_server()
                self.assertTrue(proof["final_content_only"])
                request = requests[0]
                self.assertEqual(request["model"], model)
                self.assertIn("max_completion_tokens" if provider == "openai" else "max_tokens", request)
                self.assertNotIn("chat_template_kwargs", request)
                if provider == "deepseek": self.assertEqual(request["thinking"], {"type":"disabled"})
                if provider == "gemini": self.assertEqual(request["reasoning_effort"], "low")
                client._client.close()

    def test_credentials_do_not_cross_provider_boundaries(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY":"test-placeholder"}, clear=True):
            for provider in ("deepseek", "gemini"):
                with self.assertRaisesRegex(ValueError, "API_KEY"):
                    build_llm(provider, provider + "-test")
        for text in ("<think>hidden</think>{}", "", None):
            with patch.dict(os.environ, {"OPENAI_API_KEY":"test-placeholder"}, clear=True):
                client=build_llm("openai", "gpt-4.1-mini")
                response=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",message=SimpleNamespace(content=text))])
                with self.assertRaises(InferenceContractError): client._extract_final_content(response)

    def test_single_seed_and_mixed_backends_are_valid_plans(self):
        args=parse_args(["gpt-4.1-mini", "Llama-3.1-8B-Instruct", "--seeds", "1103", "--dry-run"])
        self.assertEqual(args.seeds, [1103]);self.assertEqual(args.families, ["openai","llama"])

class AblationInterventions(unittest.TestCase):
    def test_a4_preserves_monitoring_and_daily_actions_while_preventing_switches(self):
        from test_redesign import Test42LifecycleSimplified
        f=Test42LifecycleSimplified(); symbols=[f"P{i}" for i in range(10)]
        dates, env=f._env(symbols, 75)
        full=simulate_discovery_window(dates=dates, env_data=env, symbols=symbols,
            agent=f._agent(symbols,state="BREAK"), discovery_provider=f._switching_provider(),
            lifecycle=True, anti_flat_override=False)
        control=simulate_discovery_window(dates=dates, env_data=env, symbols=symbols,
            agent=f._agent(symbols,state="BREAK"), discovery_provider=f._switching_provider(),
            lifecycle=True, initial_pair=("P0","P1"), allow_reselection=False, anti_flat_override=False)
        self.assertEqual(len(full.switch_events),3)
        self.assertEqual([e["decision_index"] for e in full.switch_events],[23,46,69])
        self.assertEqual(len(control.switch_events),0)
        self.assertEqual(len(control.monitoring),len(full.monitoring))
        self.assertTrue(any(r["lifecycle_state"]=="BREAK" for r in control.monitoring))
        self.assertFalse(any(r["pair_reselection_triggered"] for r in control.monitoring))
        self.assertEqual(len(control.decisions),75)
        self.assertEqual(full.pair_selections[0],control.pair_selections[0])

    def test_a3_warmup_is_shared_without_claiming_gate_acceptance(self):
        from test_redesign import Test35DiscoveryTraining, _causal_pool
        from experiments.run_ablations import _warmup
        from llm.policy_agent import PolicyAgent
        symbols, env, dates, provider=Test35DiscoveryTraining()._setup()
        spec=dict(train_dates=dates[:12],env_data=env,symbols=symbols,pool_df=_causal_pool(n_years=2),profile_cache={})
        agent=PolicyAgent(backend="offline",symbols=symbols)
        with patch("experiments.run_redesign.make_discovery_provider", return_value=provider):
            prompt, pair=_warmup(agent,spec)
        self.assertEqual(prompt.accepted_lessons,[])
        self.assertEqual(prompt.rejected_lessons,[])
        self.assertIsInstance(prompt.active_prompt(),str)
        self.assertGreater(len(prompt.active_prompt()),0)

    def test_paired_control_changes_only_its_requested_switch(self):
        from experiments.run_ablations import run_cell, SETTINGS
        training={"full_prompt":"consolidated", "warmup_prompt":"warmup", "pair_records":[{"quality":"shared"}]}
        def fake_eval(setting, seed, spec, prompt, records, output, **kwargs):
            return {"initial_pair":["A","B"],"metrics":{},"prompt":prompt,"records":records,**kwargs}
        for ablation in ("a3","a4"):
            with TemporaryDirectory() as tmp, patch("experiments.run_ablations._training",return_value=training), \
                    patch("experiments.run_ablations._evaluate",side_effect=fake_eval) as evaluate:
                result=run_cell(SETTINGS["gpt41_beta"],1103,ablation,{},Path(tmp),8)
                full, control=list(result.values())
                self.assertEqual(full["records"],control["records"])
                if ablation=="a3":
                    self.assertEqual(control["prompt"],"warmup");self.assertTrue(control["allow_reselection"])
                else:
                    self.assertEqual(control["prompt"],full["prompt"])
                    self.assertEqual(control["initial_pair"],full["initial_pair"])
                    self.assertFalse(control["allow_reselection"])
                run_cell(SETTINGS["gpt41_beta"],1103,ablation,{},Path(tmp),8)
                self.assertEqual(evaluate.call_count,2)
