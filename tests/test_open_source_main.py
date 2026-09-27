"""Transport contracts and a real engine/artifact/plot integration with synthetic data."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from llm.backbone import build_llm
from llm.errors import InferenceContractError
from experiments.main_artifacts import model_slug, run_directory


class TestOpenSourceTransport(unittest.TestCase):
    def test_llama_and_qwen_requests_and_credentials(self):
        for family, model in [("llama", "meta-llama/Llama-3.1-8B-Instruct"),
                              ("qwen", "Qwen/Qwen2.5-7B-Instruct")]:
            with self.subTest(family=family), patch.dict(os.environ, {
                f"{family.upper()}_MODEL": model,
                f"{family.upper()}_BASE_URL": "http://localhost:8123/v1",
                "OPENAI_API_KEY": "commercial-key-must-not-leak",
                "OPENAI_BASE_URL": "https://unrelated.invalid/v1",
            }, clear=True):
                llm = build_llm(family, seed=0)
                self.assertEqual(llm.model, model)
                self.assertEqual(llm.api_key, "local-no-key")
                self.assertEqual(llm.provider, family)
                client = Mock()
                client.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                        content='{"action":"FLAT"}'))],
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
                    system_fingerprint="test-server")
                with patch("openai.OpenAI", return_value=client) as factory:
                    text = llm.call_text(system_prompt="Return JSON", payload={"test": True})
                    request = client.chat.completions.create.call_args.kwargs
                    self.assertEqual(request["seed"], 0)
                    self.assertEqual(request["model"], model)
                    self.assertEqual(request["max_tokens"], 4096)
                    self.assertNotIn("temperature", request)
                    self.assertEqual(json.loads(text), {"action": "FLAT"})
                    self.assertEqual(factory.call_args.kwargs["base_url"], "http://localhost:8123/v1")
                    self.assertEqual(factory.call_args.kwargs["api_key"], "local-no-key")
                self.assertEqual(llm.usage_summary()["total_tokens"], 20)
                self.assertEqual(llm.usage_summary()["seed"], 0)

    def test_family_credentials_override_shared_credentials(self):
        with patch.dict(os.environ, {"QWEN_MODEL": "qwen-test", "QWEN_API_KEY": "family-key",
                                   "QWEN_BASE_URL": "http://localhost:8123/v1",
                                   "OPEN_SOURCE_API_KEY": "shared-key"}, clear=True):
            llm = build_llm("qwen", seed=42)
            self.assertEqual(llm.api_key, "family-key")
            self.assertNotIn("family-key", json.dumps(llm.inference_config()))

    def test_seed_opt_out_and_extra_parameters(self):
        with patch.dict(os.environ, {"LLM_MODEL": "Qwen/Qwen3-4B", "OPEN_SOURCE_BASE_URL": "http://localhost/v1",
                                   "OPEN_SOURCE_SEND_SEED": "0",
                                   "OPEN_SOURCE_EXTRA_BODY": '{"chat_template_kwargs":{"enable_thinking":false}}'},
                        clear=True):
            llm = build_llm("open_source", seed=99)
            request = llm._completion_request([])
            self.assertNotIn("seed", request)
            self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertFalse(llm.usage_summary()["seed_sent"])

    def test_invalid_or_missing_configuration(self):
        for env in [{},
                    {"LLM_MODEL": "test", "OPEN_SOURCE_BASE_URL": "http://user:secret@host/v1"},
                    {"LLM_MODEL": "test", "OPEN_SOURCE_BASE_URL": "http://localhost/v1",
                     "OPEN_SOURCE_EXTRA_BODY": '{"seed":999}'},
                    {"LLM_MODEL": "test", "OPEN_SOURCE_BASE_URL": "http://localhost/v1",
                     "OPEN_SOURCE_TIMEOUT": "nan"}]:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit):
                    build_llm("open_source")

    def test_failed_response_is_counted(self):
        with patch.dict(os.environ, {"LLM_MODEL": "test", "OPEN_SOURCE_BASE_URL": "http://localhost/v1"},
                        clear=True):
            llm = build_llm("qwen", seed=1)
            llm._client = Mock()
            llm._client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="<think>unfinished"))],
                usage=None, system_fingerprint=None)
            with tempfile.TemporaryDirectory() as tmp:
                llm.log_dir = tmp
                with self.assertRaises(InferenceContractError):
                    llm.call_text(system_prompt="JSON", payload={})
            self.assertEqual(llm.usage_summary()["failed_call_count"], 1)


class TestSeededMain(unittest.TestCase):
    def test_slug_is_safe_and_collision_resistant(self):
        self.assertNotIn("/", model_slug("../../Qwen/model"))
        self.assertNotEqual(model_slug("a/b"), model_slug("a-b"))

    def test_no_implicit_plot_selection(self):
        from experiments.plot_main_results import main
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--models", "test"])

    def test_runner_saves_seeds_without_figures_then_plots_only_selected_seed(self):
        import experiments.run_redesign as rr
        from experiments.plot_main_results import load_run, main as plot_main
        # Reuse the existing causal test fixture. Real training, tests, portfolio,
        # metrics, baselines and file writers execute; only data acquisition is mocked.
        from test_redesign import Test35DiscoveryTraining, _causal_pool
        symbols, env, _, provider = Test35DiscoveryTraining()._setup()
        pool = _causal_pool(n_years=2)
        settings = {"LLM_BACKEND": "offline", "LLM_MAX_CYCLES": "1", "LLM_LOOKBACK": "5"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, settings, clear=True), \
                patch.object(rr, "SYMBOLS", symbols), patch.object(rr, "TRAIN_SESSIONS", 12), \
                patch.object(rr, "TEST_SESSIONS", 18), \
                patch.object(rr, "load_offline_dataset", return_value=env), \
                patch.object(rr, "load_pair_selection_price_history", return_value=(pool, "synthetic-test")), \
                patch.object(rr, "make_discovery_provider", return_value=provider), \
                patch("experiments.baselines.select_coint_pair", return_value=("A", "B")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            roots = [rr.run_redesign_experiment(seed=s, output_root=tmp, sizing="both") for s in (7, 9)]
            self.assertFalse(list(Path(tmp).rglob("*.png")))
            self.assertIn('"seed": 7', stdout.getvalue())
            self.assertIn('"sharpe_ratio":', stdout.getvalue())
            for root, seed in zip(roots, (7, 9)):
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["seed"], seed)
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(manifest["config"]["pair_history_years"], 10)
                self.assertEqual(manifest["config"]["selection_frequency"], 23)
                self.assertEqual(manifest["config"]["break_check_frequency"], 23)
                self.assertEqual(manifest["config"]["pair_selection_rule"], "validation_window_break_only")
                metrics = json.loads((root / "metrics.json").read_text())
                self.assertEqual({m["sizing"] for m in metrics}, {"nobeta", "beta"})
                for mode in ("nobeta", "beta"):
                    decisions = pd.read_csv(root / f"logs/testing/decision_log_{mode}.csv")
                    selected = decisions.loc[decisions.is_pair_selection_day, "date"].tolist()
                    life = pd.read_csv(root / f"logs/testing/lifecycle_log_{mode}.csv")
                    triggered = life.loc[life.pair_reselection_triggered, "date"].tolist()
                    self.assertEqual(selected, [manifest["test_dates"][0], *triggered])
                    self.assertTrue(set(triggered) <= set(manifest["test_dates"][23::23]))
                    eq = pd.read_csv(root / f"equity_{mode}.csv")
                    m = next(m for m in metrics if m["sizing"] == mode)
                    self.assertAlmostEqual(eq.iloc[-1, 1], m["final_value"])
                    cf = pd.read_csv(root / f"logs/testing/counterfactual_pnl_test_{mode}.csv")
                    self.assertEqual(len(decisions), 18)
                    self.assertEqual(len(cf), 18)
                    self.assertEqual(len(eq), 19)
                    self.assertEqual(eq.date.tolist(), manifest["valuation_dates"])
                    self.assertEqual(eq.date.iloc[-1], manifest["settlement_date"])
                    self.assertAlmostEqual(decisions.realized_pnl.iloc[-1], cf.actual_pnl.iloc[-1], places=6)
                    self.assertAlmostEqual(float((1+cf.actual_pnl).prod()), m["final_value"], places=6)
                    self.assertTrue((root / f"logs/testing/lifecycle_log_{mode}.csv").exists())
            original = (roots[0] / "metrics.json").read_bytes()
            with self.assertRaises(FileExistsError):
                rr.run_redesign_experiment(seed=7, output_root=tmp)
            self.assertEqual(original, (roots[0] / "metrics.json").read_bytes())
            figures = Path(tmp) / "selected-figures"
            plot_main(["--runs-dir", tmp, "--provider", "offline", "--models", "offline-deterministic",
                       "--seeds", "9", "--output-dir", str(figures)])
            selection = json.loads((figures / "selection.json").read_text())
            self.assertEqual(selection["selected"][0]["seed"], 9)
            self.assertEqual(len(selection["selected"]), 1)
            self.assertEqual(len(list(figures.glob("*.png"))), 25)
            self.assertFalse(list(roots[0].rglob("*.png")))
            missing_out = Path(tmp) / "should-not-exist"
            with self.assertRaises(FileNotFoundError):
                plot_main(["--runs-dir", tmp, "--provider", "offline", "--models", "offline-deterministic",
                           "--seeds", "100", "--output-dir", str(missing_out)])
            self.assertFalse(missing_out.exists())
            (roots[0] / "manifest.json").write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                load_run(roots[0], ["nobeta"])

    def test_failed_run_manifest_is_preserved(self):
        import experiments.run_redesign as rr
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"LLM_BACKEND": "offline"}, clear=True), \
                patch.object(rr, "_run_redesign_experiment", side_effect=RuntimeError("test failure")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                rr.run_redesign_experiment(seed=17, output_root=tmp)
            path = run_directory(tmp, "offline", "offline-deterministic", 17)
            self.assertEqual(json.loads((path / "manifest.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
