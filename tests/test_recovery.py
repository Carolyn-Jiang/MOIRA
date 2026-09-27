"""Exercise retries, real saved-artifact recovery, and snapshot/lock boundaries."""
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import pandas as pd
from openai import APIConnectionError, RateLimitError, AuthenticationError

from experiments.main_artifacts import run_directory, write_json
from experiments.run_main import main
from experiments.recovery import batch_lock, summary_valid
from llm.backbone import build_llm
from llm.errors import InferenceContractError, InferenceServiceError, TransientInferenceError
from test_reasoning_contract import response


def fake_client():
    fake = Mock()
    fake.max_output_tokens = 16384
    fake._get_client.return_value.models.list.return_value.data = [SimpleNamespace(id="Qwen/Qwen3-4B")]
    fake.inference_config.return_value = {"thinking_mode": "off"}
    fake.validate_server.return_value = {"status": "passed"}
    fake.usage_summary.return_value = {"total_tokens": 25}
    return fake


def successful_seed(path, seed, env):
    """Saved API-independent test fixture; production load_run validates these files."""
    from experiments.run_redesign import _code_hash
    dates = pd.bdate_range("2025-02-12", periods=75).strftime("%Y-%m-%d").tolist()
    valuation = dates + ["2025-06-02"]
    identity = {"provider": "qwen", "model": "Qwen/Qwen3-4B", "seed": seed}
    write_json(path / "manifest.json", {**identity, "status": "completed", "sizing": ["beta"],
               "code_hash": _code_hash(), "test_dates": dates, "valuation_dates": valuation,
               "git_rev": env["MAIN_CODE_REVISION"], "inference": {"thinking_mode": "off"}})
    write_json(path / "metrics.json", [{**identity, "sizing": "beta", "final_value": 1.,
               "cumulative_return": 0., "finalized_transitions": 75}])
    write_json(path / "logs/training/direction_training_cycles.json", [])
    write_json(path / "comparison_beta.json", [{"strategy": "fixture", "group": "main"}])
    pd.DataFrame({"date": valuation, "fixture": [1.] * 76}).to_csv(path / "equity_beta.csv", index=False)
    logs = path / "logs/testing"
    logs.mkdir(parents=True)
    for name in ("decision_log_beta", "counterfactual_pnl_test_beta", "lifecycle_log_beta"):
        pd.DataFrame({"date": dates}).to_csv(logs / f"{name}.csv", index=False)


def fake_summary(batch):
    output = Path(batch) / "summary"
    output.mkdir()
    (output / "all_seeds.csv").write_text("test_fixture\n5\n")


class TestRequestRetry(unittest.TestCase):
    def llm(self):
        with patch.dict(os.environ, {}, clear=True):
            llm = build_llm("qwen", "Qwen/Qwen3-4B", thinking_mode="off")
        llm._client = Mock()
        return llm

    def test_transient_retry_waits_120_and_preserves_prompt_and_seed(self):
        llm = self.llm()
        error = APIConnectionError(request=httpx.Request("POST", "http://fixture/v1"))
        llm._client.chat.completions.create.side_effect = [error, response()]
        with patch("llm.open_source_llm.time.sleep") as sleep:
            self.assertEqual(llm.validate_server()["status"], "passed")
        sleep.assert_called_once_with(120)
        calls = llm._client.chat.completions.create.call_args_list
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(llm.failed_call_count, 1)

    def test_429_exhaustion_is_fatal_to_policy_not_a_fallback(self):
        llm = self.llm()
        reply = httpx.Response(429, request=httpx.Request("POST", "http://fixture/v1"))
        llm._client.chat.completions.create.side_effect = RateLimitError("busy", response=reply, body=None)
        with patch("llm.open_source_llm.time.sleep") as sleep, self.assertRaises(TransientInferenceError):
            llm.validate_server()
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(llm._client.chat.completions.create.call_count, 3)

    def test_authentication_and_reasoning_failures_are_never_retried(self):
        for error in (AuthenticationError("bad key", response=httpx.Response(
                401, request=httpx.Request("POST", "http://fixture/v1")), body=None),
                      InferenceContractError("bad reasoning")):
            llm = self.llm()
            llm._client.chat.completions.create.side_effect = error
            with patch("llm.open_source_llm.time.sleep") as sleep, \
                    self.assertRaises((InferenceServiceError, InferenceContractError)):
                llm.validate_server()
            sleep.assert_not_called()
            self.assertEqual(llm._client.chat.completions.create.call_count, 1)


class TestBatchRecovery(unittest.TestCase):
    def args(self, batch, *extra):
        return ["Qwen/Qwen3-4B", "--thinking", "off", "--sizing", "beta",
                "--output-dir", str(batch), *extra]

    def test_seed_retry_interrupt_resume_and_completed_noop(self):
        fake = fake_client()
        calls = Counter()
        def worker(command, env, console):
            seed = int(command[command.index("--seed") + 1])
            calls[seed] += 1
            path = run_directory(command[command.index("--output-dir") + 1], "qwen", "Qwen/Qwen3-4B", seed)
            console.parent.mkdir(parents=True, exist_ok=True)
            console.write_text(f"original attempt {calls[seed]} seed {seed}\n")
            if seed == 1103 and calls[seed] == 1:
                write_json(path / "manifest.json", {"status": "failed", "error_type": "TransientInferenceError"})
                return 1
            if seed == 2207 and calls[seed] == 1:
                write_json(path / "manifest.json", {"status": "running"})
                raise KeyboardInterrupt()
            successful_seed(path, seed, env)
            return 0
        from utils.data_utils import load_pair_selection_price_history
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                patch("experiments.run_main.build_llm", return_value=fake), \
                patch("experiments.run_main.run_seed_process", side_effect=worker), \
                patch("experiments.run_main.time.sleep") as sleep, \
                patch("utils.data_utils.load_pair_selection_price_history", wraps=load_pair_selection_price_history) as loader, \
                patch("yfinance.download", side_effect=AssertionError("No market-data network")), \
                patch("experiments.summarize_results.summarize_batch", side_effect=fake_summary), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            batch = Path(tmp) / "batch"
            with self.assertRaises(KeyboardInterrupt):
                main(self.args(batch))
            sleep.assert_called_once_with(120)
            snapshot = (batch / "inputs/pool.csv").read_bytes()
            first = run_directory(batch / "runs", "qwen", "Qwen/Qwen3-4B", 1103)
            preserved = (first / "manifest.json").read_bytes()
            self.assertEqual(main(self.args(batch)), 0)
            self.assertEqual(calls, {1103: 2, 2207: 2, 3301: 1})
            self.assertEqual((first / "manifest.json").read_bytes(), preserved)
            self.assertEqual((batch / "inputs/pool.csv").read_bytes(), snapshot)
            loader.assert_called_once()
            self.assertEqual(len(list((batch / "attempts").rglob("console.log"))), 2)
            self.assertTrue(summary_valid(batch))
            before = calls.copy()
            checks = fake.validate_server.call_count
            main(self.args(batch))
            self.assertEqual(calls, before)
            self.assertEqual(fake.validate_server.call_count, checks)
            self.assertEqual(len(json.loads((batch / "batch.json").read_text())["runs"]), 3)
            (batch / "inputs/pool.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                main(self.args(batch))
            self.assertEqual(calls, before)

    def test_check_only_is_one_bounded_probe_without_workers(self):
        fake = fake_client()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
                patch("experiments.run_main.build_llm", return_value=fake), \
                patch("experiments.run_main.run_seed_process") as worker, \
                patch("yfinance.download", side_effect=AssertionError("No download")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            batch = Path(tmp) / "check"
            main(self.args(batch, "--check-only"))
            worker.assert_not_called()
            fake.validate_server.assert_called_once()
            self.assertEqual(fake.max_output_tokens, 1024)
            self.assertEqual(fake.request_attempts, 1)
            self.assertIn("CHECK_OK models=1 train_steps=27 test_steps=75", output.getvalue())
            self.assertEqual(json.loads((batch / "batch.json").read_text())["status"], "checked")
            self.assertFalse((batch / "runs").exists())

    def test_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp, batch_lock(Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, "another process"), batch_lock(Path(tmp)):
                self.fail("Second writer acquired the lock")


if __name__ == "__main__":
    unittest.main()
