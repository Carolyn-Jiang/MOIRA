"""Batch contracts, ranking, failure handling and publishing boundaries (no real inference)."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from experiments.main_artifacts import write_json
from experiments.run_main import main, parse_args, DEFAULT_SEEDS
from experiments.summarize_results import select_representatives, summarize_batch
from llm.backbone import infer_family, build_llm


class TestBatchCLI(unittest.TestCase):
    def test_model_name_only_and_multiple_models(self):
        one = parse_args(["llama3.1:8b"])
        self.assertEqual(one.families, ["llama"])
        self.assertEqual(one.seeds, list(DEFAULT_SEEDS))
        for tokens in [["llama3.1:8b", "Qwen/Qwen2.5-7B-Instruct"],
                       ["--models", "llama3.1:8b,Qwen/Qwen2.5-7B-Instruct"]]:
            self.assertEqual(parse_args(tokens).families, ["llama", "qwen"])

    def test_reject_bad_or_duplicate_inputs(self):
        for tokens in [[], ["x", "x"],
                       ["qwen", "--seeds", "1", "1", "2", "3", "4"], ["claude-test"]]:
            with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(tokens)

    def test_open_source_default_never_uses_proprietary_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "do-not-use", "OPENAI_BASE_URL": "https://invalid.example"}, clear=True):
            llm = build_llm("llama", "llama3.1:8b")
            self.assertEqual(llm.base_url, "http://localhost:11434/v1")
            self.assertEqual(llm.api_key, "local-no-key")
            self.assertEqual(infer_family("gpt-oss:20b"), "open_source")

    def test_dry_run_has_no_files_or_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.run_main.build_llm") as build:
            out = Path(tmp) / "batch"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["qwen2.5:7b", "--dry-run", "--output-dir", str(out)]), 0)
            self.assertFalse(out.exists())
            build.assert_not_called()

    def test_two_models_schedule_ten_processes_and_preserve_failures(self):
        from experiments.main_artifacts import run_directory
        codes = iter([0, 1] + [0]*8)
        def fake_worker(command, env, console):
            code = next(codes)
            if code == 0:
                run_directory(command[command.index("--output-dir")+1],
                              command[command.index("--provider")+1], command[command.index("--model")+1],
                              int(command[command.index("--seed")+1])).mkdir(parents=True)
            return code
        fake = Mock()
        fake._get_client.return_value.models.list.return_value.data = [
            SimpleNamespace(id="llama-test"), SimpleNamespace(id="qwen-test")]
        fake.inference_config.return_value = {"send_seed": True}
        fake.validate_server.return_value = {"status": "passed"}
        pool = pd.DataFrame({"A": [1., 2.]}, index=pd.date_range("2025-01-01", periods=2))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("experiments.run_main.build_llm", return_value=fake), \
                patch("utils.data_utils.load_pair_selection_price_history", return_value=(pool, "synthetic")), \
                patch("experiments.run_main.run_seed_process", side_effect=fake_worker) as runner, \
                patch("experiments.run_main.completed_seed", side_effect=lambda path, *a: path.exists()), \
                contextlib.redirect_stderr(io.StringIO()):
            out = Path(tmp) / "batch"
            with self.assertRaises(RuntimeError):
                main(["llama-test", "qwen-test", "--output-dir", str(out)])
            self.assertEqual(runner.call_count, 10)
            manifest = json.loads((out / "batch.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["pair_history_years"], 10)
            self.assertEqual(len(manifest["runs"]), 10)
            self.assertFalse((out / "summary").exists())
            self.assertTrue((out / "inputs/pool.csv").is_file())
            for call in runner.call_args_list:
                self.assertIn("MAIN_POOL_SNAPSHOT", call.args[1])

    def test_unavailable_model_fails_before_any_seed(self):
        fake = Mock()
        fake._get_client.return_value.models.list.return_value.data = []
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.run_main.build_llm", return_value=fake), \
                patch("experiments.run_main.run_seed_process") as runner, contextlib.redirect_stderr(io.StringIO()):
            out = Path(tmp) / "batch"
            with self.assertRaises(ValueError):
                main(["qwen-test", "--output-dir", str(out)])
            runner.assert_not_called()
            self.assertEqual(json.loads((out / "batch.json").read_text())["status"], "failed")

    def test_bad_reasoning_preflight_fails_before_any_seed(self):
        from llm.errors import InferenceContractError
        fake = Mock()
        fake._get_client.return_value.models.list.return_value.data = [SimpleNamespace(id="Qwen/Qwen3-4B")]
        fake.inference_config.return_value = {"thinking_mode": "on"}
        fake.validate_server.side_effect = InferenceContractError("missing server reasoning")
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.run_main.build_llm", return_value=fake), \
                patch("experiments.run_main.run_seed_process") as runner, contextlib.redirect_stderr(io.StringIO()):
            out = Path(tmp) / "batch"
            with self.assertRaises(InferenceContractError):
                main(["Qwen/Qwen3-4B", "--thinking", "on", "--output-dir", str(out)])
            runner.assert_not_called()
            self.assertEqual(json.loads((out / "batch.json").read_text())["status"], "failed")

    def test_cli_thinking_mode_reaches_all_workers(self):
        from experiments.main_artifacts import run_directory
        def fake_worker(command, env, console):
            run_directory(command[command.index("--output-dir")+1], "qwen", "Qwen/Qwen3-32B",
                          int(command[command.index("--seed")+1])).mkdir(parents=True)
            return 0
        fake = Mock()
        fake._get_client.return_value.models.list.return_value.data = [SimpleNamespace(id="Qwen/Qwen3-32B")]
        fake.inference_config.return_value = {"thinking_mode": "off"}
        fake.validate_server.return_value = {"status": "passed"}
        pool = pd.DataFrame({"A": [1., 2.]}, index=pd.date_range("2025-01-01", periods=2))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_SOURCE_THINKING": "on"}), \
                patch("experiments.run_main.build_llm", return_value=fake) as build, \
                patch("utils.data_utils.load_pair_selection_price_history", return_value=(pool, "synthetic")), \
                patch("experiments.run_main.run_seed_process", side_effect=fake_worker) as runner, \
                patch("experiments.run_main.completed_seed", side_effect=lambda path, *a: path.exists()), \
                patch("experiments.summarize_results.summarize_batch"), contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "batch"
            main(["Qwen/Qwen3-32B", "--thinking", "off", "--output-dir", str(out)])
            self.assertEqual(build.call_args.kwargs["thinking_mode"], "off")
            self.assertEqual(runner.call_count, 5)
            self.assertTrue(all(c.args[1]["OPEN_SOURCE_THINKING"] == "off" for c in runner.call_args_list))
            self.assertEqual(json.loads((out / "batch.json").read_text())["thinking_mode_requested"], "off")


class TestRanking(unittest.TestCase):
    def runs(self, values):
        return [{"manifest": {"seed": s}, "metrics": {"nobeta": {"cumulative_return": v}}}
                for s, v in zip(range(1, 6), values)]

    def test_selects_whole_seed_and_numeric_max_not_absolute_value(self):
        runs = self.runs([-.4, -.1, -.3, -.2, -.5])
        selected = select_representatives(runs, "nobeta", range(1, 6))
        self.assertIs(selected["median"], runs[2])
        self.assertIs(selected["max"], runs[1])

    def test_ties_are_reproducible(self):
        selected = select_representatives(list(reversed(self.runs([0]*5))), "nobeta", range(1, 6))
        self.assertEqual(selected["median"]["manifest"]["seed"], 3)
        self.assertEqual(selected["max"]["manifest"]["seed"], 1)

    def test_rejects_incomplete_and_nonfinite(self):
        for runs in [self.runs([1, 2, 3, 4]), self.runs([1, 2, None, 4, 5]), self.runs([1, 2, float("nan"), 4, 5])]:
            with self.assertRaises(ValueError):
                select_representatives(runs, "nobeta", range(1, 6))

    def test_summary_rejects_different_data_before_writing(self):
        runs = self.runs(range(5))
        entries = []
        for i, run in enumerate(runs):
            run["manifest"].update(provider="qwen", model="qwen-test", pool_hash="changed" if i == 0 else "same")
            entries.append({"provider": "qwen", "model": "qwen-test", "seed": i+1,
                            "path": f"runs/seed_{i+1}", "status": "completed", "exit_code": 0})
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.summarize_results.load_run", side_effect=runs):
            write_json(Path(tmp) / "batch.json", {"models": [{"provider": "qwen", "model": "qwen-test"}],
                       "seeds": list(range(1, 6)), "sizing": ["nobeta"], "runs": entries})
            with self.assertRaisesRegex(ValueError, "pool_hash"):
                summarize_batch(tmp)
            self.assertFalse((Path(tmp) / "summary").exists())

    def test_summary_rejects_mixed_thinking_modes_before_writing(self):
        runs = self.runs(range(5))
        entries = []
        for i, run in enumerate(runs):
            run["manifest"].update(provider="qwen", model="Qwen/Qwen3-4B",
                                   inference={"thinking_mode": "off" if i == 0 else "on"})
            entries.append({"provider": "qwen", "model": "Qwen/Qwen3-4B", "seed": i+1,
                            "path": f"runs/seed_{i+1}", "status": "completed", "exit_code": 0})
        with tempfile.TemporaryDirectory() as tmp, patch("experiments.summarize_results.load_run", side_effect=runs):
            write_json(Path(tmp) / "batch.json", {"models": [{"provider": "qwen", "model": "Qwen/Qwen3-4B"}],
                       "seeds": list(range(1, 6)), "sizing": ["nobeta"], "runs": entries})
            with self.assertRaisesRegex(ValueError, "thinking modes"):
                summarize_batch(tmp)
            self.assertFalse((Path(tmp) / "summary").exists())

    def test_integration_tables_and_both_figure_sets(self):
        import experiments.run_redesign as rr
        from test_redesign import Test35DiscoveryTraining, _causal_pool
        symbols, data, _, provider = Test35DiscoveryTraining()._setup()
        pool = _causal_pool(n_years=2)
        keep_artifacts = os.getenv("TEST_KEEP_ARTIFACTS")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                "LLM_BACKEND": "offline", "LLM_MAX_CYCLES": "1", "LLM_LOOKBACK": "5"}, clear=True), \
                patch.object(rr, "SYMBOLS", symbols), patch.object(rr, "TRAIN_SESSIONS", 12), \
                patch.object(rr, "TEST_SESSIONS", 18), \
                patch.object(rr, "load_offline_dataset", return_value=data), \
                patch.object(rr, "load_pair_selection_price_history", return_value=(pool, "synthetic-test")), \
                patch.object(rr, "make_discovery_provider", return_value=provider), \
                patch("experiments.baselines.select_coint_pair", return_value=("A", "B")), \
                contextlib.redirect_stdout(io.StringIO()):
            batch = Path(tmp)
            runs = []
            for seed in DEFAULT_SEEDS:
                path = rr.run_redesign_experiment(seed=seed, output_root=batch / "runs", sizing="both")
                runs.append({"provider": "offline", "model": "offline-deterministic", "seed": seed,
                             "path": str(path.relative_to(batch)), "status": "completed", "exit_code": 0})
            write_json(batch / "batch.json", {"models": [{"provider": "offline", "model": "offline-deterministic"}],
                "seeds": list(DEFAULT_SEEDS), "sizing": ["nobeta", "beta"], "runs": runs})
            result = summarize_batch(batch)
            self.assertEqual(len(pd.read_csv(result / "all_seeds.csv")), 10)
            selected = pd.read_csv(result / "selected_seeds.csv")
            self.assertEqual(set(selected.selection), {"median", "max"})
            self.assertEqual(set(selected[selected.selection == "median"].seed), {3301})
            self.assertEqual(set(selected[selected.selection == "max"].seed), {1103})
            summary = pd.read_csv(result / "comparison.csv")
            self.assertTrue((summary.n_seeds == 5).all())
            for kind in ("median", "max"):
                for mode in ("nobeta", "beta"):
                    folder = result / kind / mode
                    self.assertEqual(len(list(folder.glob("*.png"))), 15)
                    self.assertEqual(len(list(folder.glob("*.pdf"))), 15)
                    self.assertTrue((folder / "selection.json").is_file())
            self.assertFalse(list(batch.rglob("*.md")))
            # Keep QA data only when explicitly requested by the development harness.
            if keep_artifacts:
                import shutil
                shutil.copytree(batch, keep_artifacts)


if __name__ == "__main__":
    unittest.main()
