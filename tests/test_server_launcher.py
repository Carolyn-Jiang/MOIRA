"""Exercise the real Bash launcher with fake Hub/vLLM executables; no downloads/GPU."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class TestServerLauncher(unittest.TestCase):
    def test_sha_is_reused_and_all_server_arguments_are_logged(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts/serve_model.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binaries = root / "bin"
            binaries.mkdir()
            (root / "huggingface_hub.py").write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
class HfApi:
    def model_info(self, model):
        target=Path(os.environ['HUB_CALLS'])
        target.write_text(target.read_text()+'call\\n' if target.exists() else 'call\\n')
        return SimpleNamespace(sha='a'*40)
''')
            (binaries / "python").write_text(
                '#!/bin/bash\nif [[ "$1" == -m ]]; then echo vllm==0.11.2; else exec ' +
                shlex.quote(sys.executable) + ' "$@"; fi\n')
            (binaries / "vllm").write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$VLLM_ARGS"\necho "fake server stopped"\n')
            for file in binaries.iterdir():
                file.chmod(0o755)
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "PYTHONPATH": str(root), "HUB_CALLS": str(root / "hub_calls"),
                   "VLLM_ARGS": str(root / "args")}
            output = root / "logs"
            command = ["bash", str(launcher), "Qwen/Qwen3-32B", str(output)]
            for _ in range(2):
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            args = (root / "args").read_text().splitlines()
            for flag, value in (("--revision", "a"*40), ("--tokenizer-revision", "a"*40),
                                ("--reasoning-parser", "qwen3"), ("--tensor-parallel-size", "2")):
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertEqual((root / "hub_calls").read_text(), "call\n")
            self.assertEqual(json.loads((output / "revision.json").read_text())["revision"], "a"*40)
            self.assertEqual(len(list(output.glob("*/server.log"))), 2)
            self.assertEqual(len(list(output.glob("*/command.txt"))), 2)
            bad = subprocess.run(["bash", str(launcher), "Qwen/Qwen3-4B", str(output)],
                                 env=env, text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("another model", bad.stderr)


if __name__ == "__main__":
    unittest.main()
