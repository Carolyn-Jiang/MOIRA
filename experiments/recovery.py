"""Conservative seed-level recovery; completed seeds and failed attempts are immutable."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from experiments.main_artifacts import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_signature(root, env):
    from experiments.run_redesign import _code_hash
    return {"code_hash": _code_hash(),
            "news": {p.name: digest(p) for p in sorted((root / "data/news").glob("*.json"))},
            "requirements": digest(root / "requirements.txt"),
            "protocol": {k: int(env.get(k, default)) for k, default in (
                ("TRAIN_SESSIONS", 27), ("LLM_LOOKBACK", 20),
                ("LLM_SELECTION_FREQUENCY", 23), ("LLM_BREAK_CHECK_FREQUENCY", 23))}}


@contextmanager
def batch_lock(batch):
    """Pass the open fd to workers so a killed parent cannot leave an unowned writer."""
    with (batch / ".batch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This batch is still running in another process") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield handle.fileno()
        finally:
            # Close only. An inherited worker fd retains the lock if necessary.
            pass


def completed_seed(path, provider, model, seed, modes, code_hash):
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        return False
    if any(manifest.get(k) != v for k, v in dict(provider=provider, model=model,
                                                seed=seed, code_hash=code_hash).items()):
        raise ValueError("Completed seed identity/code differs from its batch")
    from experiments.plot_main_results import load_run
    run = load_run(path, modes)
    for mode in modes:
        if (len(run[mode]["decisions"]) != len(manifest["test_dates"])
                or run["metrics"][mode].get("finalized_transitions") != len(manifest["test_dates"])):
            raise ValueError("Completed seed has incomplete settlement records")
    return True


def archive_attempt(batch, relative, console):
    """Move, never delete, a partial/failed seed before restarting it from step zero."""
    path = batch / relative
    if not path.exists() and not console.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = batch / "attempts" / Path(relative).relative_to("runs") / stamp
    destination.mkdir(parents=True, exist_ok=False)
    if path.exists():
        shutil.move(str(path), str(destination / "run"))
    if console.exists():
        shutil.move(str(console), str(destination / "console.log"))
    return str(destination.relative_to(batch))


def summary_valid(batch):
    marker = batch / "summary/complete.json"
    if not marker.is_file():
        return False
    files = json.loads(marker.read_text()).get("files", {})
    return bool(files) and all((batch / "summary" / name).is_file() and
                              digest(batch / "summary" / name) == checksum
                              for name, checksum in files.items())


def finish_summary(batch):
    from experiments.summarize_results import summarize_batch
    if summary_valid(batch):
        return
    output = batch / "summary"
    if output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = batch / "attempts" / f"summary_{stamp}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(archive))
    summarize_batch(batch)
    files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "complete.json", {"files": files})
