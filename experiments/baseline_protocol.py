"""Shared frozen inputs and terminal settlement for the paper's active baselines."""
from pathlib import Path
from experiments.baselines import select_coint_pair
from experiments.run_ablations import PROJECT_ROOT, _snapshot

DATASETS = {"main": "main", "a1": "herculean", "a2": "herculean_regime"}
MATCHED_PAIRS = {"a1": ("AMZN", "MSFT"), "a2": ("AAPL", "MSFT")}


def prepare_study(study, directory, cache_dir=None):
    spec = _snapshot(DATASETS[study], Path(directory),
                     cache_dir or PROJECT_ROOT / "data/herculean")
    spec["pair"] = (select_coint_pair(spec["pool_df"], spec["symbols"], spec["test_dates"][0])
                    if study == "main" else MATCHED_PAIRS[study])
    return spec


def study_metadata(spec):
    train, test = spec["train_dates"], spec["test_dates"]
    return {"pair": list(spec["pair"]), "train_start": train[0], "train_end": train[-1],
            "train_sessions": len(train), "test_start": test[0], "test_end": test[-1],
            "test_steps": len(test), "settlement_date": spec["settlement_date"],
            "dataset": spec["dataset_meta"]}
