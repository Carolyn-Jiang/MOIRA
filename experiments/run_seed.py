"""Internal isolated worker for one model/seed; use run_main for batches."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["llama", "qwen", "open_source", "openai", "deepseek", "gemini"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sizing", choices=["both", "beta", "nobeta"], default="both")
    parser.add_argument("--thinking", choices=["on", "off"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.update(LLM_BACKEND=args.provider, LLM_MODEL=args.model)
    if args.thinking:
        os.environ["OPEN_SOURCE_THINKING"] = args.thinking
    from experiments.run_redesign import run_redesign_experiment
    run_redesign_experiment(seed=args.seed, output_root=args.output_dir, sizing=args.sizing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
