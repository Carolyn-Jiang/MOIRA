# FinMem fixed-pair baseline

This adapter retains FinMem's layered memory, retrieval and reflection. It presents one fixed equity pair as a synthetic asset: `LONG_SHORT` longs the first leg and shorts the second, `SHORT_LONG` reverses the exposure, and `FLAT` closes both legs. Evaluation replays the same decisions under No-Beta and Beta sizing using the original leg prices.

Run all commands from the repository root in the environment described in [RUNNING.md](../../docs/RUNNING.md):

```bash
python -m pip install -r baselines/finmem_fixed_pair/requirements.txt
python -m experiments.prepare_finmem --study main --seed 1103 --output-dir runs/finmem-main
python baselines/finmem_fixed_pair/run.py warmup --config-path runs/finmem-main/config.json
python baselines/finmem_fixed_pair/run.py test --config-path runs/finmem-main/config.json
python baselines/finmem_fixed_pair/run.py eval --config-path runs/finmem-main/config.json
```

Set `OPENAI_API_KEY` in the root `.env` for GPT-4.1-mini and `text-embedding-3-large`. `FINMEM_ENV_FILE` can point to another local credential file. Generated configuration uses an embedded local Qdrant store; no database server is required. Preparation makes no model calls and prints `FINMEM_INPUTS_OK ... test_steps=75` for the main window.

Main uses the paper adapter's fixed BMRN/META pair. Inputs share MOIRA's 27-session training window, 75 test decisions, prior-session news and the June 2 price-only terminal observation. FinMem warmup learns from adjacent observations within the training window; test prices are not used for warmup. Test decisions run through May 30. The following price settles the final decision without another model call. Beta uses a 20-session rolling log-price regression, the shared hedge-ratio guard and gross-exposure sizing.

For independent A1/A2 runs, prepare separate directories, then use the same warmup/test/eval commands with the corresponding configuration:

```bash
python -m experiments.prepare_finmem --study a1 --seed 3301 --output-dir runs/finmem-a1
python -m experiments.prepare_finmem --study a2 --seed 3301 --output-dir runs/finmem-a2
```

A1 uses AMZN/MSFT with 28 training observations and 304 test decisions. A2 uses AAPL/MSFT with 28 and 75. Input download and freezing follow the shared ablation loader. `scripts/evaluate_a1_slices.py` is the original slice-analysis utility for an existing A1 trajectory; it does not create independent main/A2 experiments.

Resume an interrupted phase from its most recent daily checkpoint:

```bash
python baselines/finmem_fixed_pair/run.py warmup-checkpoint --config-path runs/finmem-main/config.json
python baselines/finmem_fixed_pair/run.py test-checkpoint --config-path runs/finmem-main/config.json
python baselines/finmem_fixed_pair/run.py eval --config-path runs/finmem-main/config.json
```

Use the checkpoint command only for the interrupted phase. Run directories contain local checkpoints, logs and `metrics/fixed_pair_metrics.csv`, `metrics/fixed_pair_daily_results.csv`, and `metrics/fixed_pair_equity.png`. Missing test decisions stop evaluation. Generated outputs are excluded from Git and do not replace the Overleaf results.

Offline checks (no API calls):

```bash
python -m pytest -q baselines/finmem_fixed_pair/tests
```

The adapted FinMem implementation retains its [MIT license](LICENSE).
