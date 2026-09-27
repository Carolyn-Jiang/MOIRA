# Running the experiments

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

For a local vLLM server on port 8000 without authentication, `.env.example` already has the required endpoint; no key is needed. For hosted APIs, fill only `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` or `GEMINI_API_KEY`. Provider-specific `*_BASE_URL` variables support compatible endpoints. Keys are isolated between providers. `--base-url` overrides the endpoint for that invocation.

## Open-weight models

In a separate GPU environment, install vLLM and start one model server at a time. The launcher resolves and saves an exact Hugging Face revision and reuses it on restart; model access must already be granted for gated weights.

```bash
python -m pip install 'vllm==0.11.2'
bash scripts/serve_model.sh meta-llama/Llama-3.1-8B-Instruct runs/server-llama8b
# Alternatively:
# bash scripts/serve_model.sh meta-llama/Llama-3.1-70B-Instruct runs/server-llama70b
# bash scripts/serve_model.sh Qwen/Qwen3-4B runs/server-qwen4b
# bash scripts/serve_model.sh Qwen/Qwen3-32B runs/server-qwen32b
```

In the client terminal, check readiness and make one bounded inference probe before the experiment:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
python -m experiments.run_main meta-llama/Llama-3.1-8B-Instruct --check-only --output-dir runs/check-llama8b
python -m experiments.run_main meta-llama/Llama-3.1-8B-Instruct --output-dir runs/llama8b
```

Use the same commands with `meta-llama/Llama-3.1-70B-Instruct` and separate output directories for 70B. Qwen thinking modes use separate batches against the same model revision:

```bash
python -m experiments.run_main Qwen/Qwen3-4B --thinking on --check-only --output-dir runs/check-qwen4b-on
python -m experiments.run_main Qwen/Qwen3-4B --thinking on --output-dir runs/qwen4b-on
python -m experiments.run_main Qwen/Qwen3-4B --thinking off --check-only --output-dir runs/check-qwen4b-off
python -m experiments.run_main Qwen/Qwen3-4B --thinking off --output-dir runs/qwen4b-off
# After starting the Qwen3-32B server:
python -m experiments.run_main Qwen/Qwen3-32B --thinking on --check-only --output-dir runs/check-qwen32b-on
python -m experiments.run_main Qwen/Qwen3-32B --thinking on --output-dir runs/qwen32b-on
python -m experiments.run_main Qwen/Qwen3-32B --thinking off --check-only --output-dir runs/check-qwen32b-off
python -m experiments.run_main Qwen/Qwen3-32B --thinking off --output-dir runs/qwen32b-off
```

The server must use `--reasoning-parser qwen3` in both modes (the launcher sets it). Only the final answer enters action parsing and memory. Thinking-on requires a separate nonempty reasoning field; thinking-off requires it to be empty. Malformed or truncated responses stop the batch.

## Hosted models

Pass the exact served identifier; an unavailable historical alias is not silently substituted. The paper identifiers are `gpt-4.1-mini`, `gpt-5.4-mini`, `gpt-5.4-nano`, `o4-mini`, `deepseek-v4.1-flash`, `gemini-3.1-flash-lite`, and `gemini-3.8-flash`.

```bash
python -m experiments.run_main gpt-4.1-mini --check-only --output-dir runs/check-gpt41
python -m experiments.run_main gpt-4.1-mini gpt-5.4-mini gpt-5.4-nano o4-mini --seeds 1103 --output-dir runs/openai
python -m experiments.run_main deepseek-v4.1-flash --seeds 1103 --output-dir runs/deepseek
python -m experiments.run_main gemini-3.1-flash-lite gemini-3.8-flash --seeds 1103 --output-dir runs/gemini
```

Gemini defaults to low reasoning effort and a 4,096-token output cap; DeepSeek defaults to non-thinking. OpenAI uses the provider's reasoning default unless `OPENAI_REASONING_EFFORT` is set. Hosted APIs may keep reasoning private; their final answer alone is parsed. Provider-specific token limits, timeouts and reasoning controls are in `.env.example`.

## Ablations

```bash
python -m experiments.run_ablations --dry-run
python -m experiments.run_ablations --check-only
python -m experiments.run_ablations --ablations a1 a2 a3 a4 --seeds 1103 --output-dir runs/ablations
```

The two settings are GPT-4.1-mini/Beta and GPT-5.4-mini/No-Beta. A1 uses 28 training and 304 test decisions; A2 uses 28 and 75. A3/A4 use the primary 27/75 windows and save their own matched full controls. A3 shares one warmup and initial reflection prompt, and uses identical pair-quality memory at test entry. A4 shares the trained prompt, pair-quality memory and initial pair, retaining monitoring but disabling replacement. All new test runs settle the last decision using the following session's prices.

## Inputs and local output

The bundled input panel contains only the `prices` and `news` fields consumed by the loader, keyed by date and ticker. It preserves the source experiment's summarized news; the BMRN/CRM supplements were collected through web search. Prices in this panel define the main dates; valuation uses the adjusted-price CSV cache. The cache covers the main run and its final settlement. Prior-session news is aligned before each decision; terminal settlement has prices only.

A1/A2 download the public [TheFinAI/Herculean](https://huggingface.co/datasets/TheFinAI/Herculean) price/news parquet inputs (CC BY 4.0). They may also need yfinance to extend the price cache to the required ten-year start. Each batch then freezes and hashes its inputs; a resumed batch reuses that snapshot and rejects altered inputs. Inference requires access to the configured local or hosted server.

The main runner defaults to three seeds (`1103 2207 3301`); `--seeds 1103` runs only one. `--sizing beta` or `--sizing nobeta` runs one exposure. `--dry-run` prints a plan without inference. `--check-only` sends one short probe per model, without training. Successful runs print `CHECK_OK`, `SEED_RESULT`, `BATCH_OK` or `CELL_OK` and write local metrics and plots under `runs/`.

Rerun the same command to resume. Completed seeds/cells are skipped. Transient connection, rate-limit and server failures retry up to three times with 120-second waits; failed attempts are preserved locally. Authentication and response-contract failures stop immediately. Main recovery is at seed boundaries; ablations also reuse completed training and full-control checkpoints. Partly completed trajectories restart. New local main results include comparison tables and median/max representative trajectories. They do not replace `paper/` assets.

```bash
python -m unittest discover -s tests
```

## Passive controls

```bash
python -m experiments.run_baselines --dataset main --output-dir runs/passive-main
python -m experiments.run_baselines --dataset herculean --output-dir runs/passive-a1
python -m experiments.run_baselines --dataset herculean_regime --output-dir runs/passive-a2
```

These run LS, SL and EW on the causally selected fixed pair under both exposures. The main model runner also evaluates matched-timeline controls automatically. To evaluate a separate local decision CSV, add `--timeline path/to/decision_log.csv`; its dates must match every test decision.


## EG+Z and EG-DDQN

Run from the repository root; neither baseline makes model calls:

```bash
python -m experiments.run_engle_granger_baseline --studies main a1 a2 --output-dir runs/eg-z
python -m experiments.run_eg_ddqn_baseline --studies main a1 a2 --seeds 1103 2207 3301 --output-dir runs/eg-ddqn
```

EG+Z enters at residual Z = ±1 and exits within 0.25. EG-DDQN uses a NumPy Double DQN (32 hidden units, 600 training episodes) and freezes the policy during testing. Both use the shared price/news windows, exposure rules and final price-only settlement. The main pair is selected from the ten years strictly before testing; A1/A2 use the matched fixed pairs AMZN/MSFT and AAPL/MSFT. These baselines retain a fixed pair throughout the test window.

Inputs are frozen and hashed per study; rerunning the same command skips completed cells. Results remain under `runs/`: `comparison.csv` contains CR, Sharpe and MDD (CR/MDD as fractions), and each study has `equity.csv` and local decision output. DDQN also saves per-metric medians. These outputs never replace the manuscript tables or figures.

For a cheap CPU execution check, use a separate directory:

```bash
python -m experiments.run_eg_ddqn_baseline --studies main --seeds 1103 --episodes 4 --output-dir runs/check-eg-ddqn
```

A completed main cell prints `BASELINE_OK ... test_steps=75`; its trajectory contains 76 equity observations. The four-episode check is not a reported experiment.

## FinMem

See the [main README](../README.md#finmem) for the fixed-pair runner commands. It uses GPT-4.1-mini and an embedding endpoint; unlike the two numeric baselines, a full run makes paid API calls. All checkpoints, generated data and logs stay local.
