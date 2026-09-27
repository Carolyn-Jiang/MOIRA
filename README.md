# MOIRA

MOIRA combines statistical pair screening, language-model decisions, compact policy memory, and lifecycle monitoring in a sequential pair-trading environment. Open-weight and hosted models use the same simulation engine.

## Framework

1. **Pair selection:** screen prices from the ten calendar years strictly before each selection date, then select a candidate using statistics, news and company context.
2. **Policy learning:** propose transferable guidance from training reflection; retain an edit only when replay improves reward. Freeze the learned global prompt during testing.
3. **Daily decisions:** select Long–Short, Short–Long or Flat, with feedback from settled counterfactual outcomes.
4. **Pair maintenance:** evaluate accumulated break evidence at 23-session validation boundaries. A daily break alone does not replace a pair.

The main experiment has 27 training sessions and 75 test decisions. The following price-only session settles the final decision without an additional news observation or model call. No-Beta and Beta change exposure; policy training uses No-Beta. A1/A2 retain their pre-specified 28-session training windows. A3 disables post-warmup policy consolidation; A4 disables pair replacement while retaining daily decisions and monitoring.

**Figure 2. Overall workflow**

![Overall workflow](paper/figures/figure_02_overall_workflow_readable.png)

<details>
<summary>Task and module diagrams (Figures 1, 8–10)</summary>

**Figure 1. Conceptual task**

![Conceptual task](paper/figures/figure_01_concept_task.png)

**Figure 8. Policy-memory learning and validation**

![Policy-memory learning and validation](paper/figures/figure_08_belief_update_readable.png)

**Figure 9. Pair selection**

![Pair selection](paper/figures/figure_09_pair_select_readable.png)

**Figure 10. Lifecycle monitoring**

![Lifecycle monitoring](paper/figures/figure_10_lifecycle_readable.png)

</details>

## Results reported in the paper

CR and MDD are percentages; Sharpe is unitless. These are the paper’s selected results, not averages over seeds. All reported results and figures follow the Overleaf manuscript.

| Model / baseline | No-Beta CR | Sharpe | MDD | Beta CR | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4.1-mini | 25.2 | 2.66 | -14.0 | 30.6 | 3.62 | -4.8 |
| GPT-5.4-mini | 24.3 | 2.79 | -8.4 | 23.6 | 2.58 | -13.0 |
| GPT-5.4-nano | 21.2 | 2.16 | -8.1 | 20.2 | 1.62 | -10.9 |
| DeepSeek-V4.1-Flash | 12.9 | 2.62 | -4.1 | 12.4 | 1.57 | -9.2 |
| o4-mini | -2.6 | -0.28 | -12.3 | 10.7 | 1.04 | -22.0 |
| Gemini 3.1 Flash-Lite | 12.3 | 3.25 | -3.4 | 22.6 | 4.57 | -3.7 |
| Gemini 3.8 Flash | 6.3 | 1.94 | -3.5 | 4.1 | 1.17 | -5.0 |
| Llama-3.1-8B-Instruct | 4.2 | 0.54 | -18.2 | 12.4 | 1.53 | -19.0 |
| Llama-3.1-70B-Instruct | 2.0 | 0.29 | -16.6 | 20.6 | 2.20 | -17.7 |
| Qwen3-4B (thinking off) | 4.8 | 0.58 | -16.0 | -6.3 | -0.60 | -19.0 |
| Qwen3-4B (thinking on) | -4.1 | -0.17 | -20.1 | 2.1 | 0.33 | -19.8 |
| Qwen3-32B (thinking off) | 15.8 | 1.56 | -9.8 | 15.2 | 1.31 | -19.6 |
| Qwen3-32B (thinking on) | 15.4 | 2.07 | -7.1 | 31.2 | 2.36 | -14.8 |
| Passive LS | 2.6 | 0.37 | -25.1 | 2.4 | 0.36 | -24.9 |
| Passive SL | -8.1 | -0.47 | -30.3 | -7.9 | -0.46 | -30.0 |
| Passive EW | -10.4 | -0.86 | -22.3 | -10.4 | -0.86 | -22.3 |
| EG+Z | -9.92 | -0.84 | -15.17 | -10.17 | -0.87 | -15.26 |
| EG-DDQN | 11.25 | 0.98 | -22.42 | 15.10 | 1.27 | -17.21 |
| FinMem (GPT-4.1-mini) | 13.53 | 3.07 | -2.54 | 15.64 | 2.91 | -3.71 |

**Figure 26. All 13 model configurations under both exposure rules**

![All 13 model configurations under both exposure rules](paper/figures/figure_26_metrics_all.png)

[PDF](paper/figures/figure_26_metrics_all.pdf)

| Study | Model / exposure | Condition | CR | Sharpe | MDD |
|---|---|---|---:|---:|---:|
| A1 | GPT-4.1-mini / Beta | Full | 16.19 | 0.61 | -11.1 |
| A1 | GPT-5.4-mini / No-Beta | Full | 25.41 | 0.83 | -17.4 |
| A2 | GPT-4.1-mini / Beta | Full | 8.39 | 1.93 | -3.9 |
| A2 | GPT-5.4-mini / No-Beta | Full | 6.18 | 1.27 | -4.4 |
| A3 | GPT-4.1-mini / Beta | With update | 30.6 | 3.62 | -4.8 |
| A3 | GPT-4.1-mini / Beta | Without update | 23.4 | 2.51 | -10.4 |
| A3 | GPT-5.4-mini / No-Beta | With update | 24.2 | 2.79 | -8.4 |
| A3 | GPT-5.4-mini / No-Beta | Without update | 2.7 | 0.41 | -19.2 |
| A4 | GPT-4.1-mini / Beta | With reselection | 30.6 | 3.62 | -4.8 |
| A4 | GPT-4.1-mini / Beta | Without reselection | 2.8 | 0.38 | -21.8 |
| A4 | GPT-5.4-mini / No-Beta | With reselection | 24.2 | 2.79 | -8.4 |
| A4 | GPT-5.4-mini / No-Beta | Without reselection | 15.4 | 2.02 | -7.3 |

## Model calls and experiment commands

Run the commands below in **Bash from the repository root**. Python 3.11 is required for these environment examples. A model name selects its provider automatically; one invocation can accept one or several model names.

MOIRA prompts use `step 001`, `step 002`, etc. instead of calendar dates, restarting at each training replay or test window. Model-facing histories use the same numbering; dates outside that window or without a matching trading step are masked as `step XXX`. This applies to open-weight, hosted and A1–A4 calls. Price/news alignment and saved experiment dates retain the real calendar; no extra command or configuration is needed.

### Client setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
```

For the local, unauthenticated vLLM server below, **no edits to `.env` and no API key are needed**. For hosted models, set only the corresponding `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or `GEMINI_API_KEY` in the local `.env`. Provider-specific `*_BASE_URL` entries are optional endpoint overrides. For an authenticated open-weight endpoint, set `OPEN_SOURCE_API_KEY` (or the family-specific key). Keep credentials local.

### Llama and Qwen: start a server

Use a separate terminal on a Linux machine with supported NVIDIA GPUs. Choose one exact model ID from the table. Gated Llama weights require approved access and a locally configured Hugging Face token before starting the server.

| Configuration | Model ID | Default GPU parallelism |
|---|---|---:|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 1 |
| Llama 3.1 70B | `meta-llama/Llama-3.1-70B-Instruct` | 4 |
| Qwen3 4B, thinking / no-think | `Qwen/Qwen3-4B` | 1 |
| Qwen3 32B, thinking / no-think | `Qwen/Qwen3-32B` | 2 |

These are launcher defaults; GPU memory must accommodate the weights and context. Set `TENSOR_PARALLEL_SIZE` to override the GPU count when needed.

```bash
python3.11 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install 'vllm==0.11.2'

# Replace this with any model ID from the table.
MODEL='Qwen/Qwen3-4B'
bash scripts/serve_model.sh "$MODEL" "runs/server-${MODEL//\//-}"
```

The launcher pins both weights and tokenizer to an exact revision, saves it in the server directory, and reuses it on restart. Stop the server with `Ctrl+C` before serving a different model on the same port.

### Llama and Qwen: check and run

In the client terminal, set `MODEL` to the same ID as the running server. The following block runs five seeds under **both Beta and No-Beta**. For a single seed, replace the `SEEDS` line with `SEEDS=(1103)`. Use `--sizing beta` or `--sizing nobeta` instead of `--sizing both` to run only one exposure.

```bash
set -euo pipefail
source .venv/bin/activate
MODEL='Qwen/Qwen3-4B'
BASE_URL='http://127.0.0.1:8000/v1'
SEEDS=(1103 2207 3301 4409 5501)
BATCH="runs/${MODEL//\//-}"

# The returned model list must contain MODEL before continuing.
curl -fsS "$BASE_URL/models"

run_open_batch() {
    local mode="$1"
    shift
    python -m experiments.run_main "$MODEL" "$@" \
        --base-url "$BASE_URL" --check-only \
        --output-dir "${BATCH}-${mode}-check"
    python -m experiments.run_main "$MODEL" "$@" \
        --base-url "$BASE_URL" --seeds "${SEEDS[@]}" --sizing both \
        --seed-attempts 3 --retry-delay 120 \
        --output-dir "${BATCH}-${mode}"
}

case "$MODEL" in
    Qwen/*)
        run_open_batch thinking --thinking on
        run_open_batch no-think --thinking off
        ;;
    meta-llama/*)
        run_open_batch instruct
        ;;
    *) printf 'Choose a model ID from the table above.\n' >&2; exit 1 ;;
esac
```

Repeat the server and client blocks with each desired model ID. Qwen's **thinking** (`--thinking on`) and **no-think** (`--thinking off`) use the same weights but separate output directories. The launcher enables `--reasoning-parser qwen3` for **both** modes: reasoning must arrive in a separate server response field, and only the final answer is parsed for decisions. The connectivity check validates this contract; disabling thinking does not remove the server-side parser. Llama does not use the Qwen thinking switch.

For an existing compatible inference service, skip server installation and set `BASE_URL` to its endpoint. `--base-url` overrides `.env` endpoint settings for that invocation.

### Hosted models: one or multiple names

After setting the required keys in `.env`, run the block below. It checks every requested model before its batch. These examples explicitly use **one seed** and both exposures; change `SEEDS` to `(1103 2207 3301 4409 5501)` for five seeds. Run only the provider groups whose keys you configured.

```bash
set -euo pipefail
source .venv/bin/activate
SEEDS=(1103)

run_hosted_batch() {
    local batch="$1"
    shift
    python -m experiments.run_main "$@" --check-only \
        --output-dir "runs/${batch}-check"
    python -m experiments.run_main "$@" \
        --seeds "${SEEDS[@]}" --sizing both \
        --seed-attempts 3 --retry-delay 120 \
        --output-dir "runs/${batch}"
}

# Multiple models in one batch; requires OPENAI_API_KEY.
run_hosted_batch openai \
    gpt-4.1-mini gpt-5.4-mini gpt-5.4-nano o4-mini

# One model; requires DEEPSEEK_API_KEY.
run_hosted_batch deepseek deepseek-v4.1-flash

# Multiple models; requires GEMINI_API_KEY.
run_hosted_batch gemini gemini-3.1-flash-lite gemini-3.8-flash
```

Model IDs must be available at the configured endpoint; the runner does not substitute unavailable aliases. Gemini defaults to low thinking effort and a 4,096-token output cap; DeepSeek defaults to thinking disabled. OpenAI uses the provider's reasoning default unless `OPENAI_REASONING_EFFORT` is set. The Qwen-only `--thinking` flag is not used for hosted models.

### A1–A4 model calls

The ablation runner uses GPT-4.1-mini/Beta and GPT-5.4-mini/No-Beta, so configure `OPENAI_API_KEY` first. This block runs both settings with one seed:

```bash
set -euo pipefail
source .venv/bin/activate
python -m experiments.run_ablations \
    --settings gpt41_beta gpt54_nobeta --check-only
python -m experiments.run_ablations \
    --settings gpt41_beta gpt54_nobeta \
    --ablations a1 a2 a3 a4 --seeds 1103 \
    --attempts 3 --retry-delay 120 --output-dir runs/ablations
```

### Completion and recovery

`--check-only` performs a bounded inference probe (at most 1,024 output tokens per model), without training. A successful main check prints `CHECK_OK` with `train_steps=27` and `test_steps=75`; completed main batches print `BATCH_OK`. Results, figures and logs are saved under the chosen `runs/` directory and remain local.

Rerun the **same command with the same output directory** to resume: completed seeds are skipped; an unfinished main seed restarts. Transient failures retry up to three times at each retry layer, with 120-second waits. Authentication and invalid-response errors require fixing the endpoint. Use a new output directory when changing models, seeds, exposure, thinking mode or other run settings. Ablations also reuse completed training and control checkpoints.

The main input and price caches are bundled, including final-settlement prices. A1/A2 may download public data and missing historical prices; each batch freezes its price snapshot for resumed runs. See [additional data and baseline commands](docs/RUNNING.md) for passive, EG+Z, EG-DDQN and FinMem runs.
