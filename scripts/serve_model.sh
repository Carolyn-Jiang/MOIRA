#!/usr/bin/env bash
# Run in the vLLM environment. Pin weights + tokenizer and retain every server attempt.
set -euo pipefail
model=${1:?Usage: bash scripts/serve_model.sh MODEL LOG_DIRECTORY}
output=${2:?Supply a server log directory under runs/}
case "$model" in
  meta-llama/Llama-3.1-8B-Instruct) tp=1 ;;
  meta-llama/Llama-3.1-70B-Instruct) tp=4 ;;
  Qwen/Qwen3-4B) tp=1 ;;
  Qwen/Qwen3-32B) tp=2 ;;
  *) echo "Unsupported experiment model: $model" >&2; exit 2 ;;
esac
tp=${TENSOR_PARALLEL_SIZE:-$tp}
mkdir -p "$output"
revision=$(python - "$model" "$output/revision.json" <<'PY'
import json, re, sys
from pathlib import Path
model, target = sys.argv[1], Path(sys.argv[2])
if target.exists():
    saved = json.loads(target.read_text())
    if saved['model'] != model:
        raise SystemExit('This server directory belongs to another model')
    sha = saved['revision']
else:
    from huggingface_hub import HfApi
    sha = HfApi().model_info(model).sha
if not isinstance(sha, str) or not re.fullmatch(r'[a-f0-9]{40}', sha):
    raise SystemExit('Expected an exact Hugging Face commit SHA')
if not target.exists():
    target.write_text(json.dumps({'model': model, 'revision': sha}, indent=2) + '\n')
print(sha)
PY
)
attempt="$output/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt"
python -m pip freeze > "$attempt/environment.txt"
generation='{"temperature":0.6,"top_p":0.95,"top_k":20,"repetition_penalty":1.0}'
command=(vllm serve "$model" --host 127.0.0.1 --port 8000 --dtype bfloat16
         --revision "$revision" --tokenizer-revision "$revision"
         --max-model-len 32768 --tensor-parallel-size "$tp"
         --generation-config auto --override-generation-config "$generation")
if [[ "$model" == Qwen/* ]]; then
  command+=(--reasoning-parser qwen3)
fi
printf '%q ' "${command[@]}" > "$attempt/command.txt"
printf '\n' >> "$attempt/command.txt"
printf 'SERVER_START model=%s revision=%s logs=%s\n' "$model" "$revision" "$attempt"
"${command[@]}" 2>&1 | tee "$attempt/server.log"
