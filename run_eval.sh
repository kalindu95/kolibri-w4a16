#!/usr/bin/env bash
# Evaluate the 4-bit models against the original Kolibri-1. Run on the GPU box; see README.md.
#   data      held-out English/German slices (make_heldout.py)
#   fidelity  ppl, KL, top-1 vs the reference on held-out data (04_eval_fidelity.py)
#   bench     lm-eval benchmarks, identical settings for every model
#   summary   one table of benchmark scores
# Each stage skips work that already has results, so rerunning after a failure is cheap.
#
# Defaults fit ONE A100 80 GB (container): the 4-bit models run in vLLM; the reference
# (BF16 copy, 156 GB) is scored once through transformers with ~70 GB on the GPU and the
# rest in CPU RAM (needs ~100 GB free RAM, else it spills to disk and gets very slow).
# Keep $WORK/Kolibri-1-bf16 until this is done. Reference benchmarks are off by default
# (BENCH_REF=1 runs them on the FP8 model in vLLM with CPU offload: slow, use LIMIT).
# Several GPUs: REF=$WORK/Kolibri-1 REF_BACKEND=vllm REF_TP=2 BENCH_REF=1.
#
#   WORK=/workspace bash run_eval.sh                    # all stages
#   STAGES="fidelity" bash run_eval.sh                  # just one
#   EXTRA_DATA="fin=/data/fin_heldout.jsonl" bash run_eval.sh    # add a financial slice
set -euo pipefail
cd "$(dirname "$0")"

WORK=${WORK:-/workspace}
OUT=${OUT:-$WORK/eval}
STAGES=${STAGES:-"data fidelity bench summary"}

# models: name=path
REF=${REF:-$WORK/Kolibri-1-bf16}  # reference for fidelity; same weights as the FP8 release
REF_BACKEND=${REF_BACKEND:-hf}    # hf = transformers with CPU offload (one GPU)
REF_TP=${REF_TP:-1}
REF_MAX_GPU_GB=${REF_MAX_GPU_GB:-70}   # hf reference: weights on GPU, rest in CPU RAM
BATCH_TOKENS=${BATCH_TOKENS:-16384}    # hf reference: padded tokens per pass; lower on OOM
BENCH_REF=${BENCH_REF:-0}              # 1 = also benchmark the reference
REF_BENCH=${REF_BENCH:-$WORK/Kolibri-1}          # FP8 release, 78 GB
REF_CPU_OFFLOAD_GB=${REF_CPU_OFFLOAD_GB:-24}     # one A100: FP8 weights kept in CPU RAM
MODELS=${MODELS:-"gptq=$WORK/Kolibri-1-W4A16-G128 rtn=$WORK/Kolibri-1-W4A16-RTN"}
TP=${TP:-1}

N=${N:-128}                       # samples per held-out slice
EXTRA_DATA=${EXTRA_DATA:-}        # more slices, e.g. "fin=/path/fin.jsonl"
TASKS=${TASKS:-"gsm8k,arc_challenge,mmlu,ifeval,arc_de,hellaswag_de,m_mmlu_de"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-8192}  # reasoning model: leave room to think before answering
LIMIT=${LIMIT:-}                  # e.g. LIMIT=200 for a quick pass (per task)
KV_DTYPE=${KV_DTYPE:-auto}        # fp8 needs Hopper/Ada; keep auto on A100
GPU_UTIL=${GPU_UTIL:-0.9}

export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
has() { [[ " $STAGES " == *" $1 "* ]]; }

if command -v nvidia-smi > /dev/null; then
  echo "== GPUs"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
free_gb=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo 2>/dev/null || echo 0)
if [[ $REF_BACKEND == hf && $free_gb -gt 0 && $free_gb -lt 100 ]] && has fidelity; then
  echo "WARNING: ${free_gb} GB RAM free; the offloaded reference needs ~$((160 - REF_MAX_GPU_GB + 10)) GB."
  echo "         The rest spills to disk ($OUT/offload) and scoring gets much slower."
fi

if has data; then
  echo "== data"
  if [[ -s "$OUT/data/en.jsonl" && -s "$OUT/data/de.jsonl" ]]; then
    echo "held-out data exists, skipping"
  else
    python make_heldout.py --out "$OUT/data" --n "$N"
  fi
fi

if has fidelity; then
  echo "== fidelity"
  args=(--ref "$REF" --ref-backend "$REF_BACKEND" --ref-tp "$REF_TP" --tp "$TP" --n "$N"
        --max-gpu-gb "$REF_MAX_GPU_GB" --batch-tokens "$BATCH_TOKENS"
        --offload-folder "$OUT/offload" --out "$OUT/fidelity"
        --gpu-util "$GPU_UTIL" --kv-cache-dtype "$KV_DTYPE"
        --data "en=$OUT/data/en.jsonl" --data "de=$OUT/data/de.jsonl")
  for m in $MODELS; do args+=(--model "$m"); done
  for d in $EXTRA_DATA; do args+=(--data "$d"); done
  python 04_eval_fidelity.py "${args[@]}" 2>&1 | tee "$OUT/fidelity.log"
fi

if has bench; then
  echo "== bench"
  specs=$MODELS
  if [[ $BENCH_REF == 1 ]]; then
    specs="reference=$REF_BENCH $MODELS"
    [[ -z $LIMIT ]] && echo "NOTE: reference benchmarks without LIMIT can take days with CPU offload"
  fi
  for spec in $specs; do
    name=${spec%%=*}; path=${spec#*=}
    tp=$TP; [[ $name == reference ]] && tp=$REF_TP
    dst="$OUT/lm_eval/$name"
    if compgen -G "$dst/*/results_*.json" > /dev/null; then   # lm_eval writes <dst>/<model>/results_*.json
      echo "[$name] results exist, skipping"; continue
    fi
    margs="pretrained=$path,tensor_parallel_size=$tp,max_model_len=$MAX_MODEL_LEN"
    margs+=",gpu_memory_utilization=$GPU_UTIL,enable_prefix_caching=True"
    [[ $KV_DTYPE != auto ]] && margs+=",kv_cache_dtype=$KV_DTYPE"
    [[ $name == reference && $REF_TP == 1 ]] && margs+=",cpu_offload_gb=$REF_CPU_OFFLOAD_GB"
    echo "[$name] lm_eval on $path (tp=$tp)"
    lm_eval --model vllm --model_args "$margs" --tasks "$TASKS" \
      --apply_chat_template --fewshot_as_multiturn --batch_size auto \
      --gen_kwargs "max_gen_toks=$MAX_GEN_TOKS" --seed 1234 \
      ${LIMIT:+--limit "$LIMIT"} --log_samples --output_path "$dst" 2>&1 | tee "$OUT/lm_eval_$name.log"
  done
fi

if has summary; then
  echo "== summary"
  python - "$OUT" "$TASKS" "reference $(for m in $MODELS; do printf '%s ' "${m%%=*}"; done)" <<'EOF'
import glob, json, sys
from pathlib import Path
out, tasks, names = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split()
res = {}
for n in names:
    files = sorted(glob.glob(str(out / "lm_eval" / n / "**" / "results_*.json"), recursive=True))
    if files:
        res[n] = json.load(open(files[-1]))["results"]
if not res:
    sys.exit("no lm_eval results yet")
lines = ["| task | metric | " + " | ".join(res) + " |", "|---|---|" + "---|" * len(res)]
for t in tasks:
    first = next((r[t] for r in res.values() if t in r), None)
    if first is None:
        continue
    for metric in first:
        if metric == "alias" or "_stderr" in metric or not isinstance(first[metric], (int, float)):
            continue
        base, _, filt = metric.partition(",")
        cells = []
        for r in res.values():
            v, se = r.get(t, {}).get(metric), r.get(t, {}).get(f"{base}_stderr,{filt}")
            cells.append("—" if v is None else
                         f"{100 * v:.2f}" + (f" ± {100 * se:.2f}" if isinstance(se, float) else ""))
        lines.append(f"| {t} | {metric} | " + " | ".join(cells) + " |")
table = "\n".join(lines)
(out / "benchmarks.md").write_text(table + "\n")
print(table)
print("\nA gap smaller than the ± (standard error) is noise. Good 4-bit result: within ~1-2 points.")
if "reference" not in res:
    print("No reference benchmarks (BENCH_REF=0): compare GPTQ with RTN here, and with the FP8\n"
          "scores on the Kolibri-1 model card (their harness settings may differ).")
EOF
  [[ -f "$OUT/fidelity/results.md" ]] && { echo; echo "fidelity:"; cat "$OUT/fidelity/results.md"; }
fi
echo "Done. Results in $OUT"
