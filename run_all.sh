#!/usr/bin/env bash
# Kolibri-1 -> W4A16 (GPTQ, group 128). Run on the GPU box; see README.md.
set -euo pipefail
WORK=${WORK:-/workspace}          # put this on a fast NVMe with >= 500 GB free
cd "$(dirname "$0")"

pip install -q "llmcompressor==0.15.0" "transformers==5.21.0" "compressed-tensors==0.20.1" \
               "datasets" "hf_transfer"
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false

# 0. download the released FP8 weights (~78 GB)
hf download Aleph-Alpha/Kolibri-1 --local-dir "$WORK/Kolibri-1"

# 1. FP8 -> BF16 (~156 GB, CPU only, ~20-40 min)
python 01_dequant_fp8_to_bf16.py --src "$WORK/Kolibri-1" --dst "$WORK/Kolibri-1-bf16"

# 2. GPTQ W4A16 calibration (1 GPU, a few hours)
python 02_quantize_w4a16_gptq.py \
    --model "$WORK/Kolibri-1-bf16" \
    --out   "$WORK/Kolibri-1-W4A16-G128" \
    --offload-folder "$WORK/offload" \
    --samples 512 --max-len 2048 --german-frac 0.3 2>&1 | tee "$WORK/quantize.log"

du -sh "$WORK/Kolibri-1-W4A16-G128"
echo "Done. Next: evaluate (README step 3), then delete $WORK/Kolibri-1-bf16 and $WORK/offload."
