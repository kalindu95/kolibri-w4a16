# Kolibri-1 → 4-bit (GPTQ W4A16, calibrated)

Turns `Aleph-Alpha/Kolibri-1` (78B MoE, 3.46B active, released as block-FP8, ~78 GB)
into a calibrated 4-bit checkpoint (~45 GB) in compressed-tensors format for vLLM.

```
Kolibri-1 (FP8, 78 GB) ──01──▶ BF16 (156 GB) ──02 GPTQ──▶ W4A16-G128 (~45 GB) ──03──▶ eval
```

## Hardware

| | Minimum | Recommended |
|---|---|---|
| GPU | 1× 48 GB (L40S, A6000) | 1× H100 / A100 80 GB |
| System RAM | 128 GB (rest spills to disk, slow) | 256 GB+ |
| Disk | 500 GB NVMe | 1 TB NVMe |
| Time | — | ~1 h download/dequant + a few hours calibration |

Why one GPU is enough: calibration runs one decoder layer at a time. The rest of the
model waits in CPU RAM. Per layer the GPU holds about 3 GB of BF16 weights plus about 21 GB of
GPTQ Hessians (384 experts × 3 projections), so it peaks around 30 GB.
More RAM matters more than a second GPU, because the 156 GB BF16 model has to sit
somewhere. A rented H100 with 256 GB RAM (RunPod, Lambda, Vast) for an afternoon is the
practical route.

Disk use at peak: FP8 78 + BF16 156 + output 45 + offload ≈ 300–350 GB.

## Steps

```bash
WORK=/workspace bash run_all.sh       # does 0–2 below
```

**1. Dequantize FP8 → BF16** (`01_dequant_fp8_to_bf16.py`). This step uses CPU only and processes one shard
at a time. It multiplies each 128×128 FP8 block by its scale. It doesn't depend on tensor names: it
pairs every FP8 `*.weight` with its `*.weight_scale_inv` or `*.weight_scale`.
It removes `quantization_config` from config.json.

**2. GPTQ calibration** (`02_quantize_w4a16_gptq.py`)
- Experts: in transformers, Kolibri stores each layer's 384 experts as fused 3D
  tensors that GPTQ can't see. `load_context()` from llm-compressor splits them into
  separate `nn.Linear` modules (the logs say "linearizing").
- `moe_calibrate_all_experts=True` sends every calibration token through every
  expert. That way rarely routed experts also get real calibration, not just rounding.
  The real top-6 routing is still used to produce the layer's output.
- Calibration data: 512 samples × ≤2048 tokens. The default mix is 70% open-perfectblend
  (English chat, code and maths) and 30% German chat from Aya. Kolibri's tokenizer is tuned
  for German, so leaving German out would hurt it most.
  For your own use case (financial text, say), pass real prompts with
  `--calib-jsonl my.jsonl` (`{"messages":[...]}` per line). That's usually the
  biggest single quality improvement.
- Scheme: INT4 symmetric, group size 128, `actorder=static`. The router, lm_head,
  embeddings and norms stay in BF16. `--keep-attn-bf16` also keeps attention in BF16
  (about +2.5 GB). Try it if the eval shows a quality drop.
- If you run out of GPU memory, use `--batch-size 1`. If you run out of RAM, use `--max-cpu-gb 200` (it spills to `--offload-folder`).
- If calibration stops with `linalg.cholesky: ... not positive-definite`, raise the damping with
  `--dampening-frac 0.05` (default 0.01). Layers with large activation outliers trigger this.
  It happened on Qwen3-0.6B's `layers.2.mlp.down_proj`, and 0.05 fixed it.

**3. Evaluate** (don't skip this)
```bash
# quick: perplexity + sample generation, on held-out chat data (not the calib set)
python 03_compare_ppl.py --ref $WORK/Kolibri-1-bf16 --quant $WORK/Kolibri-1-W4A16-G128 \
    --jsonl heldout.jsonl --n 64

# proper: same harness and settings for the FP8 original and the 4-bit model
lm_eval --model vllm \
  --model_args pretrained=$WORK/Kolibri-1-W4A16-G128,max_model_len=8192,kv_cache_dtype=fp8 \
  --tasks gsm8k,arc_challenge,mmlu --apply_chat_template --batch_size auto
# German: check `lm_eval --tasks list | grep -i _de` (e.g. Okapi arc_de / hellaswag_de / m_mmlu_de)
```
Good result: a perplexity increase under about 3–5% and benchmark scores within about 1–2 points of FP8.

**4. Serve**
```bash
pip install aleph-alpha-inference      # Kolibri's vLLM plugin (see model card)
vllm serve $WORK/Kolibri-1-W4A16-G128 --kv-cache-dtype fp8 \
  --reasoning-parser kolibri1 --tool-call-parser kolibri1 --enable-auto-tool-choice \
  --max-model-len 65536
```
This runs on 1× 80 GB GPU, or on 2× 24–32 GB GPUs with `--tensor-parallel-size 2`. vLLM serves W4A16
MoE through its Marlin kernels.

## Known risks

- **The vLLM plugin is untested with 4-bit.** Kolibri needs `aleph-alpha-inference` in vLLM, and
  I couldn't check that its model code accepts compressed-tensors W4A16 experts.
  If it fails to load, transformers still loads the checkpoint (tested), but it's slow.
- **Built from BF16, not native FP8.** Aleph Alpha trained the final RL stage with FP8
  quantization-aware training. The BF16 we start from is exactly that FP8 model, so
  nothing is lost in step 1. But the 4-bit model gets none of that training benefit.
- **Calibration is one layer at a time.** Kolibri's experts can't be traced as separate
  subgraphs in llm-compressor 0.15, so each whole decoder layer is one calibration step.
  This is why the GPU needs about 30 GB rather than less.

## What was tested here

The whole pipeline (fake FP8 checkpoint → 01 → 02 → reload → compare) ran on a tiny
randomly initialised Kolibri-1 model with the same architecture (`tests/make_tiny_kolibri.py`), on CPU, with
transformers 5.21.0, llm-compressor 0.15.0 and compressed-tensors 0.20.1:
- Dequant: keys match the original exactly. The only error is FP8 rounding.
- GPTQ quantized all 275 target linears, including every expert, and saved standard
  `pack-quantized` W4 g128 per-expert tensors. It reloads in transformers.
- KL divergence from the reference output: **GPTQ 0.095 vs naive RTN 0.413**, with top-1
  agreement 0.59 vs 0.25.

The full-size 78B model has not been run.
