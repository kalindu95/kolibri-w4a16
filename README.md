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
- Baseline for comparison: `--rtn` writes an uncalibrated round-to-nearest checkpoint with the
  same scheme, group size and ignore list (`--out $WORK/Kolibri-1-W4A16-RTN --rtn`). It takes
  minutes, needs no calibration data, and still linearizes the experts, so they get quantized too.
  Evaluate it the same way as the GPTQ model. GPTQ should come out clearly better.

**3. Evaluate** (don't skip this)

`run_eval.sh` compares the GPTQ and RTN checkpoints with the original model
(build RTN first with `02_quantize_w4a16_gptq.py --rtn`). Needs vLLM, `aleph-alpha-inference`
and `lm-eval[vllm]`.

The defaults fit **one A100 80 GB**:
- The 4-bit models (~45 GB) run in vLLM.
- The reference doesn't fit, so the fidelity stage scores the **BF16 copy** once through
  transformers. About 70 GB of it stays on the GPU and the rest goes to CPU RAM, which needs
  about 100 GB free. Keep `$WORK/Kolibri-1-bf16` until the evaluation is done.
- Reference benchmarks are off. Compare GPTQ with RTN, and with the scores on the model card.
  `BENCH_REF=1 LIMIT=200` runs them on the FP8 model in vLLM with 24 GB of weights in CPU RAM.
  That's slow and untested on A100.
- With 2+ GPUs: `REF=$WORK/Kolibri-1 REF_BACKEND=vllm REF_TP=2 BENCH_REF=1`.
```bash
WORK=/workspace bash run_eval.sh                               # data, fidelity, bench, summary
EXTRA_DATA="fin=/data/fin_heldout.jsonl" STAGES=fidelity bash run_eval.sh   # add a financial slice
LIMIT=100 STAGES=bench bash run_eval.sh                        # quick benchmark pass first
```
| Stage | Script | What it measures |
|---|---|---|
| data | `make_heldout.py` | English (open-perfectblend, rows after the calibration ones) and German (oasst2) held-out chat, 128 each |
| fidelity | `04_eval_fidelity.py` | Checkpoint check (size, packed tensor and expert counts), then perplexity, KL(ref‖model) and top-1 agreement per slice, plus the worst samples |
| bench | `lm_eval` on vLLM | gsm8k, arc_challenge, mmlu, ifeval and German arc_de, hellaswag_de, m_mmlu_de, with identical settings for every model |
| summary | | `$WORK/eval/benchmarks.md` and `$WORK/eval/fidelity/results.md` |

Good result: perplexity up less than about 3–5%, benchmarks within about 1–2 points of FP8,
and GPTQ clearly better than RTN on KL. If GPTQ and RTN come out about equal, the experts
probably weren't calibrated.

Notes:
- KL is a top-k approximation (vLLM returns only the top log-probs). On Qwen3-0.6B it read
  about 13% below the exact value for both GPTQ and RTN, so model-to-model comparisons hold.
  Perplexity is exact.
- Kolibri is a reasoning model: `MAX_GEN_TOKS` (default 8192) must leave room to think,
  otherwise gsm8k answers get cut off and every model scores low.
- Don't put German Aya rows in a held-out set: Aya has only ~241 German rows and the default
  calibration can use all of them.
- `03_compare_ppl.py` still works for a quick check without vLLM, but it loads models
  through transformers, which is slow at this size. `04_eval_fidelity.py --backend hf --device cpu`
  runs the same metrics on small models without vLLM.

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
