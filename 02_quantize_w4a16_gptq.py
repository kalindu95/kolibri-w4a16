#!/usr/bin/env python3
"""
Step 2 - Calibrated 4-bit quantization of Kolibri-1 (GPTQ, W4A16, group size 128).

Input : the BF16 checkpoint from 01_dequant_fp8_to_bf16.py
Output: a compressed-tensors checkpoint (~45 GB) that vLLM loads directly.

How it works
- load_context() "linearizes" Kolibri's fused 3D expert tensors into 384 separate
  ExpertMLP modules (gate/up/down nn.Linear) per layer, so GPTQ can see them, and
  sets up CPU offloading. Only the decoder layer being calibrated sits on the GPU.
- moe_calibrate_all_experts=True (default) runs every calibration token through
  every expert during calibration (the real top-6 routing is still used for the
  layer output), so all 384 experts get a proper Hessian, not just popular ones.
- Kept in BF16: lm_head, embeddings, norms, router (Kolibri1TopKRouter is not an
  nn.Linear so it is never targeted). Optional: --keep-attn-bf16.

Usage:
    python 02_quantize_w4a16_gptq.py --model ./Kolibri-1-bf16 --out ./Kolibri-1-W4A16-G128
    # uncalibrated round-to-nearest baseline, same scheme/ignore list, for comparison:
    python 02_quantize_w4a16_gptq.py --model ./Kolibri-1-bf16 --out ./Kolibri-1-W4A16-RTN --rtn
"""
import argparse
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.utils import load_context


def tokenize_messages(tok, ds, max_len):
    def to_ids(ex):
        try:
            text = tok.apply_chat_template(ex["messages"], tokenize=False)
        except Exception:
            text = "\n\n".join(m["content"] for m in ex["messages"])
        return tok(text, max_length=max_len, truncation=True, add_special_tokens=False)
    ds = ds.map(to_ids, remove_columns=ds.column_names)
    # drop very short samples, they carry little Hessian information
    return ds.filter(lambda ex: len(ex["input_ids"]) >= 64)


def build_calibration_set(tok, n_samples: int, max_len: int, german_frac: float, seed: int):
    """English/multi-domain chat (open-perfectblend) mixed with German chat (Aya).

    Kolibri is tuned for German, so German text should be in the calibration data,
    otherwise the German-heavy parts of the model get poorly calibrated scales.
    """
    n_de = int(n_samples * german_frac)
    n_en = n_samples - n_de
    parts = []

    en = load_dataset("mlabonne/open-perfectblend", split="train").shuffle(seed=seed)
    en = en.select(range(min(len(en), n_en * 2)))

    def en_msgs(ex):
        role = {"human": "user", "gpt": "assistant", "system": "system"}
        return {"messages": [{"role": role.get(t["from"], "user"), "content": t["value"]}
                             for t in ex["conversations"]]}
    parts.append(en.map(en_msgs, remove_columns=en.column_names))

    if n_de:
        try:
            de = load_dataset("CohereForAI/aya_dataset", split="train")
            de = de.filter(lambda ex: ex["language"] == "German").shuffle(seed=seed)
            de = de.select(range(min(len(de), n_de * 2)))
            parts.append(de.map(
                lambda ex: {"messages": [{"role": "user", "content": ex["inputs"]},
                                         {"role": "assistant", "content": ex["targets"]}]},
                remove_columns=de.column_names))
        except Exception as e:  # dataset gated/renamed: fall back to English-only
            print(f"WARNING: German calibration data unavailable ({e}); using English only")
            n_en = n_samples

    tokenized = [tokenize_messages(tok, p, max_len) for p in parts]
    take = [n_en, n_de] if n_de else [n_samples]
    ds = concatenate_datasets([t.select(range(min(len(t), k))) for t, k in zip(tokenized, take)])
    ds = ds.shuffle(seed=seed)
    print(f"calibration set: {len(ds)} samples, "
          f"{sum(len(x) for x in ds['input_ids']):,} tokens")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="BF16 checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--german-frac", type=float, default=0.3)
    ap.add_argument("--calib-jsonl", default=None,
                    help='use your own data instead: one {"messages": [...]} per line '
                         '(e.g. real prompts from your use case)')
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--dampening-frac", type=float, default=0.01,
                    help="GPTQ Hessian damping; raise to 0.05-0.1 if a layer fails with "
                         "'not positive-definite' (layers with activation outliers)")
    ap.add_argument("--rtn", action="store_true",
                    help="round-to-nearest baseline: same scheme and ignore list, no "
                         "calibration data or Hessians (minutes, not hours)")
    ap.add_argument("--keep-attn-bf16", action="store_true",
                    help="leave attention projections unquantized (+~2.5 GB, safer quality)")
    ap.add_argument("--seq-target", default=None,
                    help="decoder-layer class to calibrate one at a time; default: "
                         "taken from the model (Kolibri1DecoderLayer for Kolibri)")
    ap.add_argument("--offload-folder", default="./offload_folder")
    ap.add_argument("--max-cpu-gb", type=float, default=None,
                    help="cap on RAM used for weights; the rest spills to --offload-folder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto_offload",
                       offload_folder=args.offload_folder)
    if args.max_cpu_gb:
        load_kwargs["max_memory"] = {"cpu": args.max_cpu_gb * 1e9}

    with load_context():
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ignore = ["lm_head", "re:.*mlp.gate$"]
    if args.keep_attn_bf16:
        ignore.append("re:.*self_attn\\..*_proj$")
    weights = {"num_bits": 4, "type": "int", "symmetric": True,
               "strategy": "group", "group_size": args.group_size}
    scheme = "W4A16" if args.group_size == 128 else None

    if args.rtn:
        # load_context() above still linearizes the experts, so RTN quantizes them too
        quant = dict(scheme=scheme) if scheme else dict(
            config_groups={"group_0": {"targets": ["Linear"], "weights": weights}})
        oneshot(model=model, recipe=QuantizationModifier(targets="Linear", ignore=ignore, **quant))
        model.save_pretrained(args.out, save_compressed=True)
        tok.save_pretrained(args.out)
        print(f"saved (RTN) -> {args.out}")
        return

    if args.calib_jsonl:
        ds = tokenize_messages(tok, load_dataset("json", data_files=args.calib_jsonl, split="train"),
                               args.max_len).shuffle(seed=args.seed)
        ds = ds.select(range(min(len(ds), args.samples)))
        print(f"calibration set (jsonl): {len(ds)} samples")
    else:
        ds = build_calibration_set(tok, args.samples, args.max_len, args.german_frac, args.seed)

    seq_targets = [args.seq_target] if args.seq_target else list(model._no_split_modules)
    print(f"sequential targets: {seq_targets}")

    if scheme:
        recipe = GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore,
                              dampening_frac=args.dampening_frac, actorder="static")
    else:
        recipe = GPTQModifier(
            targets="Linear", ignore=ignore, dampening_frac=args.dampening_frac,
            actorder="static",
            config_groups={"group_0": {"targets": ["Linear"], "weights": weights}})

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        batch_size=args.batch_size,
        max_seq_length=args.max_len,
        num_calibration_samples=len(ds),
        pipeline="sequential",
        # one decoder layer per subgraph (~3 GB weights + ~21 GB Hessians on GPU)
        sequential_targets=seq_targets,
        moe_calibrate_all_experts=True,
    )

    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
