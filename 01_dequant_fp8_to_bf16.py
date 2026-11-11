#!/usr/bin/env python3
"""
Step 1 - Convert the released Kolibri-1 checkpoint (block-wise FP8) to plain BF16.

Kolibri-1 ships its linear weights as float8_e4m3fn in 128x128 blocks, each block
with its own scale. GPTQ needs full-precision weights to start from, so we
dequantize once, offline, shard by shard (low RAM use: ~2x the largest shard).

    w_bf16[i, j] = w_fp8[i, j] * scale[i // 128, j // 128]

The script does not hardcode tensor names. For every FP8 tensor `X.weight` it looks
for a sibling `X.weight_scale_inv` (DeepSeek/Qwen-FP8 convention, used by
transformers' FineGrainedFP8) or `X.weight_scale`. Other tensors (embeddings,
lm_head, norms, router) are copied unchanged. Tensor names stay the same, so
transformers' own Kolibri key mapping still applies when loading.

Usage:
    python 01_dequant_fp8_to_bf16.py --src ./Kolibri-1 --dst ./Kolibri-1-bf16
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCALE_SUFFIXES = ("_scale_inv", "_scale")  # appended to "...weight"
FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


def dequant_block(w: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    """Block-wise dequant. Handles edge blocks when dims are not multiples of block."""
    br, bc = block
    rows, cols = w.shape
    s = scale.float()
    if s.numel() == 1:  # per-tensor scale
        return (w.float() * s).to(torch.bfloat16)
    s = s.repeat_interleave(br, dim=0)[:rows].repeat_interleave(bc, dim=1)[:, :cols]
    return (w.float() * s).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local snapshot of Aleph-Alpha/Kolibri-1")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--block", type=int, nargs=2, default=None,
                    help="override block size; default read from config.json (or 128 128)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    qcfg = cfg.get("quantization_config", {}) or {}
    block = tuple(args.block or qcfg.get("weight_block_size") or (128, 128))
    print(f"quantization_config: {qcfg or 'none found'}")
    print(f"block size: {block}")

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        index, shards = None, ["model.safetensors"]

    # Global key -> shard map so scales stored in a different shard are still found.
    key_to_shard = {}
    for sh in shards:
        with safe_open(src / sh, "pt") as f:
            for k in f.keys():
                key_to_shard[k] = sh

    new_weight_map, total_bytes, n_deq, n_copy = {}, 0, 0, 0
    for i, sh in enumerate(shards, 1):
        out = {}
        with safe_open(src / sh, "pt") as f:
            for k in f.keys():
                if k.endswith(SCALE_SUFFIXES) and k.rsplit("_scale", 1)[0] in key_to_shard:
                    continue  # scale tensors are consumed, not copied
                t = f.get_tensor(k)
                if t.dtype in FP8_DTYPES:
                    scale_key = next((k + s for s in SCALE_SUFFIXES if k + s in key_to_shard), None)
                    if scale_key is None:
                        raise KeyError(f"No scale found for FP8 tensor {k}")
                    with safe_open(src / key_to_shard[scale_key], "pt") as fs:
                        scale = fs.get_tensor(scale_key)
                    t = dequant_block(t, scale, block)
                    n_deq += 1
                else:
                    n_copy += 1
                out[k] = t.contiguous()
                new_weight_map[k] = sh
                total_bytes += t.numel() * t.element_size()
        save_file(out, dst / sh, metadata={"format": "pt"})
        print(f"[{i}/{len(shards)}] {sh}: {len(out)} tensors")
        del out

    if index is not None:
        (dst / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}, indent=2))

    # Copy tokenizer / generation files; strip the FP8 quantization_config.
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "model.safetensors.index.json", "config.json"):
            shutil.copy2(p, dst / p.name)
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = cfg["dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"done: {n_deq} FP8 tensors dequantized, {n_copy} copied, "
          f"{total_bytes / 1e9:.1f} GB written to {dst}")


if __name__ == "__main__":
    main()
