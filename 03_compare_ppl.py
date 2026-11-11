#!/usr/bin/env python3
"""
Step 3 - Sanity check: perplexity of reference vs quantized model on held-out chat text,
plus a short greedy generation. Runs through transformers (slow but simple); for real
benchmarks use lm-eval with vLLM (see README).

Usage:
    python 03_compare_ppl.py --ref ./Kolibri-1-bf16 --quant ./Kolibri-1-W4A16-G128 \
        --jsonl heldout.jsonl --n 64
    (--ref can also be the original FP8 repo if your GPUs support FP8)
"""
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def ppl(model, tok, texts, max_len):
    nll, ntok = 0.0, 0
    dev = next(model.parameters()).device
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len,
                  add_special_tokens=False).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids)
        nll += out.loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return math.exp(nll / ntok)


def load_texts(path, tok, n):
    texts = []
    with open(path) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            try:
                texts.append(tok.apply_chat_template(msgs, tokenize=False))
            except Exception:
                texts.append("\n\n".join(m["content"] for m in msgs))
            if len(texts) >= n:
                break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--jsonl", required=True, help="held-out data, NOT the calibration set")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--prompt", default="Erkläre kurz, was Quantisierung bei LLMs bedeutet.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ref)
    texts = load_texts(args.jsonl, tok, args.n)
    res = {}
    for name, path in [("reference", args.ref), ("quantized", args.quant)]:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", dtype="auto")
        model.eval()
        res[name] = ppl(model, tok, texts, args.max_len)
        msgs = [{"role": "user", "content": args.prompt}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)["input_ids"]
        except Exception:
            ids = tok(args.prompt, return_tensors="pt").input_ids
        gen = model.generate(ids.to(model.device), max_new_tokens=64, do_sample=False)
        print(f"\n[{name}] ppl={res[name]:.3f}\n{tok.decode(gen[0, ids.shape[1]:])}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    delta = 100 * (res["quantized"] / res["reference"] - 1)
    print(f"\nPPL reference {res['reference']:.3f} | quantized {res['quantized']:.3f} "
          f"| +{delta:.2f}%")
    print("Rule of thumb: < ~3-5% PPL increase is a good 4-bit result; >10% means something's off.")


if __name__ == "__main__":
    main()
