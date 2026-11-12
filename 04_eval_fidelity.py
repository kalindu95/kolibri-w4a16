#!/usr/bin/env python3
"""
Step 4 - How closely each quantized model follows the reference, on held-out data.

Every model scores exactly the same token sequences (teacher forcing), and at every
position its next-token distribution is compared with the reference's. This is the
most sensitive quality signal for quantization and far cheaper than benchmarks.

Per data slice and model:
  ppl        perplexity of the held-out text            (good: < +3-5% vs reference)
  kl         mean per-token KL(reference || model)      (lower is better)
  top1       share of positions where the most likely next token equals the reference's
  kl_p95     95th percentile of per-sample mean KL      (are a few samples hit hard?)

Before scoring it also inspects each checkpoint (size, quantization scheme, number of
packed tensors and how many belong to experts), so a broken export shows up first.

How it runs: each model is scored in its own subprocess (vLLM only frees GPU memory
reliably on exit), one after another. Results per model are cached in --out; rerunning
only scores what's new.

One GPU: the 4-bit models (~45 GB) run in vLLM. The reference doesn't fit, so score it
with --ref-backend hf from the BF16 copy: transformers keeps --max-gpu-gb of it on the
GPU and the rest in CPU RAM (or --offload-folder on disk), and samples are batched so
each pass over the offloaded weights scores many tokens. Slow, but it runs once.
Alternatively --ref-cpu-offload-gb runs the FP8 reference in vLLM with part of its
weights in CPU RAM.

KL is a top-k approximation: vLLM returns only the top log-probs per position. KL is
taken over the reference's top --topk-ref tokens plus one bucket for the rest. If one
of those tokens is missing from the model's own top --topk-model list, its log-prob is
set to the model's lowest listed value (an upper bound); the share of such misses is
reported. The result slightly underestimates the true KL, equally for every model.

Usage (one A100 80 GB, vLLM + aleph-alpha-inference installed):
    python 04_eval_fidelity.py --ref $WORK/Kolibri-1-bf16 --ref-backend hf --max-gpu-gb 70 \\
        --model gptq=$WORK/Kolibri-1-W4A16-G128 --model rtn=$WORK/Kolibri-1-W4A16-RTN \\
        --data en=$WORK/eval/data/en.jsonl --data de=$WORK/eval/data/de.jsonl \\
        --data fin=my_financial_heldout.jsonl --out $WORK/eval/fidelity
    # several GPUs: --ref $WORK/Kolibri-1 --ref-backend vllm --ref-tp 2
    # no vLLM (small models, CPU): --backend hf --ref-backend hf --device cpu
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def pairs(items, flag):
    out = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or not name:
            sys.exit(f"{flag} expects name=path, got {it!r}")
        out[name] = path
    return out


# ---------------------------------------------------------------- checkpoint inspection
def inspect_checkpoint(path):
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    info = {"size_gb": round(sum(os.path.getsize(f) for f in files) / 1e9, 2)}
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        keys = list(json.load(open(idx))["weight_map"])
    else:
        keys = [k for f in files for k in safe_open(f, "pt").keys()]
    packed = [k for k in keys if k.endswith("weight_packed")]
    info["packed_tensors"] = len(packed)
    info["packed_experts"] = sum(".experts." in k for k in packed)
    qc = json.load(open(os.path.join(path, "config.json"))).get("quantization_config")
    if qc:
        info["quant_method"] = qc.get("quant_method")
        groups = qc.get("config_groups") or {}
        w = next(iter(groups.values()), {}).get("weights") or {}
        if w:
            info["weights"] = f"int{w.get('num_bits')} {w.get('strategy')} g{w.get('group_size')}"
        info["ignored_modules"] = len(qc.get("ignore") or [])
    return info


# ---------------------------------------------------------------- inputs
def build_inputs(tok_path, data, n, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    samples = []
    for name, path in data.items():
        k = ntok = 0
        for line in open(path):
            if k >= n:
                break
            msgs = json.loads(line)["messages"]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False)
            except Exception:
                text = "\n\n".join(m["content"] for m in msgs)
            ids = tok(text, truncation=True, max_length=max_len,
                      add_special_tokens=False).input_ids
            if len(ids) < 2:
                continue
            samples.append({"slice": name, "ids": ids})
            k += 1
            ntok += len(ids) - 1
        print(f"data {name}: {k} samples, {ntok:,} scored tokens")
    return samples


# ---------------------------------------------------------------- scoring (subprocess)
def score(args):
    """Score one model; write target log-probs and top-k log-probs per position."""
    samples = json.loads(Path(args.inputs).read_text())
    k = args.k
    tgt, top_ids, top_lp = [], [], []
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        kw = dict(model=args.path, tensor_parallel_size=args.tp,
                  max_model_len=max(len(s["ids"]) for s in samples) + 8, max_logprobs=k,
                  gpu_memory_utilization=args.gpu_util, enable_prefix_caching=False)
        if args.kv_cache_dtype != "auto":
            kw["kv_cache_dtype"] = args.kv_cache_dtype
        if args.cpu_offload_gb:
            kw["cpu_offload_gb"] = args.cpu_offload_gb
        llm = LLM(**kw)
        outs = llm.generate([TokensPrompt(prompt_token_ids=s["ids"]) for s in samples],
                            SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=k))
        for s, o in zip(samples, outs):
            for pos in range(1, len(s["ids"])):
                d = o.prompt_logprobs[pos]
                tgt.append(d[s["ids"][pos]].logprob)
                # d holds the top-k plus the actual token (which may rank lower)
                top = sorted((v.rank, tid, v.logprob) for tid, v in d.items() if v.rank <= k)[:k]
                top_ids.append([t for _, t, _ in top])
                top_lp.append([lp for _, _, lp in top])
        tgt = np.asarray(tgt, np.float32)
        top_ids = np.asarray(top_ids, np.int32)
        top_lp = np.asarray(top_lp, np.float32)
    else:
        import time
        import psutil
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM
        kw = dict(dtype="auto", device_map=args.device)
        if args.device == "auto" and args.max_gpu_gb and torch.cuda.is_available():
            cpu_gb = args.max_cpu_gb or 0.9 * psutil.virtual_memory().available / 2**30
            kw["max_memory"] = {0: f"{args.max_gpu_gb}GiB", "cpu": f"{cpu_gb:.0f}GiB"}
            kw["offload_folder"] = args.offload_folder
            print(f"  hf offload: {kw['max_memory']}, spill to {args.offload_folder}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(args.path, **kw).eval()
        dev = torch.device("cuda:0") if args.device == "auto" and torch.cuda.is_available() \
            else next(model.parameters()).device

        # longest first, packed into batches of <= batch_tokens (padded) tokens, so each
        # pass over offloaded weights scores many samples. Right padding: real tokens never
        # attend to pads, so results equal unbatched scoring.
        order = sorted(range(len(samples)), key=lambda i: -len(samples[i]["ids"]))
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) + 1) * len(samples[cur[0]]["ids"]) > args.batch_tokens:
                batches.append(cur)
                cur = []
            cur.append(i)
        batches.append(cur)
        res = {}
        t0 = time.time()
        with torch.no_grad():
            for b, idx in enumerate(batches):
                width = len(samples[idx[0]]["ids"])
                ids = torch.zeros(len(idx), width, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for r, i in enumerate(idx):
                    n = len(samples[i]["ids"])
                    ids[r, :n] = torch.tensor(samples[i]["ids"])
                    mask[r, :n] = 1
                logits = model(ids.to(dev), attention_mask=mask.to(dev)).logits
                for r, i in enumerate(idx):
                    n = len(samples[i]["ids"])
                    lp = F.log_softmax(logits[r, :n - 1].float(), -1)
                    t = ids[r, 1:n, None].to(lp.device)
                    v, ix = lp.topk(k, -1)
                    res[i] = (lp.gather(1, t)[:, 0].cpu().numpy(), ix.int().cpu().numpy(),
                              v.cpu().numpy())
                del logits
                print(f"  batch {b + 1}/{len(batches)} ({len(idx)} samples x {width} tokens) "
                      f"{time.time() - t0:.0f}s", flush=True)
        tgt, top_ids, top_lp = (np.concatenate([res[i][j] for i in range(len(samples))])
                                for j in range(3))
    np.savez(args.to, tgt=tgt, top_ids=top_ids, top_lp=top_lp)


def run_scoring(name, path, tp, k, a, inputs, out, backend, cpu_offload_gb=0):
    dump = out / f"scores_{name}.npz"
    stamp = out / f"scores_{name}.key"
    # rescore when the model path, top-k, backend or the input tokens change
    key = hashlib.sha1(f"{path}|{k}|{backend}|".encode() + inputs.read_bytes()).hexdigest()
    if dump.exists() and stamp.exists() and stamp.read_text() == key and not a.redo:
        print(f"[{name}] cached scores, skipping")
        return np.load(dump)
    print(f"[{name}] scoring {path} (tp={tp}, top-{k}, backend={backend})", flush=True)
    cmd = [sys.executable, __file__, "--_score", path, "--_inputs", str(inputs), "--_to", str(dump),
           "--_k", str(k), "--_tp", str(tp), "--backend", backend, "--device", a.device,
           "--gpu-util", str(a.gpu_util), "--kv-cache-dtype", a.kv_cache_dtype,
           "--batch-tokens", str(a.batch_tokens), "--offload-folder", a.offload_folder,
           "--_cpu_offload_gb", str(cpu_offload_gb)]
    for flag, v in [("--max-gpu-gb", a.max_gpu_gb), ("--max-cpu-gb", a.max_cpu_gb)]:
        if v:
            cmd += [flag, str(v)]
    subprocess.run(cmd, check=True)
    stamp.write_text(key)
    return np.load(dump)


# ---------------------------------------------------------------- metrics
def per_token_kl(ref, mod, k_ref, chunk=4096):
    r_ids, r_lp = ref["top_ids"][:, :k_ref], ref["top_lp"][:, :k_ref].astype(np.float64)
    m_ids, m_lp = mod["top_ids"], mod["top_lp"].astype(np.float64)
    kl = np.empty(len(r_ids))
    miss = 0
    for s in range(0, len(r_ids), chunk):
        e = s + chunk
        match = r_ids[s:e, :, None] == m_ids[s:e, None, :]           # [n, k_ref, k_model]
        found = match.any(-1)
        q_lp = np.where(found, np.where(match, m_lp[s:e, None, :], 0).sum(-1), m_lp[s:e, -1:])
        p = np.exp(r_lp[s:e])
        p_rest = np.clip(1 - p.sum(1), 1e-12, None)
        q_rest = np.clip(1 - np.exp(q_lp).sum(1), 1e-12, None)
        kl[s:e] = (p * (r_lp[s:e] - q_lp)).sum(1) + p_rest * (np.log(p_rest) - np.log(q_rest))
        miss += (~found).sum()
    return np.clip(kl, 0, None), miss / r_ids.size


def summarize(samples, scores, ref_name, k_ref):
    lens = np.array([len(s["ids"]) - 1 for s in samples])
    offs = np.concatenate([[0], np.cumsum(lens)])
    slices = list(dict.fromkeys(s["slice"] for s in samples)) + ["all"]
    ref = scores[ref_name]
    tok_kl, miss = {}, {}
    for name, sc in scores.items():
        if name != ref_name:
            tok_kl[name], miss[name] = per_token_kl(ref, sc, k_ref)

    rows, worst = [], {}
    for sl in slices:
        idx = [i for i, s in enumerate(samples) if sl == "all" or s["slice"] == sl]
        pos = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in idx])
        ref_ppl = float(np.exp(-ref["tgt"][pos].mean()))
        for name, sc in scores.items():
            row = {"slice": sl, "model": name, "samples": len(idx), "tokens": int(len(pos)),
                   "ppl": float(np.exp(-sc["tgt"][pos].mean()))}
            row["ppl_delta_pct"] = 100 * (row["ppl"] / ref_ppl - 1)
            if name != ref_name:
                row["kl"] = float(tok_kl[name][pos].mean())
                row["top1"] = float((sc["top_ids"][pos, 0] == ref["top_ids"][pos, 0]).mean())
                per_sample = np.array([tok_kl[name][offs[i]:offs[i + 1]].mean() for i in idx])
                row["kl_p95"] = float(np.percentile(per_sample, 95))
                if sl == "all":
                    order = np.argsort(per_sample)[::-1][:5]
                    worst[name] = [{"sample": int(idx[j]), "slice": samples[idx[j]]["slice"],
                                    "kl": float(per_sample[j])} for j in order]
            rows.append(row)
    return rows, worst, {n: float(m) for n, m in miss.items()}


def markdown(rows, ref_name):
    out = ["| slice | model | ppl | Δppl | KL | top-1 | KL p95 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["model"] == ref_name:
            rest = ["—"] * 4
        else:
            rest = [f"{r['ppl_delta_pct']:+.2f}%", f"{r['kl']:.4f}", f"{100 * r['top1']:.2f}%",
                    f"{r['kl_p95']:.4f}"]
        out.append("| " + " | ".join([r["slice"], r["model"], f"{r['ppl']:.3f}", *rest]) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ref", help="reference checkpoint (the original FP8 or the BF16 copy)")
    ap.add_argument("--ref-name", default="reference")
    ap.add_argument("--ref-tp", type=int, default=1, help="GPUs for the reference")
    ap.add_argument("--ref-backend", choices=["vllm", "hf"], default=None,
                    help="backend for the reference (default: --backend). On one GPU use hf "
                         "with the BF16 copy and --max-gpu-gb")
    ap.add_argument("--ref-cpu-offload-gb", type=float, default=0,
                    help="vllm reference: GB of weights kept in CPU RAM (vLLM cpu_offload_gb)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                    help="model to compare; repeat for each (e.g. gptq=... rtn=...)")
    ap.add_argument("--tp", type=int, default=1, help="GPUs for each compared model")
    ap.add_argument("--data", action="append", default=[], metavar="NAME=JSONL",
                    help='held-out slice, one {"messages": [...]} per line; repeat per slice')
    ap.add_argument("--n", type=int, default=64, help="samples per slice")
    ap.add_argument("--max-len", type=int, default=2048, help="tokens per sample")
    ap.add_argument("--topk-ref", type=int, default=20)
    ap.add_argument("--topk-model", type=int, default=50)
    ap.add_argument("--out", default="eval_fidelity")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--device", default="auto", help="hf backend device_map (e.g. cpu)")
    ap.add_argument("--max-gpu-gb", type=float, default=None,
                    help="hf backend: GPU memory for weights; the rest goes to CPU RAM "
                         "(70 leaves room for activations on an 80 GB A100)")
    ap.add_argument("--max-cpu-gb", type=float, default=None,
                    help="hf backend: CPU RAM for weights (default 90%% of free); "
                         "the rest spills to --offload-folder (slow)")
    ap.add_argument("--offload-folder", default="./offload_eval")
    ap.add_argument("--batch-tokens", type=int, default=16384,
                    help="hf backend: padded tokens per forward pass; lower it on OOM")
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--redo", action="store_true", help="ignore cached scores")
    for hidden in ["--_score", "--_inputs", "--_to"]:
        ap.add_argument(hidden, help=argparse.SUPPRESS)
    ap.add_argument("--_k", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_tp", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--_cpu_offload_gb", type=float, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._score:
        a.path, a.inputs, a.to, a.k, a.tp = a._score, a._inputs, a._to, a._k, a._tp
        a.cpu_offload_gb = a._cpu_offload_gb
        return score(a)

    if not (a.ref and a.model and a.data):
        ap.error("--ref, at least one --model and at least one --data are required")
    models = pairs(a.model, "--model")
    data = pairs(a.data, "--data")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("== checkpoints")
    ckpt = {a.ref_name: inspect_checkpoint(a.ref)}
    ckpt.update({n: inspect_checkpoint(p) for n, p in models.items()})
    for n, info in ckpt.items():
        print(f"  {n}: {info}")
    counts = {info["packed_tensors"] for n, info in ckpt.items() if n != a.ref_name}
    if len(counts) > 1:
        print("  WARNING: compared models have different numbers of packed tensors")

    print("== data")
    samples = build_inputs(a.ref, data, a.n, a.max_len)
    inputs = out / "inputs.json"
    blob = json.dumps(samples)
    if not inputs.exists() or inputs.read_text() != blob:
        inputs.write_text(blob)

    print("== scoring")
    scores = {a.ref_name: run_scoring(a.ref_name, a.ref, a.ref_tp, a.topk_ref, a, inputs, out,
                                      a.ref_backend or a.backend, a.ref_cpu_offload_gb)}
    for n, p in models.items():
        scores[n] = run_scoring(n, p, a.tp, a.topk_model, a, inputs, out, a.backend)

    rows, worst, miss = summarize(samples, scores, a.ref_name, a.topk_ref)
    table = markdown(rows, a.ref_name)
    res = {"checkpoints": ckpt, "metrics": rows, "worst_samples": worst,
           "kl_topk_miss_rate": miss,
           "settings": {k: v for k, v in vars(a).items() if not k.startswith("_")}}
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "results.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nKL top-k miss rate (lower = tighter estimate): "
          + ", ".join(f"{n} {100 * m:.2f}%" for n, m in miss.items()))
    print(f"results -> {out / 'results.json'}, {out / 'results.md'}")


if __name__ == "__main__":
    main()
