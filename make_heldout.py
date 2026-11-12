#!/usr/bin/env python3
"""
Held-out evaluation data that never overlaps the default calibration set of step 2.

English: step 2 shuffles open-perfectblend with --seed (42) and takes at most the first
2 x samples rows. This script shuffles the same way and starts after --skip rows
(default 2048, safe for --samples up to 1024), so no calibration row can appear.

German: Aya has only ~241 German rows and step 2's default calibration can use all of
them, so the German slice comes from OpenAssistant oasst2 instead (not used in
calibration): each German first prompt with its best-ranked assistant reply.

Writes en.jsonl and de.jsonl, one {"messages": [...]} per line.

Add your own slices (e.g. financial Q&A) as further jsonl files in the same format,
and keep them out of --calib-jsonl.

Usage:
    python make_heldout.py --out $WORK/eval/data --n 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def write(rows, path):
    with open(path, "w") as f:
        for msgs in rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128, help="samples per language")
    ap.add_argument("--skip", type=int, default=2048,
                    help="rows to skip after shuffling; must be >= 2 x --samples of step 2")
    ap.add_argument("--seed", type=int, default=42, help="must match step 2's --seed")
    ap.add_argument("--min-chars", type=int, default=400,
                    help="drop short conversations, they say little about the model")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    role = {"human": "user", "gpt": "assistant", "system": "system"}
    en = load_dataset("mlabonne/open-perfectblend", split="train").shuffle(seed=args.seed)
    rows = []
    for ex in en.select(range(args.skip, len(en))):
        msgs = [{"role": role.get(t["from"], "user"), "content": t["value"]}
                for t in ex["conversations"]]
        if sum(len(m["content"]) for m in msgs) >= args.min_chars:
            rows.append(msgs)
        if len(rows) >= args.n:
            break
    write(rows, out / "en.jsonl")

    oa = load_dataset("OpenAssistant/oasst2", split="train")
    oa = oa.filter(lambda ex: ex["lang"] == "de" and not ex["deleted"])
    prompts = {ex["message_id"]: ex["text"] for ex in oa
               if ex["parent_id"] is None and ex["role"] == "prompter"}
    best = {}  # prompt id -> (rank, reply text)
    for ex in oa:
        if ex["role"] == "assistant" and ex["parent_id"] in prompts:
            rank = ex["rank"] if ex["rank"] is not None else 99
            if ex["parent_id"] not in best or rank < best[ex["parent_id"]][0]:
                best[ex["parent_id"]] = (rank, ex["text"])
    pids = sorted(best)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pids)
    rows = []
    for pid in pids:
        if len(prompts[pid]) + len(best[pid][1]) >= args.min_chars:
            rows.append([{"role": "user", "content": prompts[pid]},
                         {"role": "assistant", "content": best[pid][1]}])
        if len(rows) >= args.n:
            break
    write(rows, out / "de.jsonl")


if __name__ == "__main__":
    main()
