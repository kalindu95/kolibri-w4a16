#!/usr/bin/env python3
"""Build a tiny random Kolibri-1-architecture model, saved twice:
  <out>/bf16  - reference weights
  <out>/fp8   - same weights stored as 128x128 block FP8 + weight_scale_inv
plus a small calibration jsonl. Used to dry-run the pipeline on CPU."""
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.kolibri1 import Kolibri1Config

out = Path(sys.argv[1])
torch.manual_seed(0)
random.seed(0)

words = ("der die das und ist ein eine nicht mit auf the a and is of to in model "
         "quantization expert router weight bank risk market data Haus Bank Zinsen "
         "Kredit Aktie learn token layer").split()
corpus = [" ".join(random.choices(words, k=60)) for _ in range(400)]
tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
    special_tokens=["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>"]))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>",
                              eos_token="</s>", pad_token="</s>")
tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} "
                     "{% endfor %}")

cfg = Kolibri1Config(vocab_size=256, hidden_size=256, num_hidden_layers=5,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                     moe_intermediate_size=128, shared_expert_intermediate_size=128,
                     num_experts=16, num_experts_per_tok=4, sliding_window=33,
                     max_position_embeddings=4096, eos_token_id=2, bos_token_id=1,
                     pad_token_id=2)
model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
for n, p in model.named_parameters():  # give weights some structure / outliers
    if p.dim() >= 2:
        torch.nn.init.normal_(p, std=0.05)
        p.data[..., :3] *= 8

(out / "bf16").mkdir(parents=True, exist_ok=True)
model.save_pretrained(out / "bf16")
tok.save_pretrained(out / "bf16")

# ---- fake the released FP8 format -------------------------------------------
fp8 = out / "fp8"
fp8.mkdir(exist_ok=True)
sd = {}
for f in sorted((out / "bf16").glob("*.safetensors")):
    sd.update(load_file(f))
print("checkpoint keys (sample):", [k for k in sd if "layers.0." in k][:12])
q = {}
skip = ("embed", "lm_head", "norm", "gate.weight", "router", "bias")
for k, w in sd.items():
    if w.dim() == 2 and k.endswith(".weight") and not any(s in k for s in skip):
        R, C = w.shape
        wb = w.float().reshape(R // 128, 128, C // 128, 128)
        amax = wb.abs().amax(dim=(1, 3)).clamp(min=1e-12)
        scale = amax / 448.0
        wq = (wb / scale[:, None, :, None]).reshape(R, C).to(torch.float8_e4m3fn)
        q[k] = wq
        q[k + "_scale_inv"] = scale.to(torch.float32)
    else:
        q[k] = w
half = len(q) // 2
keys = list(q)
save_file({k: q[k] for k in keys[:half]}, fp8 / "model-00001-of-00002.safetensors")
save_file({k: q[k] for k in keys[half:]}, fp8 / "model-00002-of-00002.safetensors")
(fp8 / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {
    k: ("model-00001-of-00002.safetensors" if i < half else "model-00002-of-00002.safetensors")
    for i, k in enumerate(keys)}}))
c = json.loads((out / "bf16" / "config.json").read_text())
c["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                            "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
(fp8 / "config.json").write_text(json.dumps(c))
tok.save_pretrained(fp8)
print(f"fp8 tensors: {sum(1 for v in q.values() if v.dtype == torch.float8_e4m3fn)}")

with open(out / "calib.jsonl", "w") as f:
    for i in range(48):
        f.write(json.dumps({"messages": [
            {"role": "user", "content": corpus[i]},
            {"role": "assistant", "content": corpus[i + 100]}]}) + "\n")
print("ok")
