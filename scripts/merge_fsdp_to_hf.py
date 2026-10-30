"""Merge verl FSDP actor shards (model_world_size_N_rank_*.pt, Shard(dim=0) DTensors) into a HF checkpoint.

The bundled model_merger.py imports AutoModelForVision2Seq, which transformers 5 removed.

usage: python scripts/merge_fsdp_to_hf.py <ckpt>/global_step_X/actor <out_dir> [--dtype bfloat16]
       python scripts/merge_fsdp_to_hf.py <actor_dir> <out_dir> --reference Qwen/Qwen3.5-2B
  --reference: write model.safetensors directly with the reference checkpoint's key names and per-key dtypes
  (e.g. multimodal Qwen3.5, which AutoModelForCausalLM cannot rebuild); keys absent from the reference
  (a tied lm_head) are dropped, reference-only keys (Qwen3.5's unused MTP head) are not written.
"""
import argparse
import glob
import os
import re
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("actor_dir")
parser.add_argument("out_dir")
parser.add_argument("--dtype", default="bfloat16")
parser.add_argument("--reference", default=None)
args = parser.parse_args()

files = glob.glob(os.path.join(args.actor_dir, "model_world_size_*_rank_*.pt"))
world_size = int(re.search(r"world_size_(\d+)_", files[0]).group(1))
shards = [torch.load(os.path.join(args.actor_dir, f"model_world_size_{world_size}_rank_{r}.pt"),
                     map_location="cpu", weights_only=False) for r in range(world_size)]

state_dict = {}
for key, first in shards[0].items():
    if hasattr(first, "to_local"):
        assert first.placements[0].is_shard(0), f"{key}: unexpected placement {first.placements}"
        full = torch.cat([s[key].to_local() for s in shards], dim=0)
        assert full.shape == first.shape, f"{key}: merged {tuple(full.shape)} != {tuple(first.shape)}"
    else:
        full = first
    state_dict[key] = full if args.reference else full.to(getattr(torch, args.dtype))

COPY_FILES = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

if args.reference:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    ref_dir = snapshot_download(args.reference, allow_patterns=["*.safetensors"])
    ref = {}
    for f in glob.glob(os.path.join(ref_dir, "*.safetensors")):
        with safe_open(f, "pt") as sf:
            ref.update({k: sf.get_slice(k) for k in sf.keys()})
    dropped = sorted(set(state_dict) - set(ref))
    config = AutoConfig.from_pretrained(args.actor_dir)
    assert all(k == "lm_head.weight" and config.tie_word_embeddings for k in dropped), f"unexpected keys {dropped}"
    out = {}
    for k, t in state_dict.items():
        if k in ref:
            assert list(t.shape) == ref[k].get_shape(), f"{k}: {tuple(t.shape)} != {ref[k].get_shape()}"
            out[k] = t.to({"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[ref[k].get_dtype()]).contiguous()
    os.makedirs(args.out_dir, exist_ok=True)
    save_file(out, os.path.join(args.out_dir, "model.safetensors"), metadata={"format": "pt"})
    for f in COPY_FILES + ["config.json", "processor_config.json"]:
        if os.path.exists(os.path.join(args.actor_dir, f)):
            shutil.copy(os.path.join(args.actor_dir, f), args.out_dir)
    print(f"saved {len(out)} tensors (reference dtypes) to {args.out_dir}; dropped {dropped}; "
          f"not written (reference only): {sorted(set(ref) - set(out))}")
    raise SystemExit

config = AutoConfig.from_pretrained(args.actor_dir)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, dtype=getattr(torch, args.dtype))
missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
missing = [k for k in missing if not (k == "lm_head.weight" and config.tie_word_embeddings)]
assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
if config.tie_word_embeddings:
    model.tie_weights()

model.save_pretrained(args.out_dir, safe_serialization=True)
for f in COPY_FILES:
    if os.path.exists(os.path.join(args.actor_dir, f)):
        shutil.copy(os.path.join(args.actor_dir, f), args.out_dir)
print(f"saved {len(state_dict)} tensors ({args.dtype}) to {args.out_dir}")
