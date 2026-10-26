# =====================================================================================
# FILE MAP -- models/weight.py
# =====================================================================================
# WHAT THIS FILE IS: the RECIPE-BOOK READER. It streams the weight files from disk and, for each
#                    tensor, (1) cuts out THIS GPU's share, (2) glues related matrices together,
#                    and (3) hands (name, tensor) pairs to the model.
#
# ELI5: The published recipe book (the Hugging Face checkpoint) is written for ONE big chef:
#       separate matrices q_proj, k_proj, v_proj, gate_proj, up_proj ... But our engine
#       (a) wants q, k, v glued into ONE matrix (qkv_proj) and gate, up glued into one
#       (gate_up_proj), because one big multiply is faster, and (b) with several GPUs, each GPU
#       wants only ITS slice of every matrix. So while reading, this loader cuts and glues
#       on the fly, so no GPU ever holds more than it needs.
#
# THE CUTTING RULES (n = number of GPUs, r = this GPU's rank):
#       q/k/v/gate/up   : split the OUTPUT rows  (dim 0)   -> column-parallel layers
#       o_proj/down     : split the INPUT columns (dim 1)  -> row-parallel layers
#       embed/lm_head   : split the VOCABULARY rows (dim 0) in near-equal ceil-sized pieces
#       everything else (norm weights, ...): not split, every GPU keeps a copy
#       K/V special case: if there are FEWER KV heads than GPUs, GPU r takes the single KV head
#       number  r * num_kv_heads // n  (so several GPUs hold copies of the same head).
#
# MEMORY: streams tensor by tensor, straight to the GPU (safetensors device=...). Peak extra memory
#       is about one full tensor plus the small glue buffers.
#
# MoE: the checkpoint stores each expert separately (experts.0..E-1); they are stacked into ONE
#       tensor [E, ...] to match layers/moe.py.
#
# WHO CALLS IT: engine/engine.py (_load_weight_state_dict).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

# WHAT: name fragments that identify which cutting rule applies.
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
# ELI5: "q_proj, k_proj and v_proj are all parts of qkv_proj, in the order q, k, v".
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
# WHAT: matches MoE expert keys like "model.layers.3.mlp.experts.17.gate_proj.weight".
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


# ELI5: "From this whole tensor, cut out the part GPU r should keep."
# SHAPE examples (Qwen3-0.6B, n = 2): q_proj [2048, 1024] -> rank r keeps rows [r*1024 : (r+1)*1024];
#        o_proj [1024, 2048] -> rank r keeps columns [r*1024 : (r+1)*1024];
#        embed_tokens [151936, 1024] -> rank r keeps rows [r*75968 : (r+1)*75968].
# WHY .clone(): a slice is a view of the big tensor; cloning makes it an independent, contiguous
#     tensor so the big one can be freed.
def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        # WHAT: fewer KV heads than GPUs: whole heads are replicated, not split.
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        # WHAT: vocabulary split in ceil-sized chunks; the last GPU may get fewer rows.
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


# WHAT: if this tensor belongs to a glue group, return (glued name, my slot, all slots).
#       Example: "...q_proj.weight" -> ("...qkv_proj.weight", "q", ("q","k","v")).
def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


# WHAT: for an expert tensor "…experts.17.gate_proj.weight" return ("…experts.gate_proj", 17):
#       the packed name (without the expert number and ".weight") and which expert it is.
def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


# ELI5: "Read the recipe book page by page; give me each page already cut for my GPU and glued."
# WHAT: a GENERATOR: yields (name, tensor) pairs one at a time (nothing is held longer than needed).
def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    # WHAT: local folder, or download only the *.safetensors files.
    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    # WHAT: some repos also ship a "consolidated" copy of the same weights in another format;
    #       skip it unless it is the only file.
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    # ELI5: the glue-waiting table. q arrives first and waits for k and v before they are glued.
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        # WHAT: open the file and read tensors DIRECTLY onto the GPU.
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                # 1) cut out my share
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                # 2) glue (q,k,v) or (gate,up) once all parts have arrived
                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                # 3) MoE: collect every expert's tensor, then stack them into [E, ...]
                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    # WHAT: a sanity check: nothing may be left waiting for a missing part.
    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
