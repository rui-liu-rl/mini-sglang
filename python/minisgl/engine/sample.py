# =====================================================================================
# FILE MAP -- engine/sample.py
# =====================================================================================
# WHAT THIS FILE IS: the WORD-PICKER. The model hands back a score (logit) for every word in the
#                    vocabulary; this file turns those scores into ONE chosen token per order.
#
# ELI5: The chef finishes cooking and says: "my scores for the next word are: 'the' 9.1,
#       'a' 7.4, 'cat' 2.0, ..." (151,936 numbers for Qwen3). Now somebody must PICK a word.
#         GREEDY   : pick the highest score. Same input -> same output. Boring but exact.
#         SAMPLING : turn scores into percentages, roll a weighted die. Creative, varied.
#       Knobs the customer can turn (per order):
#         temperature : low = confident and repetitive, high = adventurous
#         top_k       : only consider the k best words
#         top_p       : only consider the smallest group of words covering p of the probability
#
# KEY DESIGN POINT: every order on a tray has ITS OWN knobs, but the GPU wants one big batch
#       operation. So `prepare` (CPU, once per tray) builds one tensor of temperatures/top-k/top-p
#       (one entry per order) and `sample` (GPU) applies them all in one go.
#
# WHO CALLS IT: engine/engine.py (Engine.sampler), scheduler/scheduler.py (_prepare_batch calls
#               prepare; Engine.forward_batch calls sample).
# WHAT IT CALLS: FlashInfer's GPU sampling kernels.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


# ELI5: the knob settings for the WHOLE tray, ready for the GPU.
@dataclass
class BatchSamplingArgs:
    # SHAPE: (batch size,) float32 on the GPU, one temperature per order.
    # WHAT: None means "every order on this tray is greedy": take the plain argmax shortcut.
    temperatures: torch.Tensor | None
    # SHAPE: (batch size,) int32. None = no order uses top-k.
    top_k: torch.Tensor | None = None
    # SHAPE: (batch size,) float32. None = no order uses top-p.
    top_p: torch.Tensor | None = None


# WHAT: python list -> pinned CPU tensor -> asynchronous copy to the GPU.
# WHY pin_memory: page-locked host memory lets the GPU fetch it by DMA without stalling the CPU.
def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


# ELI5: the real dice-rolling, in four flavours depending on which filters are active.
# WHAT: 1) divide logits by temperature and softmax -> probabilities (one row per order)
#       2) optionally keep only the top-k and/or top-p words
#       3) draw one word per row
# SHAPE: logits (batch, vocab_size) float32 -> returns (batch,) of token ids.
def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    # WHAT: softmax with a per-row temperature. enable_pdl is a Hopper-and-newer kernel-launch
    #       optimisation (programmatic dependent launch).
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    # ELI5: "Read every order's knob card and set up one sheet for the whole tray."
    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        # SHORTCUT: if nobody wants randomness, skip everything.
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # WHAT: a greedy order in a MIXED tray cannot say "temperature 0" (that would divide by
        #       zero), so it gets a tiny temperature 1e-6: the softmax becomes essentially
        #       one-hot on its best word, and it picks that word anyway.
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # WHAT: top_k <= 0 means "no limit", encoded as vocab_size (consider every word).
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        # WHAT: clamp top_p into [1e-6, 1.0].
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        # WHAT: only build the top-k / top-p tensors if some order actually uses them, so the
        #       cheaper sampling kernels are chosen when possible.
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    # ELI5: "Pick one word per order."
    # WHY greedy skips softmax: softmax never changes the ORDER of scores, so argmax of logits is
    #     the same word as argmax of probabilities, with less work.
    # WHY .float(): do the softmax in float32 even if the model runs in bf16/fp16, for accuracy.
    # SHAPE: logits (batch, vocab_size) -> (batch,) token ids.
    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
