# =====================================================================================
# FILE MAP -- core.py
# =====================================================================================
# WHAT THIS FILE IS: the shared "forms" (small data classes) the whole engine passes around.
#
# ELI5: Picture the Kitchen's paperwork. The front desk, the head waiter and the chef all
#       fill in or read the SAME few forms, so nobody ever misunderstands anybody:
#         SamplingParams : the customer's "how adventurous should the chef be?" card
#         Req            : ONE order slip, plus the chef's progress notes written on it
#         Batch          : a TRAY of order slips that get cooked together in one pass
#         Context        : the kitchen's WHITEBOARD (which pantry, which tools, which tray)
#
# WHO USES IT: scheduler/* builds Req and Batch; engine/* runs a Batch; layers/* read Context.
# WHAT IT CALLS: nothing at run time. It only imports types, so it sits at the very bottom of
#                the stack and every other file can import it safely.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
# WHAT: makes every type hint a lazy string, so a hint may name a class that is not imported yet.
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

# WHY: these names are used ONLY inside type hints. Importing them for real would create
#      circular imports (attention -> core -> attention ...). TYPE_CHECKING is False at run
#      time, so this block is skipped by Python and read only by type checkers.
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


# ELI5: the customer's "how adventurous should the chef be?" card. It is attached to an order
#       at the front desk and carried along until the last word is picked.
@dataclass
class SamplingParams:
    # WHAT: 0.0 means "always take the single most likely next word" (greedy). Larger values
    #       flatten the probability table first, so unlikely words get picked more often.
    temperature: float = 0.0
    # WHAT: only consider the k most likely words. -1 means "no limit, consider all of them".
    top_k: int = -1
    # WHAT: nucleus sampling. Consider the smallest set of words whose probabilities add up to
    #       p. 1.0 means "consider everything".
    top_p: float = 1.0
    # WHAT: keep generating even after the model writes its end-of-sequence token.
    # WHY: handy for benchmarks, where every request must produce exactly max_tokens tokens.
    ignore_eos: bool = False
    # WHAT: the most NEW tokens this request may produce (the prompt does not count).
    max_tokens: int = 1024

    # ELI5: "Is the chef allowed to roll dice at all?" If not, the answer is deterministic.
    # WHAT: greedy if temperature is 0 (or top_k is 1, which leaves only one candidate) AND
    #       top_p does not trim anything. The sampler can then skip the random-number path.
    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


# ELI5: ONE order slip, plus the head waiter's progress notes. A Req lives from the moment
#       the order is accepted until the last word is served.
# WHY eq=False: two Reqs are equal only if they are the SAME object (identity), not if their
#       fields match. That lets a Req sit inside a Python set (DecodeManager.running_reqs).
@dataclass(eq=False)
class Req:
    # WHAT: every token ID known so far (prompt, then each generated token appended on the CPU).
    # SHAPE: 1-D int32 tensor of length (number of tokens known on the CPU side).
    input_ids: torch.Tensor  # cpu tensor
    # ELI5: the number of this order's PEG on the wall. The scheduler owns a table with one row
    #       per running request; this is our row number.
    # WHAT: row index into page_table and token_pool (see scheduler/table.py). Row r, column p
    #       tells you the KV slot (page_table) or the token ID (token_pool) at position p.
    table_idx: int
    # ELI5: "How many words at the front of this order do the chef's notes already cover?"
    # WHAT: count of leading tokens whose K/V is already stored in the KV cache. It comes from a
    #       prefix-cache hit (another request already did this work) and/or earlier steps.
    cached_len: int
    # WHAT: the maximum number of new tokens this request may still produce (max_tokens, after
    #       the scheduler clipped it to fit the model's maximum length).
    output_len: int
    # WHAT: unique request id, assigned by the front desk (API server). Used to route results back.
    uid: int
    sampling_params: SamplingParams
    # ELI5: a claim ticket for the shared prefix entry this order is using. While we hold the
    #       ticket, the cache may not throw that entry away.
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        # WHAT: device_len = how many tokens EXIST from the GPU's point of view. At first that
        #       is just the prompt. It grows by 1 per step (see complete_one).
        self.device_len = len(self.input_ids)
        # WHAT: the hard ceiling: prompt length + the most tokens we may generate.
        self.max_device_len = len(self.input_ids) + self.output_len
        # WHY cached_len < device_len (strictly): at least ONE token must go through the model to
        #     produce the logits for the next word, even if the whole prompt is already cached.
        #     That is why the cache lookup (scheduler/cache.py match_req) leaves out the last token.
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    # WHAT: how many more tokens this request is still allowed to produce.
    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    # ELI5: "How many words must the chef actually think about in THIS pass?"
    # WHAT: tokens that exist but have no K/V notes yet = device_len - cached_len.
    #       Prefill example: prompt of 10 tokens, 6 cached  -> extend_len = 4 (only 4 are computed).
    #       Decode: exactly 1, the newest token.
    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    # ELI5: tick the box for this step. "Everything I just cooked now has notes (cached_len
    #       catches up), and one NEW word is now owed to me (device_len goes up by one)."
    # GOTCHA: the engine calls this right after it LAUNCHES the GPU work, before the sampled
    #         token's value has reached the CPU. The bookkeeping deliberately runs one step
    #         ahead of the data. That is what lets the CPU plan step N+1 while the GPU is still
    #         busy with step N ("overlap scheduling", see scheduler.py).
    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    # WHAT: record the token that just arrived from the GPU in the CPU-side copy of the ids.
    # WHY: the prefix cache uses these ids as its lookup keys when this request's work is saved.
    # SHAPE: next_token is a 1-element 1-D tensor; input_ids grows by one element.
    # GOTCHA: torch.cat allocates a new tensor every token, which is O(length) per step.
    #         Fine for a teaching engine; a production engine would use a growable buffer.
    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    # WHAT: True while the request still has room to produce at least one more token.
    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


# ELI5: a TRAY of order slips that are cooked together in a single pass of the recipe book.
#       Remember Chapter 2 of Part I: one pass over the weights serves every slip on the tray.
@dataclass
class Batch:
    reqs: List[Req]
    # WHAT: "prefill" = the tray holds fresh orders (read the whole slip, many tokens each).
    #       "decode"  = the tray holds orders already underway (one new token each).
    #       A tray is never a mix: the scheduler picks one kind per step.
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    # SHAPE: 1-D, one entry per token being processed in this pass (all requests concatenated).
    input_ids: torch.Tensor = field(init=False)
    # WHAT: the position (0,1,2,...) of each token inside its own request. Used by RoPE.
    positions: torch.Tensor = field(init=False)
    # WHAT: for each token, the KV-cache slot where its NEW K and V must be written.
    out_loc: torch.Tensor = field(init=False)
    # ELI5: sometimes the tray is topped up with blank "dummy" slips so it has exactly the size
    #       the pre-recorded GPU recipe (CUDA graph) was made for.
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    # WHAT: extra bookkeeping the attention kernels need (page indices, lengths, ...).
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    # WHAT: number of REAL requests (padding not counted).
    @property
    def size(self) -> int:
        return len(self.reqs)

    # WHAT: number of requests including dummy padding slips (what the GPU actually processes).
    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


# ELI5: the kitchen's WHITEBOARD. Deep inside the model, a layer needs to know "which pantry are
#       we using, which attention tool, which tray is on the counter right now?" Instead of
#       passing those through every function call, they hang on this one board.
@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    # WHAT: page_table[row, position] = the KV-cache slot holding that token's K/V.
    #       "Always page_size = 1" means the entries are PER-TOKEN slot numbers, even when the
    #       cache itself is organised in bigger pages. Pages are only an allocation unit.
    # SHAPE: (max_running_req + 1, max_seq_len aligned up to a multiple of 32), int32.
    page_table: torch.Tensor = field(init=False)
    # WHAT: the attention implementation (FlashAttention / FlashInfer / TensorRT-LLM variants).
    attn_backend: BaseAttnBackend = field(init=False)
    # WHAT: only set for mixture-of-experts models.
    moe_backend: BaseMoeBackend = field(init=False)
    # WHAT: the pantry: one big pre-allocated pool of K/V slots (kvcache/mha_pool.py).
    kv_cache: BaseKVCachePool = field(init=False)
    # WHAT: the tray currently on the counter. None whenever no forward pass is running.
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    # ELI5: "Put this tray on the counter, cook, then clear the counter."
    # WHAT: a with-block that sets the current batch and ALWAYS clears it afterwards (even if
    #       the forward pass raises an error). Used as: with ctx.forward_batch(batch): ...
    # GOTCHA: nesting is forbidden (the assert) because there is only one counter.
    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


# ELI5: there is exactly ONE whiteboard per worker process, hung on the wall where every layer
#       can reach it.
# WHY a global: model layers have signatures like forward(x). They have no "batch" or "cache"
#     parameter, so they call get_global_ctx() to find those. It keeps the model code close to
#     a plain PyTorch model.
# GOTCHA: one Context per process (per GPU worker). It is not meant to be shared across threads.
_GLOBAL_CTX: Context | None = None


# WHAT: called once by Engine.__init__. The assert catches an accidental second engine.
def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
