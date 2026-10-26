# =====================================================================================
# FILE MAP -- moe/fused.py
# =====================================================================================
# WHAT THIS FILE IS: the working Mixture-of-Experts backend ("fused"): given every word's hidden
#                    vector and the router's scores, run the chosen experts and combine them.
#
# ELI5: An MoE layer is a team of specialist cooks (experts). A tiny router looks at each word and
#       says "send this to specialists 3 and 17". Only those cook for that word, and their dishes
#       are blended using the router's confidence. Doing this naively means thousands of tiny
#       matrix multiplies. This file reorganises the work so the GPU does a few big, regular ones:
#
#   1. fused_topk               router scores -> for each word, its top-k experts and their weights
#   2. moe_align_block_size     list every (word, expert) pair, SORT the list by expert, and pad
#                               each expert's group to a multiple of the tile height, so each
#                               tile of rows belongs to exactly one expert
#   3. multiply #1 (gate+up)    for each tile: rows @ that expert's w1  -> [pairs, 2 x intermediate]
#   4. silu_and_mul             activation: silu(gate) * up              -> [pairs, intermediate]
#   5. multiply #2 (down)       each tile: rows @ that expert's w2, times the router weight
#                                                                       -> [words, top_k, hidden]
#   6. moe_sum_reduce           add each word's top_k results            -> [words, hidden]
#
#       WORKED EXAMPLE of step 2 is in the docstring of moe_align_block_size below. The GPU
#       programs for steps 3, 5, 6 are in kernel/triton/fused_moe.py.
#
# SHAPES: hidden_states [M, hidden]; w1 [E, 2*I, hidden] (gate and up stacked); w2 [E, hidden, I];
#         gating_output [M, E]; topk_ids / topk_weights [M, top_k].
#
# WHO CALLS IT: layers/moe.py (the MoE layer) through the backend registry in moe/__init__.py.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import functools
from typing import Dict, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.utils import div_ceil


# ELI5: "For each word, which top-k experts does the router pick, and how much does it trust each?"
# WHAT: sgl_kernel.topk_softmax does softmax over the E router scores and keeps the k largest
#       (writes weights and expert ids into the buffers given). If `renormalize`, the kept weights
#       are rescaled to add up to 1. Python then divides by the sum again (a near no-op, since the
#       kernel flag already renormalised; the 1e-8 avoids dividing by zero).
#       `num_token_non_padded`, if given, marks rows beyond that count as padding (expert id -1).
# SHAPE: -> topk_weights [M, top_k] float32, topk_ids [M, top_k] int32.
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


# ELI5: sort the (word, expert) pairs by expert and pad every expert's group to a multiple of
#       block_size, so each tile of block_size rows uses ONE expert's weights.
# GOTCHA: the example in the docstring labels its experts 1..4 while saying num_experts = 4; with
#         zero-based expert ids that would be 0..3. I replayed the example in plain Python: the
#         sorted ids come out exactly as printed ([3,6,9,12, 0,4,10,12, 1,7,11,12, 2,5,8,12]),
#         so read it as "experts numbered from 1".
# SHAPE: out sorted_token_ids [<= numel + (num_experts + 1) * (block_size - 1)] (a worst-case size;
#        only num_tokens_post_pad entries are meaningful), expert_ids [number of blocks],
#        num_tokens_post_pad [1].
#        A pair number p means: word p // top_k, its (p % top_k)-th chosen expert. Pair number
#        == numel marks padding.
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    # WHAT: allocate output buffers for the worst case (every expert group needs the maximum
    #       padding of block_size - 1), then let the compiled kernel fill them in. The kernel is
    #       told num_experts + 1 (one extra slot, probably for the -1 "padding" expert id) and
    #       given a scratch array `cumsum_buffer` for running totals. The final True is a flag
    #       whose meaning I did not check.
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


# ELI5: pick the tile sizes for the grouped multiply.
# WHAT: two presets. Many words (M > E): big tiles, 64 x 64 x 32, with row tiles processed in
#       groups of 8 for L2 reuse. Few words (M <= E, e.g. a small decode batch): each expert
#       only gets a handful of words, so use short, wide tiles (16 rows x 32 columns x 64 deep)
#       to waste little padding. N, K and topk are accepted but not used for the choice.
def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


# WHAT: a thin wrapper that unpacks (E, _, N) from w2's shape and asks get_default_config. The
#       name promises an "optimal" search; today it is just the two presets above.
def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


# ELI5: the whole expert computation for one batch (steps 2 to 6 of the file map).
# GOTCHA: `out_hidden_states = hidden_states`: the final sum is written INTO the input tensor,
#         overwriting it. That is safe here because every use of hidden_states has finished by
#         then, but the caller's tensor is modified.
# MEMORY: one buffer `cache` is shared by intermediate_cache1 (first multiply's output) and
#         intermediate_cache3 (second multiply's output): they are never needed at the same time,
#         because the activation has consumed cache1 before the second multiply writes cache3.
def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    # SHAPE: cache1 [M, top_k, N=2I]; cache2 [M * top_k, I] (after silu_and_mul halves the width);
    #        cache3 [M, top_k, hidden]. All in the model dtype.
    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    # NOTE: begin_token_idx / end_token_idx cover the whole batch here. The slicing looks like a
    #       leftover from a version that processed the batch in chunks; it is harmless.
    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    # STEP 2: sort and pad the (word, expert) pairs using the tile height as the block size.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    # STEP 3: first multiply, gate and up projections together: rows of hidden_states (each word
    #         used top_k times) times each chosen expert's w1. top_k = topk, so row = pair // topk.
    #         The router weight is multiplied here only if apply_router_weight_on_input is set.
    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    # STEP 4: activation: split each row of width N into (gate, up) halves and compute
    #         act(gate) * up into cache2 (width N // 2).
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    # STEP 5: second multiply, the down projection. Each row of cache2 is already one (word, expert)
    #         pair, so top_k = 1 here. By default (apply_router_weight_on_input False) the router
    #         weight is applied now: mul_routed_weight = not apply_router_weight_on_input.
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
    )

    # STEP 6: add each word's top_k expert answers into one vector (written into the input tensor).
    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


# ELI5: the registered backend object: forward() = pick the experts, then run them.
# WHAT: BaseMoeBackend (moe/base.py) defines the interface; this is the only implementation.
class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
