# =====================================================================================
# FILE MAP -- kernel/moe_impl.py
# =====================================================================================
# WHAT THIS FILE IS: the Python launchers for the two Triton GPU kernels used by Mixture-of-Experts
#                    layers: the "grouped" matrix multiply (fused_moe_kernel_triton) and the
#                    final sum over each token's chosen experts (moe_sum_reduce_triton).
#                    The GPU programs themselves are in kernel/triton/fused_moe.py.
#
# ELI5 (MoE in one paragraph): a Mixture-of-Experts model has many small MLPs ("experts") per layer
#       (e.g. 128). For each word, a tiny router picks the best few (top-k, e.g. 8). Only those
#       experts process the word, and their answers are added up with the router's weights.
#       So the work is "many different small matrix multiplies, each for a different subset of
#       words". These launchers set up that irregular work for the GPU: how many blocks, what
#       strides, which settings. The heavy lifting is the Triton code.
#
# WHO CALLS IT: moe/fused.py (fused_experts_impl).
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from typing import Any, Dict

import torch


# ELI5: "For every (word, chosen expert) pair, multiply the word's vector by that expert's weight
#       matrix, optionally scaling by the router weight."
# SHAPE: A [rows, K]        input vectors (hidden states, or the activated intermediate)
#        B [E, N, K]        one weight matrix per expert (E experts; N outputs; K inputs)
#        C [M, top_k, N]    output (one row per (word, expert) pair)
#        sorted_token_ids   word-slots sorted by expert, padded so each expert's group is a multiple
#                           of BLOCK_SIZE_M (see moe_align_block_size in moe/fused.py)
#        expert_ids         one expert id per BLOCK of BLOCK_SIZE_M sorted slots
#        num_tokens_post_padded  [1] how many sorted slots (including padding) are real work
# WHAT: the launch "grid" has one program per (block of rows) x (block of output columns):
#       cdiv(len(sorted_token_ids), BLOCK_SIZE_M) x cdiv(N, BLOCK_SIZE_N) programs, flattened to 1-D.
#       even_Ks tells the kernel whether K is an exact multiple of BLOCK_SIZE_K so it can skip masking.
def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    import triton
    import triton.language as tl

    from .triton.fused_moe import fused_moe_kernel

    # WHAT: layout requirements of the kernel (rows of topk_weights contiguous; the sorted id list contiguous).
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0
    # WHAT: Triton calls this function with the chosen meta-parameters to size the grid.
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    K = B.shape[2] - padded_size
    if K % config["BLOCK_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    # WHAT: launch. Arguments, in order: the pointers; N (outputs per expert), K (inputs), EM (number of
    #       sorted slots incl. padding), number of valid (word, expert) pairs; then the strides of A,
    #       B (expert, K, N) and C; then the compile-time settings.
    #       Note B's strides are passed as (stride(0), stride(2), stride(1)) = (expert, K, N).
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


# ELI5: "Each word got top_k expert answers; add them up into one vector per word."
# SHAPE: input [tokens, top_k, hidden] (contiguous) -> output [tokens, hidden] (contiguous).
# WHAT: the grid has one program per (BLOCK_M = 1 token) x (a chunk of BLOCK_DIM = 2048 columns).
#       Each program loops over the top_k answers, adding them in float32, then stores in the
#       input dtype. 8 warps (256 threads) per program.
def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    import triton

    from .triton.fused_moe import moe_sum_reduce_kernel

    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 8

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
    )
