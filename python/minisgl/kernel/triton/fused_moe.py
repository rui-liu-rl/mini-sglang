# =====================================================================================
# FILE MAP -- kernel/triton/fused_moe.py
# =====================================================================================
# WHAT THIS FILE IS: the two GPU programs (Triton kernels) for Mixture-of-Experts layers:
#     fused_moe_kernel      the "grouped" matrix multiply: many experts, each applied only to the
#                           words that chose it
#     moe_sum_reduce_kernel add up each word's top-k expert answers
#
# ELI5: Triton lets you write a GPU program in Python-looking code. Each "program instance" (a
#       Triton program id, `pid`) is in charge of one small tile of the output and does its tile
#       completely: load a tile of A, load a tile of B, multiply-accumulate, repeat along K, store.
#
# THE KEY IDEA OF fused_moe_kernel (sorted, padded routing):
#       Before this kernel, moe/fused.py sorts all (word, chosen expert) pairs BY EXPERT and pads
#       each expert's group to a multiple of BLOCK_SIZE_M. Then every tile of BLOCK_SIZE_M rows
#       belongs to exactly ONE expert, so one weight matrix can be loaded and reused for the
#       whole tile. `expert_ids[pid_m]` says which expert that is; `sorted_token_ids` says which
#       (word, expert) pairs fill the tile (a padding id means "skip this row").
#
# GLOSSARY: M = rows (sorted slots), N = output features, K = input features, E = experts.
#       A [rows, K] x B[e] [N, K]ᵀ -> C [rows, N], with B[e] chosen per tile.
#
# Wrappers: kernel/moe_impl.py. Orchestration: moe/fused.py.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import triton
import triton.language as tl


# ELI5: "Add up the top_k answers of each word." One program handles BLOCK_M tokens (here 1) and
#       BLOCK_DIM columns (here 2048) of the output.
# SHAPE: input [token_num, topk_num, hidden_dim] -> output [token_num, hidden_dim].
@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    # WHAT: program (token block, column block). (The int64 casts of the strides above avoid
    #       overflow when token_index * stride is a big number.)
    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    # WHAT: for each token: start a float32 total (better accuracy than adding in bf16), add the
    #       top_k answers one by one (`mask` guards the last, possibly short, column chunk), then
    #       store the total converted back to the input dtype.
    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


# ELI5: one program computes one tile C[BLOCK_SIZE_M rows, BLOCK_SIZE_N columns] for ONE expert.
# THE TILE MAP (see the code): program ids are handed out in "groups" of GROUP_SIZE_M row-tiles so
#       that consecutive programs reuse the same B tiles from the fast L2 cache.
#       Example: GROUP_SIZE_M = 2, num_pid_n = 3 column tiles: pid 0..5 form group 0:
#                (m0,n0) (m1,n0) (m0,n1) (m1,n1) (m0,n2) (m1,n2) -> walk down a column of 2 tiles
#                before moving right, instead of the plain row-by-row order.
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # WHAT: the launch grid is sized for the worst case; programs beyond the real number of
    #       (padded) rows have nothing to do and leave at once.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    # WHAT: which (word, expert) pairs fill this tile? offs_token holds flat pair numbers. A pair
    #       number >= num_valid_tokens is PADDING (token_mask False): its rows load zeros and are
    #       never stored.
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # WHAT: pair number // top_k = the WORD's row in A (each word appears top_k times in the flat
    #       list: pairs 0..top_k-1 are word 0, and so on). In the second multiply the rows of A
    #       are already one per pair, and the launcher passes top_k = 1.
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    # ELI5: the whole tile uses ONE expert: look up which, and point b_ptrs at that expert's matrix.
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ELI5: slide along the K dimension BLOCK_SIZE_K at a time: load a tile of A and a tile of B,
    #       multiply them (tl.dot, on the GPU's matrix units), add into the accumulator.
    #       even_Ks (K is an exact multiple of BLOCK_SIZE_K) skips the end-of-K masking.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if even_Ks:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.

        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # WHAT: optionally scale each row by the router's weight for that (word, expert) pair. In
    #       moe/fused.py this is switched on in the SECOND multiply (down projection) by default.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    # WHAT: convert the float32 accumulator to the model dtype and store only real rows/columns
    #       (padding rows and columns beyond N are masked out).
    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
