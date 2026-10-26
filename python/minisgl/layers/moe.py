# =====================================================================================
# FILE MAP -- layers/moe.py
# =====================================================================================
# WHAT THIS FILE IS: the weights and wrapper for a MIXTURE-OF-EXPERTS (MoE) feed-forward layer.
#
# ELI5: A normal MLP is ONE private office every word visits. A MoE has MANY offices ("experts"),
#       each a small MLP with its own weights, plus a RECEPTIONIST (the router) that looks at each
#       word and says "you go to offices #5 and #17". Each word visits only `top_k` offices (for
#       example 8 of 128), so the model can have a huge number of parameters while each word
#       pays for only a few of them.
#
# THIS FILE holds the experts' weights (stacked in two big tensors) and delegates the actual work
# (routing, grouping words by expert, running the expert matmuls) to the MoE backend (moe/fused.py).
#
# TENSOR PARALLELISM: each expert's intermediate dimension is cut across GPUs; each GPU computes
#       a partial result for every selected expert; one all_reduce adds them.
#
# SHAPES (E = experts, I = intermediate size, H = hidden size, tp = GPUs):
#       gate_up_proj : [E, 2 * (I / tp), H]    the gate and up matrices of all experts, stacked
#       down_proj    : [E, H, I / tp]          the down matrix of all experts
#
# WHO USES IT: models/utils.py (MoEMLP). WHAT IT CALLS: the MoE backend (ctx.moe_backend),
#               DistributedCommunicator.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        # WHAT: after choosing the top-k experts, rescale their weights to add up to 1.
        renormalize: bool = True,
        activation: str = "silu",
        # WHAT: multiply the router weight into the INPUT of the expert instead of its output.
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        # SHAPE: [num_experts, 2 * I_local, hidden]. The "2" is gate and up stacked, as in GatedMLP.
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        # SHAPE: [num_experts, hidden, I_local].
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    # ELI5: "Send each word to its chosen offices, combine the answers."
    # SHAPE: hidden_states [num_tokens, hidden]; router_logits [num_tokens, num_experts]
    #        (the receptionist's scores, from the replicated gate matrix) -> [num_tokens, hidden].
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        # WHAT: each GPU computed only its slice of every expert's intermediate dimension, so the
        #       results are partial sums: add them across GPUs.
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
