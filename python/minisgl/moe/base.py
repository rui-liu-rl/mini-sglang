# FILE MAP -- moe/base.py
# WHAT: the contract for a mixture-of-experts backend: one forward() that, given every token's
#       hidden vector, the stacked expert weights and the router's scores, returns the combined
#       expert outputs.
# ELI5: the "hall of many offices" needs a manager who (1) asks the receptionist's scores which
#       offices each word should visit, (2) groups words by office so each office processes its
#       visitors in one batch, (3) collects and merges the answers. That manager is a MoE backend.
# ARGUMENTS: hidden_states [tokens, hidden]; w1 = stacked gate_up weights [E, 2I, H];
#            w2 = stacked down weights [E, H, I]; gating_output = router scores [tokens, E];
#            topk = experts per token; renormalize = rescale chosen weights to sum to 1;
#            activation = "silu"/"gelu"; apply_router_weight_on_input = where to multiply the weight.
# Tag legend: see zero-to-sglang/eli5/README.md
from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
