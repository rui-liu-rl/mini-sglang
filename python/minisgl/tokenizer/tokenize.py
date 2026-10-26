# =====================================================================================
# FILE MAP -- tokenizer/tokenize.py
# =====================================================================================
# WHAT THIS FILE IS: the card-maker's workbench: text in, numbered cards out.
#
# ELI5: The model only understands numbers. The tokenizer is a big dictionary of word-PIECES
#       ("Hel", "lo", " world"), each with a number. This file cuts a customer's text into those
#       pieces and writes down their numbers.
#       For a CHAT, it first fills in the model's ORDER FORM (the chat template) so the model
#       can tell who said what.
#
# WHO CALLS IT: tokenizer/server.py (tokenize_worker).
# WHAT IT CALLS: the Hugging Face tokenizer object.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    # WHAT: one tensor of token ids per message, in the same order as the input list.
    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        # WHAT: processed one message at a time (the TODO notes that Hugging Face can tokenize a
        #       whole list at once, which would be a bit faster).
        for msg in msgs:
            # ELI5: a LIST means "a chat" (many role-stamped messages); a plain string means
            #       "raw text, no order form".
            if isinstance(msg.text, list):
                # WHAT: apply_chat_template glues the messages together with the model's special
                #       marker tokens (for Qwen, text like <|im_start|>user ... <|im_end|>).
                # WHY tokenize=False: we want the formatted TEXT first; the encode call below
                #     turns it into ids in one place for both code paths.
                # WHY add_generation_prompt=True: appends the "assistant is about to speak"
                #     marker, so the model's very next token is the start of its answer.
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            # SHAPE: encode(..., return_tensors="pt") gives shape (1, number_of_tokens).
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # SHAPE: view(-1) flattens to 1-D (number_of_tokens,); int32 is the dtype the rest of
            #        the engine expects for token ids (and it halves the bytes vs int64).
            results.append(input_ids.view(-1).to(torch.int32))
        return results
