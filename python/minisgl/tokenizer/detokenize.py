# =====================================================================================
# FILE MAP -- tokenizer/detokenize.py
# =====================================================================================
# WHAT THIS FILE IS: the card-READER's workbench: new token ids in, a safe chunk of new text out.
#
# ELI5: Reading numbered cards back into words sounds easy: look each card up, glue the pieces.
#       The trap: ONE LETTER can be several BYTES (an "e with an accent" is two bytes; a Chinese
#       character is three; an emoji is four), and the tokenizer may cut a letter's bytes
#       BETWEEN two cards. Read the first half-letter alone and you get a broken-letter symbol
#       (the replacement character that looks like a diamond with a question mark).
#       So the reader keeps a little notebook per customer and only speaks when a letter is whole.
#
#       Worked example, with "cafe-with-accent" (accent = bytes C3 A9, split over two cards):
#         card " caf"   -> says " caf"
#         card [C3]     -> window decodes to " caf" + BROKEN   -> HOLD BACK, say nothing
#         card [A9]     -> window decodes to " caf" + "e-acute" -> say the new letter
#       Decoding each card alone and gluing would have printed " caf" + BROKEN + BROKEN.
#
# WHO CALLS IT: tokenizer/server.py (tokenize_worker).
# WHAT IT CALLS: the Hugging Face tokenizer's batch_decode.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


# WHAT: True if the unicode code point cp is a CJK (Chinese/Japanese kanji) ideograph.
# WHY: each such character is a complete "word" on its own, so it is safe to show immediately.
def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


# ELI5: "Of the text I just got, how much can I safely say out loud RIGHT NOW?"
# WHAT: the rules, in order:
#       1) text ends with a newline          -> say it all (a line is finished)
#       2) last char is CJK                  -> say it all (each such char is a whole word)
#       3) second-to-last char is CJK        -> say all but the last char (it may be half-formed)
#       4) otherwise                         -> say up to and including the last space, because the
#                                               word after it might still change as more tokens
#                                               arrive
# GOTCHA: in rule 4, if there is no space at all, rfind returns -1 so the slice is text[:0], an
#         empty string: nothing is sent until a space (or newline) shows up.
def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


# ELI5: ONE customer's notebook page.
@dataclass
class DecodeStatus:
    # WHAT: every token id received so far for this request.
    decoded_ids: List[int]
    # WHAT: all text that has been fully decoded (not necessarily sent) so far.
    decoded_str: str
    # WHAT: tokens [0, read_offset) have been turned into text and "committed".
    read_offset: int  # length of read ids
    # WHAT: tokens [surr_offset, read_offset) are the small LOOK-BEHIND window. They are decoded
    #       again each time so a letter that straddles the old/new boundary decodes correctly.
    surr_offset: int  # length of surr ids
    # WHAT: how many CHARACTERS of text have been sent to the front desk already.
    sent_offset: int  # length of sent out string


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        # ELI5: the notebook, one page per customer ticket.
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    # WHAT: msgs is one new token per request. Returns, for each, the NEW text to send (maybe "").
    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            # ELI5: first time we hear from this ticket? Start a fresh notebook page.
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            # WHAT: record the new token, except when it is the final end-of-sequence marker
            #       (so the answer doesn't end with a literal "<|im_end|>").
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            # WHAT: read_ids = the window to decode now = look-behind PLUS everything new.
            #       surr_ids = just the look-behind part, decoded separately for subtraction.
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # WHY batch_decode: decode all requests' windows in one call (faster than one by one).
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            # ELI5: text(window) minus text(look-behind) = the BRAND-NEW text. Subtracting the
            #       look-behind is what makes a half-finished letter come out right once complete.
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            # WHAT: complete case: non-empty AND not ending in the broken-letter symbol.
            #       Commit it: append to decoded_str and slide the window forward
            #       (surr_offset catches up to the old read_offset; read_offset to "now").
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            # WHAT: incomplete case: do NOT commit and do NOT move the offsets, so the same
            #       tokens are re-decoded next time with one more token available. Only the
            #       safely printable part (find_printable_text) is allowed out meanwhile.
            else:
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # WHAT: send only what the front desk has not seen: everything after sent_offset.
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            # ELI5: customer done? Tear out their notebook page.
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
