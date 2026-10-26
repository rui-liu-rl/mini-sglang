# =====================================================================================
# FILE MAP -- utils/hf.py
# =====================================================================================
# WHAT THIS FILE IS: the librarian that fetches things from the Hugging Face model hub (or a
#                    local folder): the tokenizer, the model's config.json, and the weight files.
#
# ELI5: A model "recipe book" lives in a folder of files:
#         config.json        how big the model is (layers, heads, widths ...)
#         tokenizer files    the card-maker's dictionary + the chat order-form template
#         *.safetensors      the actual numbers (weights)
#       Given a name like "Qwen/Qwen3-0.6B" (or a local path) this file gets each of them.
#
# Tag legend (ELI5 / WHAT / WHY / SHAPE / GOTCHA): see zero-to-sglang/eli5/README.md
# =====================================================================================
import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


# WHAT: a progress bar that never shows. Passed to the downloader so weight downloads are quiet.
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


# ELI5: get the card-maker's dictionary and chat order-form for this model.
# WHAT: AutoTokenizer picks the right tokenizer class from the model's files. A few Mistral models
#       keep the chat template in a separate chat_template.json instead of the tokenizer files;
#       if no template was found, try to download that file and attach the template. Any failure
#       there is ignored (the tokenizer simply stays without a template).
def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


# WHAT: read config.json (downloading it if needed). functools.cache means each path is loaded once
#       per process.
@functools.cache
def _load_hf_config(model_path: str) -> Any:
    return AutoConfig.from_pretrained(model_path)


# WHY the copy: callers sometimes tweak the config (for example the multimodal handling in
#     models/config.py). Returning a fresh copy of the cached object keeps those tweaks from
#     leaking into the cache or into other callers.
def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


# ELI5: "Where on disk are the weight files?" A local folder is used as is; a hub name is
#       downloaded (ONLY the *.safetensors files) into the local cache and that folder returned.
def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
