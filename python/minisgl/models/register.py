# =====================================================================================
# FILE MAP -- models/register.py
# =====================================================================================
# WHAT THIS FILE IS: the directory that maps a Hugging Face "architecture name" to the
#                    mini-sglang class that implements it.
#
# ELI5: Every model's config.json says what it is, e.g. "architectures": ["Qwen3ForCausalLM"].
#       This file is the phone book: "Qwen3ForCausalLM -> look in models/qwen3.py, class
#       Qwen3ForCausalLM". To support a new model family you add a model file and one line here.
#
# WHY importlib (lazy import): only the file for the model you actually serve is imported, so you
#     don't pay to import (and don't need the dependencies of) every other model.
#
# NOTE: "Mistral3ForConditionalGeneration" (a multimodal wrapper) is served by the text-only
#       MistralForCausalLM class; models/config.py unwraps its nested text_config, and
#       models/weight.py skips the vision-tower weights.
#
# Tag legend: see zero-to-sglang/eli5/README.md
# =====================================================================================
import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


# WHAT: look up the (module, class) pair, import the module, and build the model skeleton from the
#       config. Raises a clear error for unknown architectures.
def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
