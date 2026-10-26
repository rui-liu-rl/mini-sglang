# FILE MAP -- models/__init__.py
# WHAT: the models package's front door.
#   create_model(model_config) : look up the right model class from the config's architecture name
#                                (models/register.py) and build the (still empty) skeleton.
#   load_weight                : the streaming checkpoint reader (models/weight.py)
#   RotaryConfig               : the RoPE settings type (models/config.py)
# PACKAGE AT A GLANCE:
#   config.py   ModelConfig / RotaryConfig   clean description of a model's shape
#   register.py architecture name -> class
#   base.py     BaseLLMModel                 the "forward() takes no args" contract
#   utils.py    RopeAttn / GatedMLP / MoEMLP the reusable rooms
#   qwen3.py, qwen2.py, llama.py, mistral.py, qwen3_moe.py   the recipes
#   weight.py   shard + glue + load checkpoints
# Tag legend: see zero-to-sglang/eli5/README.md
from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
