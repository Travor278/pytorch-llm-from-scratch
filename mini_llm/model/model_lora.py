from __future__ import annotations

import torch.nn as nn


def apply_lora(model: nn.Module, *, rank: int = 8, alpha: int = 16, target_modules: tuple[str, ...] = ("q_proj", "v_proj")) -> nn.Module:
    """Attach native LoRA adapters to selected modules.

    The implementation will live here instead of depending on external PEFT
    wrappers, matching the MiniMind-style native training path.
    """

    raise NotImplementedError("Native LoRA injection is not implemented yet.")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    raise NotImplementedError("LoRA merge/export is not implemented yet.")
