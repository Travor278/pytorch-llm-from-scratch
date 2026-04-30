from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniMindConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int | None = None
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    router_aux_loss_coef: float = 5e-4
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2


class MiniMindForCausalLM(nn.Module):
    """Engineering placeholder for the MiniMind-style Causal LM.

    This file will host the real decoder-only implementation: RMSNorm, RoPE,
    GQA, SwiGLU, optional MoE, KV cache, and generation support.
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

    def forward(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError("MiniMindForCausalLM.forward is not implemented yet.")

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs):
        raise NotImplementedError("MiniMindForCausalLM.generate is not implemented yet.")
