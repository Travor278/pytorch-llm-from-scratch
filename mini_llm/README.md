# mini_llm

`mini_llm/` is the engineering track for a MiniMind-style small LLM stack.
It is intentionally separate from `GPT/`: `GPT/` stays as the teaching
implementation, while this folder grows toward a trainable and deployable
project layout.

The layout follows the upstream MiniMind shape:

```text
mini_llm/
  dataset/              # local JSONL datasets, ignored except README
  model/                # config, Dense/MoE model, LoRA
  trainer/              # tokenizer, pretrain, SFT, LoRA, DPO, PPO/GRPO, agent
  scripts/              # chat, conversion, OpenAI-compatible serving, WebUI
  eval_llm.py           # inference/eval entrypoint
```

Target path:

1. `model/model_minimind.py`: Qwen-style decoder-only core with RMSNorm,
   RoPE, GQA, SwiGLU, optional MoE, KV cache.
2. `trainer/train_tokenizer.py`: BPE/ByteLevel tokenizer and chat special
   tokens.
3. `trainer/train_pretrain.py`: `{"text": ...}` causal LM pretraining.
4. `trainer/train_full_sft.py`: `{"conversations": [...]}` supervised tuning.
5. `trainer/train_lora.py`: native LoRA training and merge/export.
6. `trainer/train_dpo.py`: preference optimization.
7. `trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py`:
   rollout-based alignment and tool-use training.
8. `scripts/serve_openai_api.py`: OpenAI-compatible local serving.

VLM should grow separately from `LLaVA/` as an engineering stack around the
existing ViT, projector, LLM bridge, data pipeline, training, and evaluation.
