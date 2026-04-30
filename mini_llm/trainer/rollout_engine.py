from __future__ import annotations


class RolloutEngine:
    """Rollout abstraction for PPO/GRPO/agent training."""

    def generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError("Rollout generation is not implemented yet.")
