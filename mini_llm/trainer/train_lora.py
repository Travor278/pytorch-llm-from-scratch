from __future__ import annotations

try:
    from .trainer_utils import stage_main
except ImportError:
    from trainer_utils import stage_main


if __name__ == "__main__":
    stage_main("lora")
