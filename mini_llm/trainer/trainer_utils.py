from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUT_DIR = PROJECT_ROOT / "out"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def stage_main(stage: str) -> None:
    print(f"mini_llm trainer stage '{stage}' is scaffolded.")
