from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

DATASET2_ROOT = REPO_ROOT / "dataset2"
HYMENOPTERA_TRAIN_ROOT = REPO_ROOT / "dataset" / "hymenoptera_data" / "train"

RUNS_ROOT = SCRIPT_DIR / "runs"
CHECKPOINTS_ROOT = SCRIPT_DIR / "checkpoints"


def log_dir(name: str) -> str:
    path = RUNS_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def checkpoint_path(name: str) -> str:
    path = CHECKPOINTS_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
