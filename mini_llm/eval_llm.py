from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mini_llm inference/eval entrypoint.")
    parser.add_argument("--load-from", default="mini_llm/model", help="Model directory or checkpoint path.")
    parser.add_argument("--weight", default="full_sft", help="Weight prefix, e.g. pretrain/full_sft/lora.")
    parser.add_argument("--prompt", default="你好", help="Prompt for smoke testing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("mini_llm eval entrypoint is scaffolded.")
    print(f"load_from={args.load_from}")
    print(f"weight={args.weight}")
    print(f"prompt={args.prompt}")


if __name__ == "__main__":
    main()
