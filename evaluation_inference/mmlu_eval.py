# mmlu_eval.py
# MMLU：多学科多选题评估（Massive Multitask Language Understanding）
#
# MMLU 的核心不是“聊天效果”，而是“知识 + 推理 + 选项决策”。
# 形式通常是：
#   - 一个题干
#   - 四个选项 A/B/C/D
#   - 模型输出最终选项
#
# 常见做法：
#   1. zero-shot：直接答题
#   2. few-shot：每个 subject 前面拼几道示例题
#   3. 指标：每个 subject 的 accuracy + macro average
#
# 这份脚本实现的是“教学版 MMLU evaluator”：
# - 自带小型 toy 数据
# - 展示 few-shot prompt 如何拼
# - 展示如何从模型输出中解析 A/B/C/D

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from collections import defaultdict
import re


print("=== MMLU 评估 ===")
print(
    """
MMLU 适合回答的问题是：
- 这个模型知识面够不够广？
- 跨学科能力是否均衡？
- 是不是只会聊天，但客观题不稳？

MMLU 不擅长衡量的是：
- 多轮对话体验
- 长文写作质量
- 用户偏好
"""
)
print()


class MultipleChoiceModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class MMLUExample:
    subject: str
    question: str
    choices: list[str]
    answer: str


def format_question(example: MMLUExample) -> str:
    choice_lines = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(("A", "B", "C", "D"), example.choices)
    )
    return (
        f"Subject: {example.subject}\n"
        f"Question: {example.question}\n"
        f"{choice_lines}\n"
        "Answer:"
    )


def build_fewshot_prompt(
    examples: list[MMLUExample],
    target: MMLUExample,
) -> str:
    shots = []
    for ex in examples:
        shots.append(f"{format_question(ex)} {ex.answer}")
    shots.append(format_question(target))
    return "\n\n".join(shots)


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def evaluate_mmlu(
    model: MultipleChoiceModel,
    examples: list[MMLUExample],
    fewshot_bank: dict[str, list[MMLUExample]],
) -> tuple[float, dict[str, float]]:
    by_subject: dict[str, list[float]] = defaultdict(list)

    for ex in examples:
        prompt = build_fewshot_prompt(fewshot_bank.get(ex.subject, []), ex)
        raw_output = model.generate(prompt)
        pred = extract_choice(raw_output)
        score = float(pred == ex.answer)
        by_subject[ex.subject].append(score)

        print(f"[{ex.subject}]")
        print(f"Q    : {ex.question}")
        print(f"pred : {pred}  raw={raw_output!r}")
        print(f"gold : {ex.answer}")
        print()

    subject_acc = {
        subject: sum(scores) / len(scores)
        for subject, scores in by_subject.items()
    }
    macro = sum(subject_acc.values()) / max(len(subject_acc), 1)
    return macro, subject_acc


class ToyMMLUModel:
    """一个基于关键词的玩具模型，仅用于演示评测流程。"""

    def generate(self, prompt: str) -> str:
        lower = prompt.lower()
        if "2 + 2" in lower:
            return "The answer is B."
        if "capital of france" in lower:
            return "C"
        if "h2o" in lower:
            return "A"
        if "red planet" in lower:
            return "B"
        return "D"


def demo_dataset() -> tuple[list[MMLUExample], dict[str, list[MMLUExample]]]:
    fewshot_bank = {
        "elementary_mathematics": [
            MMLUExample(
                subject="elementary_mathematics",
                question="1 + 1 = ?",
                choices=["1", "2", "3", "4"],
                answer="B",
            )
        ],
        "high_school_geography": [
            MMLUExample(
                subject="high_school_geography",
                question="The capital of Italy is:",
                choices=["Paris", "Rome", "Berlin", "Madrid"],
                answer="B",
            )
        ],
    }

    eval_examples = [
        MMLUExample(
            subject="elementary_mathematics",
            question="2 + 2 = ?",
            choices=["3", "4", "5", "6"],
            answer="B",
        ),
        MMLUExample(
            subject="high_school_geography",
            question="The capital of France is:",
            choices=["Berlin", "Madrid", "Paris", "Lisbon"],
            answer="C",
        ),
        MMLUExample(
            subject="college_biology",
            question="H2O is the chemical formula for:",
            choices=["Water", "Hydrogen peroxide", "Oxygen", "Salt"],
            answer="A",
        ),
        MMLUExample(
            subject="astronomy",
            question="Which planet is often called the Red Planet?",
            choices=["Venus", "Mars", "Jupiter", "Mercury"],
            answer="B",
        ),
    ]
    return eval_examples, fewshot_bank


print("=== few-shot MMLU prompt 示例 ===")
example_eval, example_bank = demo_dataset()
example_prompt = build_fewshot_prompt(example_bank["elementary_mathematics"], example_eval[0])
print(example_prompt)
print()


if __name__ == "__main__":
    eval_examples, fewshot_bank = demo_dataset()
    model = ToyMMLUModel()
    macro_acc, subject_acc = evaluate_mmlu(model, eval_examples, fewshot_bank)

    print("=== MMLU 结果汇总 ===")
    for subject, acc in subject_acc.items():
        print(f"{subject:>26}: {acc:.3f}")
    print(f"{'macro average':>26}: {macro_acc:.3f}")
    print()
    print(
        """
实践里常见的坑：
1. 模型经常输出整句解释，必须有稳健的 A/B/C/D 解析器
2. few-shot 示例要按 subject 匹配，别把历史题和化学题混在一起
3. 看总平均还不够，最好保留 per-subject accuracy
4. 中文模型做英文 MMLU 时，经常吃 prompt 格式亏，模板要统一
"""
    )
