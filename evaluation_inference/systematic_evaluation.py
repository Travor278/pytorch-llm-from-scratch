# systematic_evaluation.py
# 大语言模型系统评估：从离线指标到成对对战（pairwise battle）
#
# 这份脚本不是某一个具体 benchmark，而是把常见评估方法的“骨架”讲清楚：
#   1. 离线客观题：如 MMLU、HellaSwag、ARC，指标通常是 accuracy / EM
#   2. 开放式问答：如 MT-Bench，通常用 LLM-as-a-Judge 或人工打分
#   3. 在线偏好对战：如 LMSYS Arena，通常收集 A/B 偏好，再换算 Elo / Bradley-Terry
#
# 设计目标：
# - 用最小可运行代码把“评估体系”讲完整
# - 保持风格接近本仓库里的教学脚本：注释充分、demo 可直接运行
# - 后续可作为 MMLU / MT-Bench / LMSYS 三个脚本的公共概念参考

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import random


print("=== LLM 系统评估总览 ===")
print(
    """
评估维度大致分三层：
1. 能不能做对：客观题准确率（accuracy / exact match）
2. 做得好不好：开放生成质量（helpfulness / correctness / safety）
3. 用户更喜欢谁：A/B 对战偏好（win rate / Elo）

三者的关系：
- accuracy 高，不代表对话体验一定更好
- judge score 高，不代表真实用户一定更喜欢
- Arena 胜率高，也不一定说明知识测验更强

所以真正系统的评估，一般是“多指标并行”而不是只看一个榜单。
"""
)
print()


class TextModel(Protocol):
    def generate(self, prompt: str) -> str:
        """给定 prompt，返回文本输出。"""


@dataclass
class MultipleChoiceExample:
    prompt: str
    choices: list[str]
    answer: str
    category: str


@dataclass
class OpenEndedExample:
    prompt: str
    reference: str | None
    category: str


@dataclass
class PairwiseBattle:
    prompt: str
    answer_a: str
    answer_b: str
    winner: str  # 'A' / 'B' / 'tie'
    category: str


def exact_match(prediction: str, target: str) -> float:
    return float(prediction.strip() == target.strip())


def multiple_choice_accuracy(predictions: list[str], answers: list[str]) -> float:
    correct = sum(pred == ans for pred, ans in zip(predictions, answers))
    return correct / max(len(answers), 1)


def pairwise_win_rate(battles: list[PairwiseBattle]) -> tuple[float, float, float]:
    wins_a = sum(b.winner == "A" for b in battles)
    wins_b = sum(b.winner == "B" for b in battles)
    ties = sum(b.winner == "tie" for b in battles)
    total = max(len(battles), 1)
    return wins_a / total, wins_b / total, ties / total


def elo_from_battles(
    battles: list[PairwiseBattle],
    rating_a: float = 1000.0,
    rating_b: float = 1000.0,
    k: float = 32.0,
) -> tuple[float, float]:
    """最简 Elo 更新：按 battle 顺序逐场更新。"""
    ra = rating_a
    rb = rating_b

    for battle in battles:
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        expected_b = 1.0 - expected_a

        if battle.winner == "A":
            score_a, score_b = 1.0, 0.0
        elif battle.winner == "B":
            score_a, score_b = 0.0, 1.0
        else:
            score_a, score_b = 0.5, 0.5

        ra = ra + k * (score_a - expected_a)
        rb = rb + k * (score_b - expected_b)

    return ra, rb


def bootstrap_win_rate_ci(
    battles: list[PairwiseBattle],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """对 A 的胜率做一个简单 bootstrap 置信区间。"""
    if not battles:
        return 0.0, 0.0

    rng = random.Random(seed)
    sampled_rates: list[float] = []
    n = len(battles)

    for _ in range(n_bootstrap):
        sampled = [battles[rng.randrange(n)] for _ in range(n)]
        win_a, _, _ = pairwise_win_rate(sampled)
        sampled_rates.append(win_a)

    sampled_rates.sort()
    low = sampled_rates[int(0.025 * len(sampled_rates))]
    high = sampled_rates[int(0.975 * len(sampled_rates))]
    return low, high


class ToyMultipleChoiceModel:
    """一个玩具模型：用关键词做最粗糙的“答题”。"""

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "2 + 2" in prompt_lower:
            return "B"
        if "capital of france" in prompt_lower:
            return "C"
        if "water freezes" in prompt_lower:
            return "A"
        return "D"


class ToyChatModel:
    """一个玩具对话模型：输出较完整、结构化的回答。"""

    def generate(self, prompt: str) -> str:
        return (
            "先给结论，再给原因：\n"
            f"对于问题“{prompt}”，我建议分步骤分析，"
            "先明确目标，再说明假设，最后给出可执行方案。"
        )


class ShortChatModel:
    """另一个玩具模型：回答很短，经常缺少解释。"""

    def generate(self, prompt: str) -> str:
        return f"结论：{prompt} 可以做。"


class LengthBiasJudge:
    """
    一个故意很“粗糙”的 judge：
    - 更偏好内容完整、长度适中的答案
    - 这也说明 judge 本身会带偏置，真实系统必须警惕“judge bias”
    """

    def judge_pair(self, prompt: str, answer_a: str, answer_b: str) -> str:
        score_a = self._score(answer_a)
        score_b = self._score(answer_b)
        if abs(score_a - score_b) < 0.2:
            return "tie"
        return "A" if score_a > score_b else "B"

    @staticmethod
    def _score(text: str) -> float:
        length_score = min(len(text), 120) / 120
        structure_bonus = 0.2 if ("步骤" in text or "结论" in text) else 0.0
        return length_score + structure_bonus


def demo_offline_eval() -> None:
    print("=== 1. 离线客观题评估（accuracy）===")
    examples = [
        MultipleChoiceExample(
            prompt="What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
            choices=["3", "4", "5", "6"],
            answer="B",
            category="math",
        ),
        MultipleChoiceExample(
            prompt="What is the capital of France?\nA. Berlin\nB. Rome\nC. Paris\nD. Madrid",
            choices=["Berlin", "Rome", "Paris", "Madrid"],
            answer="C",
            category="geography",
        ),
        MultipleChoiceExample(
            prompt="Water freezes at:\nA. 0°C\nB. 50°C\nC. 100°C\nD. -50°C",
            choices=["0°C", "50°C", "100°C", "-50°C"],
            answer="A",
            category="science",
        ),
    ]
    model = ToyMultipleChoiceModel()
    preds = [model.generate(ex.prompt) for ex in examples]
    acc = multiple_choice_accuracy(preds, [ex.answer for ex in examples])
    for ex, pred in zip(examples, preds):
        print(f"[{ex.category}] pred={pred}  gold={ex.answer}")
    print(f"accuracy = {acc:.3f}")
    print()


def demo_open_ended_eval() -> None:
    print("=== 2. 开放式生成评估（judge score / pairwise）===")
    prompts = [
        OpenEndedExample(prompt="如何制定一个学习计划？", reference=None, category="helpfulness"),
        OpenEndedExample(prompt="怎么向初学者解释反向传播？", reference=None, category="reasoning"),
    ]
    strong_model = ToyChatModel()
    weak_model = ShortChatModel()
    judge = LengthBiasJudge()

    battles: list[PairwiseBattle] = []
    for item in prompts:
        ans_a = strong_model.generate(item.prompt)
        ans_b = weak_model.generate(item.prompt)
        winner = judge.judge_pair(item.prompt, ans_a, ans_b)
        battles.append(
            PairwiseBattle(
                prompt=item.prompt,
                answer_a=ans_a,
                answer_b=ans_b,
                winner=winner,
                category=item.category,
            )
        )
        print(f"prompt : {item.prompt}")
        print(f"winner : {winner}")
        print(f"A      : {ans_a}")
        print(f"B      : {ans_b}")
        print()

    win_a, win_b, tie = pairwise_win_rate(battles)
    elo_a, elo_b = elo_from_battles(battles)
    ci_low, ci_high = bootstrap_win_rate_ci(battles, n_bootstrap=500)
    print(f"win rate A/B/tie = {win_a:.3f} / {win_b:.3f} / {tie:.3f}")
    print(f"Elo(A)={elo_a:.1f}, Elo(B)={elo_b:.1f}")
    print(f"95% bootstrap CI of A win rate = [{ci_low:.3f}, {ci_high:.3f}]")
    print()


print("=== 评估方法如何配合使用 ===")
print(
    """
推荐的组合方式：
1. 先跑 offline benchmark：排查“模型会不会”
2. 再跑 open-ended judge：看“回答好不好”
3. 最后做 pairwise battle：看“用户更喜欢谁”

这就是后面 MMLU / MT-Bench / LMSYS 三个脚本分别承担的角色。
"""
)
print()


if __name__ == "__main__":
    demo_offline_eval()
    demo_open_ended_eval()
