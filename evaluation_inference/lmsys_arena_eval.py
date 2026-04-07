# lmsys_arena_eval.py
# LMSYS / Chatbot Arena：匿名 A/B 对战、用户偏好、Elo 排名
#
# 与 MMLU、MT-Bench 最大的不同在于：
# - MMLU 看“做对没”
# - MT-Bench 看“judge 怎么打分”
# - Arena 看“真实用户更喜欢谁”
#
# 典型 Arena 数据流：
#   1. 给同一 prompt，同时生成 answer A / answer B
#   2. 随机打乱左右顺序，避免位置偏差
#   3. 用户投票：A 更好 / B 更好 / 平局
#   4. 汇总成 battle log
#   5. 由 battle log 计算 win rate、Elo 或 Bradley-Terry score
#
# 这份脚本实现“教学版 Arena”：
# - 演示 battle log 的组织方式
# - 演示随机匿名化顺序
# - 演示 vote -> Elo 的完整链路

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import random


print("=== LMSYS / Arena 评估 ===")
print(
    """
Arena 关心的不是“标准答案”，而是：
- 同一个问题下，用户主观上更喜欢哪个回答
- 回答是否更自然、更有帮助、更少废话

它常用来回答：
- 新模型上线后，真实偏好有没有提升？
- 两个模型体验差距到底大不大？
"""
)
print()


class ChatModel(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class ArenaBattle:
    prompt: str
    answer_a: str
    answer_b: str
    winner: str


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float,
    rating_b: float,
    winner: str,
    k: float = 32.0,
) -> tuple[float, float]:
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a

    if winner == "A":
        score_a, score_b = 1.0, 0.0
    elif winner == "B":
        score_a, score_b = 0.0, 1.0
    else:
        score_a, score_b = 0.5, 0.5

    rating_a += k * (score_a - exp_a)
    rating_b += k * (score_b - exp_b)
    return rating_a, rating_b


def run_arena(
    model_a: ChatModel,
    model_b: ChatModel,
    prompts: list[str],
    judge_vote_fn,
    seed: int = 42,
) -> tuple[list[ArenaBattle], float, float]:
    rng = random.Random(seed)
    battles: list[ArenaBattle] = []
    rating_a, rating_b = 1000.0, 1000.0

    for prompt in prompts:
        ans_a = model_a.generate(prompt)
        ans_b = model_b.generate(prompt)

        if rng.random() < 0.5:
            shown_left, shown_right = ans_a, ans_b
            mapping = {"left": "A", "right": "B"}
        else:
            shown_left, shown_right = ans_b, ans_a
            mapping = {"left": "B", "right": "A"}

        vote = judge_vote_fn(prompt, shown_left, shown_right)
        if vote == "tie":
            winner = "tie"
        else:
            winner = mapping[vote]

        rating_a, rating_b = update_elo(rating_a, rating_b, winner)
        battles.append(ArenaBattle(prompt, ans_a, ans_b, winner))

        print(f"prompt  : {prompt}")
        print(f"vote    : {vote}  -> winner={winner}")
        print(f"Elo A/B : {rating_a:.1f} / {rating_b:.1f}")
        print()

    return battles, rating_a, rating_b


class DetailedModel:
    def generate(self, prompt: str) -> str:
        return (
            f"关于“{prompt}”，我的回答分三部分："
            "先给结论，再解释原因，最后给出可执行建议。"
        )


class BriefModel:
    def generate(self, prompt: str) -> str:
        return f"{prompt} 可以，直接做。"


def heuristic_user_vote(prompt: str, left: str, right: str) -> str:
    """
    一个模拟“用户投票”的启发式函数：
    - 更偏好解释更充分的回答
    - 若两边长度接近，则判 tie
    """
    diff = len(left) - len(right)
    if abs(diff) < 12:
        return "tie"
    return "left" if diff > 0 else "right"


def summarize_battles(battles: list[ArenaBattle]) -> None:
    wins_a = sum(b.winner == "A" for b in battles)
    wins_b = sum(b.winner == "B" for b in battles)
    ties = sum(b.winner == "tie" for b in battles)
    total = max(len(battles), 1)
    print("=== 对战统计 ===")
    print(f"A wins : {wins_a} ({wins_a / total:.3f})")
    print(f"B wins : {wins_b} ({wins_b / total:.3f})")
    print(f"ties   : {ties} ({ties / total:.3f})")
    print()


if __name__ == "__main__":
    prompts = [
        "如何开始学习 PyTorch？",
        "为什么训练时要区分 train 和 eval？",
        "怎么给初学者解释 Transformer？",
        "写一个一周学习计划。",
    ]

    battles, rating_a, rating_b = run_arena(
        model_a=DetailedModel(),
        model_b=BriefModel(),
        prompts=prompts,
        judge_vote_fn=heuristic_user_vote,
    )
    summarize_battles(battles)
    print(f"Final Elo A/B = {rating_a:.1f} / {rating_b:.1f}")
    print()
    print(
        """
实践里常见的坑：
1. 左右展示顺序必须随机化，否则会出现位置偏差
2. Arena 很容易受文风、长度、格式偏置影响，不等于“真实能力”唯一真相
3. Elo 只是把 battle log 压缩成一个数，原始对战记录同样重要
4. 如果要更严谨，常会再配 bootstrap 置信区间或 Bradley-Terry 拟合
"""
    )
