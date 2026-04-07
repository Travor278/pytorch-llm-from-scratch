# mt_bench_eval.py
# MT-Bench：多轮对话评测（LLM-as-a-Judge）
#
# MT-Bench 的重点不是“选项题做对没”，而是：
# - 多轮对话是否连贯
# - 回答是否有帮助
# - 是否遵循约束
# - 推理、写作、编码、角色扮演等开放任务做得如何
#
# 标准 MT-Bench 常见流程：
#   1. 准备多轮问题集（通常 2 turn）
#   2. 被测模型生成每一轮回答
#   3. 由 judge model 按 rubric 打分（例如 1~10）
#   4. 汇总 turn score / question score / category score
#
# 这份脚本实现的是“教学版 MT-Bench”：
# - 问题集、回答、judge 都可本地运行
# - 用一个规则 judge 演示“LLM-as-a-Judge 的数据流”
# - 后续可替换成真实大模型 judge

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from collections import defaultdict


print("=== MT-Bench 评估 ===")
print(
    """
MT-Bench 适合衡量：
- 多轮对话上下文保持
- 解释是否清晰
- 指令遵循是否稳定
- 开放生成质量

它不适合单独衡量：
- 学科知识覆盖面
- 真实用户偏好
"""
)
print()


class ChatModel(Protocol):
    def reply(self, messages: list[dict[str, str]]) -> str:
        ...


class Judge(Protocol):
    def score(
        self,
        question: "MTBenchQuestion",
        answers: list[str],
    ) -> list[float]:
        ...


@dataclass
class MTBenchQuestion:
    question_id: str
    category: str
    turns: list[str]


def run_mt_bench(
    model: ChatModel,
    questions: list[MTBenchQuestion],
    judge: Judge,
) -> tuple[float, dict[str, float]]:
    by_category: dict[str, list[float]] = defaultdict(list)

    for question in questions:
        messages: list[dict[str, str]] = []
        answers: list[str] = []

        print(f"[{question.question_id}] category={question.category}")
        for turn_id, user_turn in enumerate(question.turns, start=1):
            messages.append({"role": "user", "content": user_turn})
            answer = model.reply(messages)
            answers.append(answer)
            messages.append({"role": "assistant", "content": answer})
            print(f"user[{turn_id}]      : {user_turn}")
            print(f"assistant[{turn_id}] : {answer}")

        turn_scores = judge.score(question, answers)
        question_score = sum(turn_scores) / len(turn_scores)
        by_category[question.category].append(question_score)
        print(f"turn scores    : {[round(s, 2) for s in turn_scores]}")
        print(f"question score : {question_score:.2f}")
        print()

    category_scores = {
        category: sum(scores) / len(scores)
        for category, scores in by_category.items()
    }
    overall = sum(category_scores.values()) / max(len(category_scores), 1)
    return overall, category_scores


class StructuredTutorModel:
    def reply(self, messages: list[dict[str, str]]) -> str:
        latest = messages[-1]["content"]
        if len(messages) >= 3:
            return f"结合上文，我分三步回答：先回顾约束，再解释 {latest}，最后给出可执行建议。"
        return f"我先给出结论，再补充原因：关于“{latest}”，建议按步骤拆解。"


class ShortAnswerModel:
    def reply(self, messages: list[dict[str, str]]) -> str:
        latest = messages[-1]["content"]
        return f"可以，答案是：{latest}"


class RubricJudge:
    """
    一个教学版 judge：
    - 长度更充分
    - 出现“步骤/原因/建议/上文”等结构化词汇
    - 第二轮能体现多轮上下文
    """

    def score(
        self,
        question: MTBenchQuestion,
        answers: list[str],
    ) -> list[float]:
        scores: list[float] = []
        for turn_id, answer in enumerate(answers, start=1):
            score = 4.0
            score += min(len(answer), 120) / 60
            if any(keyword in answer for keyword in ("步骤", "原因", "建议", "结论")):
                score += 1.0
            if turn_id > 1 and "上文" in answer:
                score += 1.0
            scores.append(min(score, 10.0))
        return scores


def demo_questions() -> list[MTBenchQuestion]:
    return [
        MTBenchQuestion(
            question_id="writing-1",
            category="writing",
            turns=[
                "请写一个学习计划模板。",
                "把它改成适合上班族的版本，并解释为什么这样改。",
            ],
        ),
        MTBenchQuestion(
            question_id="reasoning-1",
            category="reasoning",
            turns=[
                "为什么反向传播适合标量损失训练？",
                "请再用更口语化的方式给初学者解释一次。",
            ],
        ),
    ]


if __name__ == "__main__":
    questions = demo_questions()
    judge = RubricJudge()

    print("=== 模型 A：结构化回答 ===")
    overall_a, category_a = run_mt_bench(StructuredTutorModel(), questions, judge)
    print("A overall :", round(overall_a, 2))
    print("A by cat  :", {k: round(v, 2) for k, v in category_a.items()})
    print()

    print("=== 模型 B：简短回答 ===")
    overall_b, category_b = run_mt_bench(ShortAnswerModel(), questions, judge)
    print("B overall :", round(overall_b, 2))
    print("B by cat  :", {k: round(v, 2) for k, v in category_b.items()})
    print()

    print(
        """
实践里常见的坑：
1. judge prompt 不稳定，会引入较大方差
2. 单个总分不够，要保留 turn score 和 category score
3. 多轮题必须真的把历史对话喂给模型，而不是每轮独立问
4. MT-Bench 很适合比较“会不会聊天”，但不等于真实用户偏好
"""
    )
