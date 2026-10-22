"""规则化模拟用户：隐藏条件只有被问到才回答，第一次作答后按场景追问。

《项目定位与其他AI须知》建议第一版用结构化、规则式的模拟用户，便于核验真值，
也避免让另一个 LLM 随意扮演买家带来的噪声。回答的措辞跟随任务的说法风格（A 或 B）。
"""
from __future__ import annotations

import random

from agentic_rl import phrasing
from agentic_rl.tasks import Task

SLOT_WORDS = {
    "ship_to": ("配送", "寄", "收货", "国家", "目的地", "地址", "ship"),
    "budget": ("预算", "价格", "价位", "多少钱", "budget", "price"),
    "excluded": ("材质", "材料", "material"),
}


def detect_slot(question: dict) -> str | None:
    text = f"{question.get('id', '')} {question.get('label', '')}".lower()
    for slot, words in SLOT_WORDS.items():
        if any(word in text for word in words):
            return slot
    return None


class SimulatedUser:
    def __init__(self, task: Task):
        self.task = task
        self.rng = random.Random(task.task_id)
        self.answers_given = 0
        # 改条件之前的取值：第一次作答前被追问时，买家按当时的条件回答
        self._before = dict(task.final_constraints())
        if task.change.get("slot") == "budget":
            self._before["budget"] = task.change["from"]
        elif task.change.get("slot") == "ship_to":
            self._before["ship_to"] = task.change["from"]
        elif task.change.get("slot") == "excluded":
            self._before["excluded"] = [m for m in task.excluded if m != task.change["add"]]

    def _current(self) -> dict:
        return self.task.final_constraints() if self.answers_given else self._before

    def answer_form(self, questions: list) -> tuple[str, list]:
        """返回买家提交表单后的消息，以及每道题识别出的槽位。"""
        current, style, parts, slots = self._current(), self.task.style, [], []
        for question in questions if isinstance(questions, list) else []:
            question = question if isinstance(question, dict) else {}
            slot = detect_slot(question)
            slots.append(slot)
            label = question.get("label") or question.get("id") or "补充信息"
            if slot in self.task.hidden:
                answer = self.task.hidden[slot]
            elif slot == "ship_to":
                answer = phrasing.pick(self.rng, phrasing.DESTINATIONS[current["ship_to"]], style)
            elif slot == "budget":
                answer = phrasing.budget_phrase(self.rng, style, current["budget"])
            elif slot == "excluded":
                excluded = current["excluded"]
                answer = phrasing.pick(self.rng, phrasing.MATERIALS[excluded[0]], style) if excluded else "材质无所谓"
            else:
                answer = "没有特别要求"
            parts.append(f"{label}：{answer}")
        return "已提交表单。" + "；".join(parts or ["没有可回答的问题"]), slots

    def after_answer(self) -> str | None:
        """助手给出一次答复后，买家的下一句；None 表示买家满意，对话结束。"""
        self.answers_given += 1
        if self.answers_given == 1 and self.task.followup:
            return self.task.followup
        return None
