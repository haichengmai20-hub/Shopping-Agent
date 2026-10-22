"""SFT 数据清洗：08-1 第 2.5 节的五条红线。

1. 无效循环：同一工具连续调用 ≥ 4 次，整条删除。
2. 过长轨迹：估算 token 总数 > 16384，截断到最后一次完整的工具调用之后。
3. 格式错误：工具名、参数 JSON、调用 ID、工具回复配对任一不合格，整条删除。
4. 结果泄露：作答里出现内部工具名或内部字段，整条删除。
5. 分数不达标：原文是 Rubric < 70 不入库；这里用补强版奖励，未成功或奖励 < 0.9 不入库。
"""
from __future__ import annotations

from collections import Counter

from agentic_rl.reward import LEAK_TERMS, final_text
from agentic_rl.trajectory import format_errors, message_tokens

MAX_TOKENS = 16384
MIN_REWARD = 0.9


def _consecutive_same_tool(messages: list) -> int:
    best, run, last = 0, 0, None
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            name = call["function"]["name"]
            run = run + 1 if name == last else 1
            last, best = name, max(best, run)
    return best


def truncate(messages: list, limit: int = MAX_TOKENS) -> list:
    """截断到超限之前最后一次「工具调用 + 全部回复」完整的位置。"""
    total, cut = 0, 0
    for index, message in enumerate(messages):
        total += message_tokens(message)
        if total > limit:
            break
        complete = message["role"] == "tool" and (index + 1 == len(messages) or messages[index + 1]["role"] != "tool")
        if complete or message["role"] in ("system", "user"):
            cut = index + 1
    return messages[:cut]


def clean(scored: list[dict]) -> tuple[list[dict], dict]:
    """scored：evaluate.score() 的输出。返回保留的条目和每条红线的删除计数。"""
    kept, dropped = [], Counter()
    for item in scored:
        episode = item["episode"]
        messages = episode.messages
        if _consecutive_same_tool(messages) >= 4:
            dropped["loop"] += 1
            continue
        if format_errors(messages):
            dropped["format"] += 1
            continue
        if any(term in final_text(episode) for term in LEAK_TERMS):
            dropped["leak"] += 1
            continue
        if not item["verified"]["success"] or item["verified"]["reward"] < MIN_REWARD:
            dropped["score"] += 1
            continue
        if sum(message_tokens(m) for m in messages) > MAX_TOKENS:
            dropped["truncated"] += 1  # 截断后保留，计数只为统计
            item = {**item, "messages": truncate(messages)}
        kept.append(item)
    return kept, dict(dropped)
