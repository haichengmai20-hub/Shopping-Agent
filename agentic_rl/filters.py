"""SFT 数据清洗：08-1 第 2.5 节的五条红线。

1. 无效循环：同一工具连续调用 ≥ 4 次，整条删除。
2. 过长轨迹：估算 token 总数 > 16384，整条剔除（2026-11-01 M1 改；08-1 原文是截断保留，但本实现的
   截断必丢终局作答，示范不完整——对齐升级方案 6.2「首版优先剔除或重采样」）。
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
            dropped["too_long"] += 1  # M1-3：超限整条剔除，不再截断入库（截断必丢终局作答）
            continue
        kept.append(item)
    return kept, dict(dropped)


def final_messages(item: dict) -> list:
    """M1-3 数据血缘：导出与统计消费轨迹消息的统一入口（clean 不再返回截断副本，此口径防回归）。"""
    return item.get("messages") or item["episode"].messages
