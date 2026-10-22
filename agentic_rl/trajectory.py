"""轨迹格式（08-1 的多轮 JSONL）、格式校验和 loss mask。

一条轨迹存成一行 JSON：messages 按 system、user、assistant（可带 tool_calls）、tool 的顺序排列。
只有 assistant 生成的内容计入 loss，system、user 和工具返回只作上下文。token 数在这里按字数估算，
真实训练要用 Qwen3 的 tokenizer 和对话模板重算，并核对模板渲染后的 mask 边界。
"""
from __future__ import annotations

import json
import re

from agentic_rl.env import TOOL_NAMES, Episode

CALL_ID = re.compile(r"^tc_\d{3,}$")
_CJK = re.compile(r"[一-鿿]")


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中文按每字 1 个，其余按每 4 个字符 1 个。只用于清洗阈值和统计。"""
    text = text or ""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4


def message_tokens(message: dict) -> int:
    calls = json.dumps(message.get("tool_calls", []), ensure_ascii=False) if message.get("tool_calls") else ""
    return estimate_tokens(message.get("content") or "") + estimate_tokens(calls)


def format_errors(messages: list) -> list[str]:
    """08-1 第 2.4 节的格式要求：工具名合法、参数可解析、调用 ID 唯一、每个调用都有对应的工具回复。"""
    errors, open_calls, seen = [], {}, set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls", []) or []:
                call_id, function = call.get("id", ""), call.get("function", {})
                if not CALL_ID.match(call_id) or call_id in seen:
                    errors.append(f"{index}: 调用 ID 不合法或重复 {call_id!r}")
                seen.add(call_id)
                if function.get("name") not in TOOL_NAMES:
                    errors.append(f"{index}: 未知工具 {function.get('name')!r}")
                try:
                    json.loads(function.get("arguments", ""))
                except (TypeError, ValueError):
                    errors.append(f"{index}: 参数不是合法 JSON")
                open_calls[call_id] = index
        elif role == "tool":
            if message.get("tool_call_id") not in open_calls:
                errors.append(f"{index}: 工具回复没有对应的调用")
            open_calls.pop(message.get("tool_call_id"), None)
    errors += [f"{index}: 调用 {call_id} 没有工具回复" for call_id, index in open_calls.items()]
    return errors


def loss_mask(messages: list) -> list[dict]:
    """逐条消息给出是否计入 loss 和估算的 token 数。"""
    return [{"index": i, "role": m["role"], "loss": int(m["role"] == "assistant"), "tokens": message_tokens(m)}
            for i, m in enumerate(messages)]


def to_record(episode: Episode, rewards: dict | None = None, policy_version: str = "rule-v1") -> dict:
    task = episode.task
    mask = loss_mask(episode.messages)
    return {
        "task_id": task.task_id, "split": task.split, "scenario": task.scenario, "policy": episode.policy,
        "policy_version": policy_version, "termination": episode.termination,
        "messages": episode.messages, "loss_mask": [item["loss"] for item in mask],
        "tokens_total": sum(item["tokens"] for item in mask),
        "tokens_policy": sum(item["tokens"] for item in mask if item["loss"]),
        "rewards": rewards or {},
    }


def write_jsonl(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
