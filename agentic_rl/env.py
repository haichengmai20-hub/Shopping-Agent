"""可重置的选购训练环境：工具调用线上真实的商品检索，买家由规则化模拟用户扮演。

每条轨迹从任务种子重建：初始状态只有系统提示和买家的第一句话。任务只检索、不下单，
商品目录只读，所以轨迹之间没有共享的可变状态，不会出现线上「新建会话不等于重置库存」的问题。
动作空间与线上对齐：product_search_tool 就是线上的工具函数；show_shopping_form 沿用线上的参数
和「展示后结束本轮、等待买家提交」的约定，提交内容由模拟用户给出。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from agentscope.message import ToolResultState

from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import InMemoryProductRepository

from agentic_rl.catalog import CURRENCY, Catalog, default_catalog
from agentic_rl.tasks import Task
from agentic_rl.user_sim import SimulatedUser

MAX_ASSISTANT_TURNS = 15  # 08-2：单条轨迹最大轮数 15，超限强制终止
MAX_USER_TURNS = 4

SYSTEM_PROMPT = (
    "你是 Globex 跨境电商选购助手。可用工具：product_search_tool 检索商品；show_shopping_form 向买家补充询问条件，"
    "展示后结束本轮，等待买家提交。\n"
    "1. 只询问会改变候选集合的未知条件，已经明确的不要重复问。\n"
    "2. 买家说的预算按到手价理解，包含运费和关税；商品卡里的 landed_price 是单件到手价，price_max_major 只过滤单件商品价。\n"
    "3. 只推荐本轮检索结果里出现过、并且核对过满足全部条件的商品，推荐时附 product_id；"
    "没有满足条件的商品时如实说明，不列编号，也不擅自放宽条件。\n"
    "4. 买家改了条件时，按新条件重新检索，不要沿用旧候选。\n"
    "5. 买家要求核对某件商品的当前价格或库存时，先用 product_id 精确检索再回答。\n"
    "6. 回答里不要出现工具名或内部字段。"
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "product_search_tool",
        "description": "检索商品目录；传 ship_to 时商品卡附单件到手价 landed_price。price_max_major 只比较单件商品价。",
        "parameters": {"type": "object", "properties": {
            "normalized_query": {"type": "string"}, "ship_to": {"type": "string", "enum": ["CN", "US", "JP", "SG", "EU"]},
            "price_max_major": {"type": "number"}, "target_currency": {"type": "string"}, "top_k": {"type": "integer"},
            "excluded_material_tags": {"type": "array", "items": {"type": "string"}},
            "product_id": {"type": "string"}, "sku_id": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "show_shopping_form",
        "description": "向买家展示澄清表单；展示后结束本轮，等待买家提交，不得按默认值继续检索。",
        "parameters": {"type": "object", "required": ["title", "questions"], "properties": {
            "title": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "type": {"type": "string"}, "label": {"type": "string"}}}}}}}},
]
TOOL_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    arguments: dict
    ok: bool
    result: dict | None
    assistant_turn: int
    after_user_turn: int


@dataclass
class Episode:
    task: Task
    policy: str
    messages: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    forms: list = field(default_factory=list)  # [{"turn", "after_user_turn", "slots"}]
    finals: list = field(default_factory=list)  # 每次作答的消息下标
    termination: str = ""
    assistant_turns: int = 0
    user_turns: int = 1


class SearchBackend:
    """进程内复用一份检索用例；每次调用都是线上的工具函数。"""

    def __init__(self, catalog: Catalog | None = None):
        catalog = catalog or default_catalog()
        self.tool = build_product_search_tool(CatalogSearchUseCase(InMemoryProductRepository(catalog.products)), TradeEventBus())

    async def search(self, arguments: dict, session_id: str) -> tuple[bool, str]:
        token = ShoppingContext.set(ShoppingContextSnapshot(shopping_session_id=session_id, buyer_id="rl-buyer",
                                                            locale="zh-CN", currency=CURRENCY))
        try:
            chunk = await self.tool(**arguments)
        except TypeError as err:  # 模型传了工具不认识的参数
            return False, json.dumps({"error": f"参数无效：{err}"}, ensure_ascii=False)
        finally:
            ShoppingContext.reset(token)
        return chunk.state == ToolResultState.SUCCESS, chunk.content[0].text


async def run_episode(policy, task: Task, backend: SearchBackend, policy_name: str | None = None) -> Episode:
    """跑完一条轨迹。policy(messages, task_view) 返回 {"content": str, "tool_calls": [{"name", "arguments"}]}。"""
    user = SimulatedUser(task)
    episode = Episode(task=task, policy=policy_name or getattr(policy, "__name__", "policy"),
                      messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.initial_query}])
    pending_form, counter = None, 0
    while True:
        if episode.assistant_turns >= MAX_ASSISTANT_TURNS:
            episode.termination = "max_turns"
            break
        action = policy(episode.messages, {"tools": TOOL_SCHEMAS})
        episode.assistant_turns += 1
        message = {"role": "assistant", "content": action.get("content", "")}
        calls = action.get("tool_calls") or []
        if calls:
            message["tool_calls"] = []
            for call in calls:
                counter += 1
                message["tool_calls"].append({"id": f"tc_{counter:03d}", "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        episode.messages.append(message)
        if calls:
            for call, wire in zip(calls, message["tool_calls"]):
                ok, text, parsed = await _execute(call, backend, task, episode, user)
                if call["name"] == "show_shopping_form" and ok:
                    pending_form = call.get("arguments", {}).get("questions", [])
                episode.calls.append(ToolCallRecord(wire["id"], call["name"], call.get("arguments", {}), ok, parsed,
                                                    episode.assistant_turns, episode.user_turns))
                episode.messages.append({"role": "tool", "tool_call_id": wire["id"], "content": text})
            continue
        if pending_form is not None:
            reply, slots = user.answer_form(pending_form)
            episode.forms.append({"turn": episode.assistant_turns, "after_user_turn": episode.user_turns, "slots": slots})
            pending_form = None
            episode.messages.append({"role": "user", "content": reply})
            episode.user_turns += 1
            continue
        episode.finals.append(len(episode.messages) - 1)
        follow = user.after_answer()
        if follow and episode.user_turns < MAX_USER_TURNS:
            episode.messages.append({"role": "user", "content": follow})
            episode.user_turns += 1
            continue
        episode.termination = "answered"
        break
    return episode


async def _execute(call: dict, backend: SearchBackend, task: Task, episode: Episode, user: SimulatedUser):
    name, arguments = call.get("name"), call.get("arguments", {})
    if name not in TOOL_NAMES or not isinstance(arguments, dict):
        text = json.dumps({"error": f"未知工具或参数格式错误：{name}"}, ensure_ascii=False)
        return False, text, None
    if name == "product_search_tool":
        ok, text = await backend.search(arguments, session_id=task.task_id)
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return ok and isinstance(parsed, dict) and "hits" in parsed, text, parsed
    questions = arguments.get("questions")
    if not arguments.get("title") or not isinstance(questions, list) or not questions:
        return False, json.dumps({"error": "表单需要 title 和非空的 questions"}, ensure_ascii=False), None
    text = json.dumps({"form_id": f"form-{len(episode.forms) + 1}", "status": "awaiting_input",
                       "notice": "澄清表单已显示；请结束本轮等待买家提交。不得按默认值继续检索或当作已确认约束。"},
                      ensure_ascii=False)
    return True, text, None


def run_many(policy, tasks: list[Task], backend: SearchBackend | None = None, policy_name: str | None = None) -> list[Episode]:
    backend = backend or SearchBackend()

    async def _all():
        return [await run_episode(policy, task, backend, policy_name) for task in tasks]

    return asyncio.run(_all())
