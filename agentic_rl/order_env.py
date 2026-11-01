"""E1 状态化下单环境（升级方案 5.1 的隔离状态层，2026-11-01 M2 落地，Proposed→Repository）。

与 env.py 的只读选购环境互补：这里维护每个 episode 独立的可变状态——数字库存、待确认单、订单。
接口按环境合同组织：

- ``reset(task, snapshot, seed)``：从快照重建状态，同 task+seed 可完整重放；
- ``step(action)``：执行动作并转移状态；失败动作不产生半写入；
- ``state_for_verifier()``：给验证器的私有状态（库存、确认、订单、调用记录）。

确认语义对齐线上 ConfirmationService：``prepare_order`` 只生成待确认单、不扣库存；确认由环境内
模拟买家在助手作答后给出（Actor 不能自行伪造），确认成功才扣库存并创建订单；``cancel_order``
按 confirmation_id 取消已确认订单并恢复库存。数量固定 1（升级方案 5.1 第一版边界）。

库存是目录之上的叠加层：目录只提供 in_stock 布尔值，数字库存由任务快照给出，检索结果不暴露
叠加库存——策略只能通过 prepare_order 的成败感知库存，这是「正确无结果」任务的机制来源。
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from agentic_rl.catalog import Catalog, default_catalog
from agentic_rl.env import SearchBackend

MAX_STEPS = 12  # 单条轨迹的动作预算；超限记 truncated（外部截断），不是 terminated（任务终结）

ORDER_SYSTEM_PROMPT = (
    "你是 Globex 跨境电商选购助手。可用工具：product_search_tool 检索商品；prepare_order 生成下单确认单"
    "（生成后等待买家确认，确认前不扣库存）；cancel_order 取消一笔已确认的订单。\n"
    "1. 先按到手价核对条件，再准备订单；商品没货时如实说明，不要硬下单。\n"
    "2. 只有买家明确确认后才算成交；确认由买家自己完成，你不能替买家确认。\n"
    "3. 买家要求取消时，用确认单号调用 cancel_order。\n"
    "4. 回答里不要出现工具名或内部字段。"
)

ORDER_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "product_search_tool",
        "description": "检索商品目录；传 ship_to 时商品卡附单件到手价 landed_price。",
        "parameters": {"type": "object", "properties": {
            "normalized_query": {"type": "string"}, "ship_to": {"type": "string"}, "price_max_major": {"type": "number"},
            "target_currency": {"type": "string"}, "top_k": {"type": "integer"},
            "excluded_material_tags": {"type": "array", "items": {"type": "string"}},
            "product_id": {"type": "string"}, "sku_id": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "prepare_order",
        "description": "为一件商品生成待确认单；不扣库存，等待买家确认。",
        "parameters": {"type": "object", "required": ["product_id"], "properties": {
            "product_id": {"type": "string"}, "ship_to": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "cancel_order",
        "description": "按确认单号取消一笔已确认（placed）的订单并恢复库存。",
        "parameters": {"type": "object", "required": ["confirmation_id"], "properties": {"confirmation_id": {"type": "string"}}}}},
]
ORDER_TOOL_NAMES = frozenset(schema["function"]["name"] for schema in ORDER_TOOL_SCHEMAS)

NO_STOCK_TEXT = re.compile(r"没有.{0,8}货|无货|缺货|没有库存|暂无库存|无法下单|买不到|先不下单")
# 线上检索结果里嵌着墙钟时间戳（ISO 8601），会破坏同 seed 重放的一致性；
# 对训练观测而言时间戳是环境噪声，进入 messages 前归一化掉（环境合同的一部分）。
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")


def _normalize_observation(text: str) -> str:
    return _TIMESTAMP.sub("[ts]", text or "")


@dataclass
class OrderTask:
    task_id: str
    family: str            # order_place / order_out_of_stock / order_cancel
    split: str             # train / dev
    keyword: str
    ship_to: str
    budget: int
    excluded: list
    initial_query: str
    followup: str = ""     # order_cancel：买家确认成交后的反悔话术
    goal: dict = field(default_factory=dict)
    solvability: dict = field(default_factory=dict)
    env_version: str = "e1-v1"


@dataclass
class OrderSnapshot:
    """episode 的初始状态（库存叠加层）。"""
    inventory: dict

    def copy(self) -> "OrderSnapshot":
        return OrderSnapshot(dict(self.inventory))


class OrderEnvironment:
    """隔离的下单环境：每个实例独立状态，episode 与 group 内副本互不污染。"""

    def __init__(self, catalog: Catalog | None = None, backend: SearchBackend | None = None):
        self.catalog = catalog or default_catalog()
        self.backend = backend or SearchBackend(self.catalog)
        self.task: OrderTask | None = None
        self.snapshot: OrderSnapshot | None = None
        self.inventory: dict = {}
        self.orders: dict = {}
        self.confirmations: dict = {}
        self.messages: list = []
        self.calls: list = []
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self._order_seq = 0
        self._conf_seq = 0
        self._answers = 0

    # ---------- 生命周期 ----------
    def reset(self, task: OrderTask, snapshot: OrderSnapshot, seed: int = 0) -> list[dict]:
        self.task, self.snapshot = task, snapshot.copy()
        self.inventory = dict(snapshot.inventory)
        self.orders, self.confirmations, self.calls = {}, {}, []
        self.messages = [{"role": "system", "content": ORDER_SYSTEM_PROMPT},
                          {"role": "user", "content": task.initial_query}]
        self.steps, self._order_seq, self._conf_seq, self._answers = 0, 0, 0, 0
        self._followup_sent = False
        self.terminated = self.truncated = False
        return list(self.messages)

    def step(self, action: dict):
        """action：{"name": 工具名, "arguments": {...}} 或 {"type": "answer", "content": str}。
        返回 (新增消息列表, reward=None, terminated, truncated, info)。失败动作只产生错误消息，不改状态。"""
        if self.terminated or self.truncated:
            raise RuntimeError("episode 已结束，请先 reset")
        self.steps += 1
        if action.get("type") == "answer" or (not action.get("name") and action.get("content") is not None):
            message = {"role": "assistant", "content": action.get("content", "")}
            self.messages.append(message)
            self._answers += 1
            appended = [message] + self._user_turn_after_answer()
        else:
            appended = [self._run_tool(action.get("name", ""), action.get("arguments", {}) or {})]
        if self.steps >= MAX_STEPS and not self.terminated:
            self.truncated = True  # 外部预算截断，与任务终结分开记录
        return appended, None, self.terminated, self.truncated, {"steps": self.steps}

    # ---------- 工具执行（事务性：先验证后变更，失败不半写入） ----------
    def _run_tool(self, name: str, arguments: dict) -> dict:
        call_id = f"tc_{self.steps:03d}"
        if name == "product_search_tool":
            ok, text = asyncio.run(self.backend.search(arguments, session_id=self.task.task_id))
            text = _normalize_observation(text)
            self.calls.append((name, arguments, ok))
            message = {"role": "tool", "tool_call_id": call_id, "content": text}
            self.messages.append(message)
            return message
        if name == "prepare_order":
            pid = arguments.get("product_id") or ""
            offer = self.catalog.offers.get(pid)
            if offer is None:
                return self._tool_error(call_id, name, f"商品不存在：{pid}")
            if not offer.in_stock:
                return self._tool_error(call_id, name, f"商品 {pid} 当前无可用规格")
            if self.inventory.get(pid, 0) <= 0:
                return self._tool_error(call_id, name, f"商品 {pid} 库存为 0，暂时无法下单")
            self._conf_seq += 1
            conf_id = f"conf-{self._conf_seq:03d}"
            self.confirmations[conf_id] = {"product_id": pid, "status": "pending"}
            self.calls.append((name, arguments, True))
            message = {"role": "tool", "tool_call_id": call_id, "content": json.dumps(
                {"confirmation_id": conf_id, "product_id": pid, "status": "awaiting_buyer_confirmation",
                 "notice": "确认单已生成；等待买家确认，确认后才扣库存并创建订单。"}, ensure_ascii=False)}
            self.messages.append(message)
            return message
        if name == "cancel_order":
            conf_id = arguments.get("confirmation_id") or ""
            order = next((o for o in self.orders.values()
                          if o["confirmation_id"] == conf_id and o["status"] == "placed"), None)
            if order is None:
                return self._tool_error(call_id, name, f"没有对应已确认订单的确认单：{conf_id}")
            order["status"] = "cancelled"
            self.inventory[order["product_id"]] = self.inventory.get(order["product_id"], 0) + 1
            self.calls.append((name, arguments, True))
            message = {"role": "tool", "tool_call_id": call_id, "content": json.dumps(
                {"confirmation_id": conf_id, "status": "cancelled", "restocked": True}, ensure_ascii=False)}
            self.messages.append(message)
            return message
        return self._tool_error(call_id, name, f"未知工具：{name}")

    def _tool_error(self, call_id: str, name: str, reason: str) -> dict:
        self.calls.append((name, None, False))
        message = {"role": "tool", "tool_call_id": call_id,
                   "content": json.dumps({"error": reason}, ensure_ascii=False)}
        self.messages.append(message)
        return message

    # ---------- 模拟买家（确认 / 反悔；Actor 不得伪造） ----------
    def _user_turn_after_answer(self) -> list[dict]:
        appended = []
        pending = [c for c in self.confirmations.values() if c["status"] == "pending"]
        if pending:
            conf = pending[0]
            pid = conf["product_id"]
            if self.inventory.get(pid, 0) <= 0:  # 确认时库存不足：确认失败，不创建订单、不扣库存
                conf["status"] = "failed_insufficient_stock"
                message = {"role": "user", "content": "刚才那件确认时没货了，这次先不买了。"}
            else:
                conf["status"] = "confirmed"
                self.inventory[pid] -= 1
                self._order_seq += 1
                offer = self.catalog.offers[pid]
                order_id = f"order-{self._order_seq:03d}"
                self.orders[order_id] = {"product_id": pid, "status": "placed",
                                         "confirmation_id": next(k for k, v in self.confirmations.items() if v is conf),
                                         "landed_price": offer.landed.get(self.task.ship_to)}
                message = {"role": "user", "content": "确认，就买这件。"}
            appended.append(message)
            self.messages.append(message)
            return appended
        if (self.task.family == "order_cancel" and not self._followup_sent
                and any(o["status"] == "placed" for o in self.orders.values())):
            self._followup_sent = True
            message = {"role": "user", "content": self.task.followup or "想了想还是不要了，帮我取消吧。"}
            appended.append(message)
            self.messages.append(message)
            return appended
        self.terminated = True  # 流程走完，episode 结束
        return appended

    # ---------- 验证器接口 ----------
    def state_for_verifier(self) -> dict:
        return {"inventory": dict(self.inventory),
                "orders": {k: dict(v) for k, v in self.orders.items()},
                "confirmations": {k: dict(v) for k, v in self.confirmations.items()},
                "initial_inventory": dict(self.snapshot.inventory) if self.snapshot else {},
                "terminated": self.terminated, "truncated": self.truncated, "steps": self.steps,
                "calls": [tuple(c) for c in self.calls]}

    def invariant_checks(self) -> list[str]:
        """状态不变量：库存非负；当前库存 + 未取消订单占用 = 初始库存（守恒，取消已还回）。"""
        problems = []
        for pid, qty in self.inventory.items():
            if qty < 0:
                problems.append(f"inventory_negative:{pid}")
        if self.snapshot:
            for pid, initial in self.snapshot.inventory.items():
                active = sum(1 for o in self.orders.values() if o["product_id"] == pid and o["status"] == "placed")
                if self.inventory.get(pid, 0) != initial - active:
                    problems.append(f"inventory_not_conserved:{pid}:{self.inventory.get(pid, 0)}!={initial - active}")
        return problems


# ---------------- 终局核验（oracle） ----------------

def verify_order_episode(env: OrderEnvironment, answer_text: str | None = None) -> dict:
    """按任务族检查最终状态 + 不变量；answer_text 缺省取最后一条 assistant 消息。"""
    state = env.state_for_verifier()
    task = env.task
    if answer_text is None:
        texts = [m.get("content", "") for m in env.messages if m["role"] == "assistant"]
        answer_text = texts[-1] if texts else ""
    placed = [o for o in state["orders"].values() if o["status"] == "placed"]
    cancelled = [o for o in state["orders"].values() if o["status"] == "cancelled"]
    invariants = env.invariant_checks()
    checks = {"invariants": invariants, "terminated": state["terminated"], "truncated": state["truncated"]}
    if task.family == "order_place":
        ok_orders = [o for o in placed if not env.catalog.violations(
            o["product_id"], task.keyword, task.ship_to, task.budget, task.excluded)]
        checks["goal"] = "存在满足全部条件且已确认的订单"
        checks["passed"] = bool(ok_orders) and state["terminated"] and not invariants
    elif task.family == "order_out_of_stock":
        checks["goal"] = "无订单且如实说明缺货"
        checks["passed"] = (not state["orders"] and bool(NO_STOCK_TEXT.search(answer_text))
                            and state["terminated"] and not invariants)
    elif task.family == "order_cancel":
        checks["goal"] = "订单已取消且库存恢复"
        checks["passed"] = bool(cancelled) and not placed and state["terminated"] and not invariants
    else:
        checks["passed"] = False
    return {"passed": checks["passed"], "checks": checks}


# ---------------- 见证策略（规则求解器，用于任务准入） ----------------

class OrderCarefulPolicy:
    """确定性见证策略：检索 → 逐件试 prepare（库存不可见，失败换下一件）→ 买家确认 → 作答；
    全部缺货时如实说明；cancel 族在反悔后按确认单号取消。作为任务可解性的执行见证。"""

    def __init__(self, task: OrderTask):
        self.task = task
        self.searched = False

    def act(self, messages: list) -> dict:
        view = self._view(messages)
        if view["cancelled"]:
            return {"type": "answer", "content": "已经为你取消了这笔订单，库存会恢复，下次有需要再找我。"}
        if view["cancel_requested"] and view["prepared_confirmation_id"]:
            return {"name": "cancel_order", "arguments": {"confirmation_id": view["prepared_confirmation_id"]}}
        if not self.searched:
            self.searched = True
            return {"name": "product_search_tool", "arguments": {
                "normalized_query": self.task.keyword, "ship_to": self.task.ship_to,
                "price_max_major": self.task.budget, "target_currency": "CNY", "top_k": 10,
                **({"excluded_material_tags": list(self.task.excluded)} if self.task.excluded else {})}}
        if view["prepared_confirmation_id"]:
            if view["buyer_confirmed"]:
                return {"type": "answer",
                        "content": f"确认收到，已为你下单：{view['picked_title']}（{view['picked_pid']}），安排发货。"}
            return {"type": "answer",
                    "content": f"已经为你准备好订单：{view['picked_title']}（{view['picked_pid']}），请确认。"}
        pick = view["next_candidate"]
        if pick is None:
            return {"type": "answer", "content": "这个条件下暂时没有有货又能满足预算的商品，这次先不下单。"}
        return {"name": "prepare_order", "arguments": {"product_id": pick["product_id"], "ship_to": self.task.ship_to}}

    def _view(self, messages: list) -> dict:
        failed, prepared_id, picked_pid, cancelled = set(), None, None, False
        last_hits = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message.get("content", ""))
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            if "confirmation_id" in payload and "error" not in payload:
                prepared_id, picked_pid = payload["confirmation_id"], payload.get("product_id")
            if payload.get("status") == "cancelled":
                cancelled = True
            error = payload.get("error") or ""
            found = re.search(r"商品 (P\d{4,})", error)
            if found:
                failed.add(found.group(1))
            if isinstance(payload.get("hits"), list):
                last_hits = payload["hits"]
        budget = self.task.budget
        candidates = [h for h in last_hits
                      if h.get("product_id") and h.get("product_id") not in failed
                      and self.task.keyword in (h.get("title") or "")
                      and ((h.get("landed_price") or {}).get("landed_total_major") or 10 ** 9) <= budget]
        user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        picked_title = next((h.get("title") for h in last_hits if h.get("product_id") == picked_pid), "")
        return {"prepared_confirmation_id": prepared_id, "picked_pid": picked_pid, "picked_title": picked_title,
                "cancelled": cancelled, "cancel_requested": any("取消" in t for t in user_texts),
                "buyer_confirmed": any("确认，就买这件" in t for t in user_texts),
                "next_candidate": candidates[0] if candidates else None}


def run_order_episode(task: OrderTask, snapshot: OrderSnapshot, catalog: Catalog | None = None,
                      seed: int = 0, backend: SearchBackend | None = None) -> OrderEnvironment:
    """用见证策略跑完一条轨迹（供准入与重放验收复用）。"""
    env = OrderEnvironment(catalog, backend)
    policy = OrderCarefulPolicy(task)
    env.reset(task, snapshot, seed)
    while not env.terminated and not env.truncated:
        env.step(policy.act(env.messages))
    return env
