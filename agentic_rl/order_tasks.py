"""E1 订单任务生成与准入（升级方案 5.2 的最小落地，2026-11-01 M2）。

从有限、可枚举的目录生成可证明的任务：程序负责计算与核验，见证策略负责执行见证。
准入门槛按顺序：Schema 合法 → 初始状态不变量 → 目标可实现或有「正确无结果」证据（见证求解）→
文本与结构化目标一致 → 按任务族/关键词/目的地/预算去重并切分。拒绝的任务连同原因保留。
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from agentic_rl import phrasing
from agentic_rl.catalog import Catalog, default_catalog
from agentic_rl.env import SearchBackend
from agentic_rl.order_env import OrderSnapshot, OrderTask, run_order_episode, verify_order_episode
from agentic_rl.tasks import TRAIN_KEYWORDS, _query

FAMILIES = ("order_place", "order_out_of_stock", "order_cancel")
PER_FAMILY = 24  # M2 验收：每族至少 20 个可读样本


def _round_up(value: float, step: int = 10) -> int:
    return int(-(-value // step) * step)


def _make_task(rng: random.Random, family: str, index: int, catalog: Catalog):
    keyword = rng.choice(TRAIN_KEYWORDS)
    dests = [d for d in phrasing.DESTINATIONS if len(catalog.eligible(keyword, d)) >= 3]
    if not dests:
        return None, None
    dest = rng.choice(dests)
    excluded = []
    present = sorted({m for pid in catalog.eligible(keyword, dest) for m in catalog.offers[pid].materials}
                     & set(phrasing.MATERIALS))
    if present and rng.random() < 0.3:
        candidate = rng.choice(present)
        if len(catalog.eligible(keyword, dest, [candidate])) >= 2:
            excluded = [candidate]
    base = catalog.eligible(keyword, dest, excluded)
    if not base:
        return None, None
    landed = sorted(catalog.offers[p].landed.get(dest, 10 ** 9) for p in base)
    budget = _round_up(landed[min(len(landed) - 1, len(landed) // 2)]) + rng.choice((0, 10, 20))
    feasible = catalog.feasible(keyword, dest, budget, excluded)
    if not feasible:
        return None, None
    inventory = {pid: rng.randint(1, 4) for pid in base}
    if family == "order_out_of_stock":
        for pid in feasible:
            inventory[pid] = 0  # 合格品全部无货：可证明的「正确无结果」
    query = _query(rng, "A", keyword, rng.choice(phrasing.DESTINATIONS[dest]["A"]),
                   phrasing.budget_phrase(rng, "A", budget),
                   phrasing.pick(rng, phrasing.MATERIALS[excluded[0]], "A") if excluded else "")
    goals = {"order_place": {"type": "order_placed"},
             "order_out_of_stock": {"type": "no_order", "reason": "out_of_stock"},
             "order_cancel": {"type": "order_cancelled"}}
    task = OrderTask(task_id=f"order-{family}-{index:03d}", family=family,
                     split="train" if index % 10 < 7 else "dev",
                     keyword=keyword, ship_to=dest, budget=budget, excluded=excluded, initial_query=query,
                     followup=rng.choice(("想了想还是不要了，帮我取消吧。", "先不买了，取消吧。"))
                     if family == "order_cancel" else "",
                     goal=goals[family], solvability={"feasible": feasible, "inventory": inventory})
    return task, OrderSnapshot(inventory)


def admit(task: OrderTask, snapshot: OrderSnapshot, catalog: Catalog, backend: SearchBackend | None = None) -> dict:
    """准入：schema → 初始不变量 → 见证求解（可解或正确无结果）→ 文本一致性。"""
    reasons = []
    if task.budget <= 0 or task.ship_to not in phrasing.DESTINATIONS or task.keyword not in task.initial_query:
        reasons.append("schema")
    if any(not isinstance(v, int) or v < 0 for v in snapshot.inventory.values()):
        reasons.append("initial_invariant")
    env = run_order_episode(task, snapshot, catalog, seed=zlib.crc32(task.task_id.encode()) % 1000, backend=backend)
    verdict = verify_order_episode(env)
    if env.truncated:
        reasons.append("witness_truncated")
    if not verdict["passed"]:
        reasons.append("witness_failed")
    return {"admitted": not reasons, "reasons": reasons,
            "witness": {"passed": verdict["passed"], "steps": env.steps, "checks": verdict["checks"]}}


def generate_order_pool(catalog: Catalog | None = None, per_family: int = PER_FAMILY,
                        seed: int = 101) -> tuple[list[dict], list[dict]]:
    """生成并通过准入的任务池；返回 (任务记录列表, 拒绝记录列表)。"""
    catalog = catalog or default_catalog()
    backend = SearchBackend(catalog)
    rng = random.Random(seed)
    records, rejected, seen = [], [], set()
    for family in FAMILIES:
        made, index, guard = 0, 0, per_family * 80
        while made < per_family and guard > 0:
            guard -= 1
            index += 1
            task, snapshot = _make_task(rng, family, index, catalog)
            if task is None:
                continue
            key = (family, task.keyword, task.ship_to, task.budget, tuple(task.excluded))
            if key in seen:
                continue
            seen.add(key)
            decision = admit(task, snapshot, catalog, backend)
            if decision["admitted"]:
                records.append({"task": task, "snapshot": snapshot})
                made += 1
            else:
                rejected.append({"task_id": task.task_id, "family": family, "reasons": decision["reasons"]})
    return records, rejected


def save_pool(records: list[dict], rejected: list[dict], directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)

    def row(item: dict) -> dict:
        task, snapshot = item["task"], item["snapshot"]
        return {"task_id": task.task_id, "family": task.family, "split": task.split, "keyword": task.keyword,
                "ship_to": task.ship_to, "budget": task.budget, "excluded": task.excluded,
                "initial_query": task.initial_query, "followup": task.followup, "goal": task.goal,
                "solvability": task.solvability, "inventory": snapshot.inventory, "env_version": task.env_version}

    with open(directory / "tasks_order.jsonl", "w", encoding="utf-8") as fh:
        for item in records:
            fh.write(json.dumps(row(item), ensure_ascii=False) + "\n")
    with open(directory / "rejected.jsonl", "w", encoding="utf-8") as fh:
        for item in rejected:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"admitted": len(records), "rejected": len(rejected),
               "by_family": {f: sum(1 for r in records if r["task"].family == f) for f in FAMILIES},
               "rejected_reasons": sorted({reason for r in rejected for reason in r["reasons"]})}
    (directory / "admission_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
