"""E1 下单环境（order_env / order_tasks）的单测：状态转移、事务性、守恒、隔离、重放、准入。

全部 CPU 运行、0 次模型调用（检索走线上工具函数的关键词兜底路径）。对应升级方案 5.1 的验收项。
"""
from __future__ import annotations

import json

import pytest

from agentic_rl.catalog import default_catalog
from agentic_rl.env import SearchBackend
from agentic_rl.order_env import (MAX_STEPS, OrderCarefulPolicy, OrderEnvironment, OrderSnapshot, OrderTask,
                                  run_order_episode, verify_order_episode)
from agentic_rl.order_tasks import admit, generate_order_pool


@pytest.fixture(scope="module")
def catalog():
    return default_catalog()


@pytest.fixture(scope="module")
def backend(catalog):
    return SearchBackend(catalog)


def _place_task(catalog, family="order_place"):
    """从目录构造一个可解的下单任务：取第一个有关键词存量的目的地，预算锚在可行商品上。"""
    for keyword in ("背包", "保温杯", "收纳盒"):
        for dest in ("CN", "US", "JP"):
            base = catalog.eligible(keyword, dest)
            if len(base) < 3:
                continue
            landed = sorted(catalog.offers[p].landed[dest] for p in base)
            budget = int(landed[len(landed) // 2]) + 50
            feasible = catalog.feasible(keyword, dest, budget)
            if not feasible:
                continue
            inventory = {pid: 2 for pid in base}
            if family == "order_out_of_stock":
                for pid in feasible:
                    inventory[pid] = 0
            task = OrderTask(task_id=f"t-{family}", family=family, split="train", keyword=keyword, ship_to=dest,
                             budget=budget, excluded=[], initial_query=f"帮我找一个{keyword}，预算 {budget} 元以内。",
                             followup="想了想还是不要了，帮我取消吧。" if family == "order_cancel" else "",
                             goal={"order_place": {"type": "order_placed"},
                                   "order_out_of_stock": {"type": "no_order", "reason": "out_of_stock"},
                                   "order_cancel": {"type": "order_cancelled"}}[family],
                             solvability={"feasible": feasible})
            return task, OrderSnapshot(inventory)
    pytest.skip("目录里找不到可构造的任务组合")


def test_prepare_confirm_deducts_inventory_and_conserves(catalog, backend):
    task, snapshot = _place_task(catalog)
    env = run_order_episode(task, snapshot, catalog, seed=1, backend=backend)
    verdict = verify_order_episode(env)
    assert verdict["passed"], verdict["checks"]
    placed = [o for o in env.orders.values() if o["status"] == "placed"]
    assert len(placed) == 1
    pid = placed[0]["product_id"]
    assert env.inventory[pid] == snapshot.inventory[pid] - 1
    assert env.invariant_checks() == []


def test_failed_actions_never_half_write(catalog, backend):
    task, snapshot = _place_task(catalog)
    env = OrderEnvironment(catalog, backend)
    env.reset(task, snapshot, seed=1)
    before = json.dumps(env.state_for_verifier(), sort_keys=True, default=str)
    for action in ({"name": "prepare_order", "arguments": {"product_id": "P9999"}},
                   {"name": "cancel_order", "arguments": {"confirmation_id": "conf-999"}},
                   {"name": "not_a_tool", "arguments": {}}):
        env.step(action)
    after_orders = {k: v for k, v in env.state_for_verifier()["orders"].items()}
    assert not after_orders and not env.confirmations
    assert env.inventory == snapshot.inventory, "失败动作不得改变库存"
    assert json.dumps(env.state_for_verifier()["initial_inventory"]) == json.dumps(snapshot.inventory)
    assert before != json.dumps(env.state_for_verifier(), sort_keys=True, default=str)  # 消息在变，状态没变


def test_cancel_restores_inventory(catalog, backend):
    task, snapshot = _place_task(catalog, family="order_cancel")
    env = run_order_episode(task, snapshot, catalog, seed=1, backend=backend)
    assert verify_order_episode(env)["passed"], verify_order_episode(env)["checks"]
    cancelled = [o for o in env.orders.values() if o["status"] == "cancelled"]
    assert len(cancelled) == 1
    assert env.inventory[cancelled[0]["product_id"]] == snapshot.inventory[cancelled[0]["product_id"]]
    assert env.invariant_checks() == []


def test_out_of_stock_requires_honest_no_order(catalog, backend):
    task, snapshot = _place_task(catalog, family="order_out_of_stock")
    env = run_order_episode(task, snapshot, catalog, seed=1, backend=backend)
    verdict = verify_order_episode(env)
    assert verdict["passed"], verdict["checks"]
    assert not env.orders, "无货任务不应产生任何订单"

    silent = OrderEnvironment(catalog, backend)
    silent.reset(task, snapshot, seed=1)
    policy_answers = 0
    while not silent.terminated and not silent.truncated and policy_answers < 2:
        action = OrderCarefulPolicy(task).act(silent.messages) if policy_answers == 0 else {"type": "answer", "content": "已经买好了。"}
        if action.get("type") == "answer":
            policy_answers += 1
        silent.step(action)
    assert not verify_order_episode(silent)["passed"], "没下成单却谎称买好，必须判失败"


def test_reset_replay_is_deterministic(catalog, backend):
    task, snapshot = _place_task(catalog, family="order_cancel")
    digests = []
    for _ in range(2):
        env = run_order_episode(task, snapshot, catalog, seed=3, backend=backend)
        digests.append(json.dumps(env.messages, ensure_ascii=False, sort_keys=True))
    assert digests[0] == digests[1]


def test_concurrent_copies_are_isolated(catalog, backend):
    task, snapshot = _place_task(catalog)
    envs = [OrderEnvironment(catalog, backend) for _ in range(8)]
    policies = [OrderCarefulPolicy(task) for _ in range(8)]
    for env in envs:
        env.reset(task, snapshot, seed=5)
    done = [False] * 8
    while not all(done):
        for i, (env, policy) in enumerate(zip(envs, policies)):
            if not done[i]:
                env.step(policy.act(env.messages))
                done[i] = env.terminated or env.truncated
    states = {json.dumps(e.state_for_verifier(), sort_keys=True, default=str) for e in envs}
    assert len(states) == 1, "8 个并发副本的最终状态必须一致且互不污染"
    assert all(not e.invariant_checks() for e in envs)


def test_step_budget_truncates_instead_of_terminating(catalog, backend):
    task, snapshot = _place_task(catalog)
    env = OrderEnvironment(catalog, backend)
    env.reset(task, snapshot, seed=1)
    for _ in range(MAX_STEPS):
        if env.terminated or env.truncated:
            break
        env.step({"name": "product_search_tool",
                  "arguments": {"normalized_query": task.keyword, "ship_to": task.ship_to, "top_k": 5}})
    assert env.truncated and not env.terminated, "超预算必须记 truncated（外部截断），不能记 terminated"


def test_pool_generation_admits_and_rejects(catalog, backend):
    records, rejected = generate_order_pool(catalog, per_family=20, seed=202)
    assert len(records) >= 60, "每族至少 20 个通过准入的任务"
    by_family = {}
    for item in records:
        verdict = verify_order_episode(
            run_order_episode(item["task"], item["snapshot"], catalog, seed=0, backend=backend))
        assert verdict["passed"], f"准入任务 {item['task'].task_id} 的见证必须通过"
        by_family[item["task"].family] = by_family.get(item["task"].family, 0) + 1
    assert set(by_family) == {"order_place", "order_out_of_stock", "order_cancel"}
    # 准入门槛真的会拒绝：构造一个合格品有货但目标要求「正确无结果」的自相矛盾任务
    task, snapshot = _place_task(catalog, family="order_place")
    task.family, task.goal = "order_out_of_stock", {"type": "no_order", "reason": "out_of_stock"}
    decision = admit(task, snapshot, catalog, backend=backend)
    assert not decision["admitted"] and "witness_failed" in decision["reasons"]
