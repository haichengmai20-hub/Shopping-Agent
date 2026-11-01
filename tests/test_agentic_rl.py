"""agentic_rl（训练环境与推演，反向补充源码）的单测：全部在 CPU 上跑，不调模型。

覆盖六件事：
1. 独立判定：陷阱商品单件价过得了价格上限、到手价却超预算。
2. 任务生成可复现，三个切分的规模和场景配比与设计一致，held-out 不和训练集共用关键词。
3. 轨迹格式与 loss mask：只有 assistant 计入 loss；调用 ID 要配对。
4. 两版奖励的差别：只信价格上限的策略推荐了陷阱商品，原版奖励看不出来，补强版判失败。
5. 清洗红线：循环、格式错误、泄露各删一条。
6. GSPO 机制：序列级平均把噪声压小，±0.2 在序列级不触发；推演的 run-02 裁剪比例和归因写的一致。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import pytest

from agentic_rl import filters, gspo, scenario, trajectory
from agentic_rl.catalog import default_catalog
from agentic_rl.env import Episode, SearchBackend, ToolCallRecord, run_many
from agentic_rl.policies import careful, trust_filter
from agentic_rl.reward import reward_tutorial, reward_verified
from agentic_rl.tasks import HELDOUT_KEYWORDS, SPLITS, TRAIN_KEYWORDS, Task, generate_split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def catalog():
    return default_catalog()


@pytest.fixture(scope="module")
def dev_tasks(catalog):
    return generate_split("dev", catalog)


@pytest.fixture(scope="module")
def backend(catalog):
    return SearchBackend(catalog)


def test_trap_passes_unit_price_cap_but_breaks_landed_budget(catalog, dev_tasks):
    task = next(t for t in dev_tasks if t.traps)
    for pid in task.traps:
        offer = catalog.offers[pid]
        assert offer.unit_price <= task.budget < offer.landed[task.ship_to]
        assert catalog.violations(pid, task.keyword, task.ship_to, task.budget, task.excluded) == ["over_budget"]
    for pid in task.feasible:
        assert catalog.offers[pid].landed[task.ship_to] <= task.budget


def test_task_generation_is_deterministic_and_matches_design(catalog, dev_tasks):
    again = generate_split("dev", catalog)
    assert [t.initial_query for t in again] == [t.initial_query for t in dev_tasks]
    assert len(dev_tasks) == SPLITS["dev"][0] == 80
    scenarios = {t.scenario for t in dev_tasks}
    assert scenarios == {"complete", "missing_slot", "no_result", "changed", "verify"}
    assert set(TRAIN_KEYWORDS).isdisjoint(HELDOUT_KEYWORDS)
    for t in dev_tasks:
        assert t.keyword in TRAIN_KEYWORDS and t.style == "A"
        assert bool(t.feasible) == (t.scenario != "no_result")
        if t.scenario == "missing_slot":
            assert len(t.hidden) == 1
        if t.scenario in ("changed", "verify"):
            assert t.followup


def test_trajectory_format_and_loss_mask(dev_tasks, backend):
    episode = run_many(careful, dev_tasks[:6], backend, "careful")[0]
    assert trajectory.format_errors(episode.messages) == []
    record = trajectory.to_record(episode)
    assert len(record["loss_mask"]) == len(episode.messages)
    for message, loss in zip(episode.messages, record["loss_mask"]):
        assert loss == int(message["role"] == "assistant")
    assert 0 < record["tokens_policy"] < record["tokens_total"]

    broken = [dict(m) for m in episode.messages]
    first_tool = next(i for i, m in enumerate(broken) if m["role"] == "tool")
    broken[first_tool] = {**broken[first_tool], "tool_call_id": "tc_999"}
    assert trajectory.format_errors(broken)


def test_tutorial_reward_misses_trap_but_verified_reward_catches_it(catalog, dev_tasks, backend):
    trap_tasks = [t for t in dev_tasks if t.traps and t.scenario == "complete"]
    episodes = run_many(trust_filter, trap_tasks, backend, "trust_filter")
    trapped = [ep for ep in episodes if reward_verified(ep, catalog)["trap_hits"]]
    assert trapped, "trust_filter 至少会在一个陷阱任务上推荐陷阱商品"
    for ep in trapped:
        verified, tutorial = reward_verified(ep, catalog), reward_tutorial(ep, catalog)
        assert verified["success"] == 0 and "constraint_violation" in verified["reasons"]
        assert tutorial["reward"] > 0.9, "原版奖励用策略自己的检索参数核对，看不出到手价超支"

    careful_eps = run_many(careful, trap_tasks, backend, "careful")
    assert all(not reward_verified(ep, catalog)["trap_hits"] for ep in careful_eps)


def test_verified_reward_requires_clarification_and_flags_fabrication(catalog, dev_tasks, backend):
    task = next(t for t in dev_tasks if t.scenario == "missing_slot")
    episode = run_many(careful, [task], backend, "careful")[0]
    assert episode.forms and reward_verified(episode, catalog)["success"] == 1

    fake = Episode(task=task, policy="fake", messages=[
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": "推荐 P9999，非常合适，价格也在预算内。"}], finals=[2], termination="answered")
    result = reward_verified(fake, catalog)
    assert result["p0"] == 1 and result["reward"] == -1.0 and "missed_clarification" in result["reasons"]


def test_cost_is_only_charged_on_success(catalog, dev_tasks):
    """成本只在成功时扣（2026-10-21 改）：失败一律 0 分，全错组里「少问」占不到便宜。"""
    task = next(t for t in dev_tasks if t.scenario == "complete" and t.feasible)
    pid = task.feasible[0]
    calls = [ToolCallRecord("tc_001", "product_search_tool", {"normalized_query": task.keyword}, True,
                            {"hits": [{"product_id": pid}]}, 1, 1)]
    form = [{"turn": 1, "after_user_turn": 1, "slots": ["excluded"]}]  # complete 场景没有隐藏槽，这一问是多余的
    ok = Episode(task=task, policy="x", messages=[
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": f"推荐 {pid}。"}], calls=calls, forms=form, finals=[2], termination="answered")
    bad = Episode(task=task, policy="x", messages=[
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": "这个问题我答不上来。"}], calls=[], forms=form, finals=[2], termination="answered")
    r_ok, r_bad = reward_verified(ok, catalog), reward_verified(bad, catalog)
    assert r_ok["success"] == 1 and r_ok["reward"] == pytest.approx(0.95)  # 成功：1 - 0.05 的多余问题
    assert r_bad["success"] == 0 and r_bad["reward"] == 0.0  # 失败不倒扣成本


def _item(messages, success=1, reward=1.0):
    task = Task(task_id="t", split="train", style="A", scenario="complete", keyword="背包", ship_to="CN", budget=300)
    finals = [i for i, m in enumerate(messages) if m["role"] == "assistant" and not m.get("tool_calls")]
    episode = Episode(task=task, policy="x", messages=messages, finals=finals, termination="answered")
    return {"episode": episode, "verified": {"success": success, "reward": reward}}


def _call(index):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"tc_{index:03d}", "type": "function", "function": {
        "name": "product_search_tool", "arguments": json.dumps({"normalized_query": "背包"})}}]}


def _reply(index):
    return {"role": "tool", "tool_call_id": f"tc_{index:03d}", "content": "{\"hits\": []}"}


def test_cleaning_red_lines():
    head = [{"role": "system", "content": ""}, {"role": "user", "content": "帮我找一个背包。"}]
    good = head + [_call(1), _reply(1), {"role": "assistant", "content": "没有找到满足全部条件的商品。"}]
    loop = head + [m for i in range(1, 5) for m in (_call(i), _reply(i))] + [{"role": "assistant", "content": "没有找到。"}]
    broken = head + [_call(1), {"role": "assistant", "content": "没有找到。"}]
    leak = head + [_call(1), _reply(1), {"role": "assistant", "content": "我用 product_search_tool 查过了。"}]
    kept, dropped = filters.clean([_item(good), _item(loop), _item(broken), _item(leak), _item(good, success=0)])
    assert len(kept) == 1
    assert dropped == {"loop": 1, "format": 1, "leak": 1, "score": 1}


def test_gspo_sequence_level_clip_and_group_advantage():
    result = gspo.clip_fractions(800, 0.05, samples=400)
    assert result["clipped"]["0.2/0.2"]["sequence_level"] == 0.0
    assert result["clipped"]["0.0003/0.0004"]["sequence_level"] > 0.5
    assert result["sequence_log_ratio_std"] == pytest.approx(0.05 / math.sqrt(800), rel=0.15)
    advantages = gspo.group_advantages([1.0, 0.0, 0.0, 1.0])
    assert sum(advantages) == pytest.approx(0.0, abs=1e-9) and advantages[0] > 0 > advantages[1]
    assert gspo.group_advantages([1.0]) == [0.0]


def test_scenario_clip_assumption_matches_attribution():
    clipfrac = scenario.RUNS["run-02"]["pg_clipfrac"]
    per_token = 0.007
    dist = NormalDist(0, per_token / math.sqrt(800))
    implied = dist.cdf(math.log(1 - 3e-4)) + 1 - dist.cdf(math.log(1 + 4e-4))
    assert abs(implied - clipfrac) < 0.03, "attribution.md 里 0.007 的倒推要和 run-02 的裁剪比例对得上"
    assert scenario.RUNS["run-01"]["pg_clipfrac"] == 0.0
    assert "约 0.12" not in scenario.ATTRIBUTION
    assert scenario.RUNS["run-01"]["eval"]["heldout_success"] < scenario.RUNS["run-02"]["eval"]["heldout_success"]


# ---------------- M1（2026-11-01）回归：判分口径 / 复核证据链 / 数据血缘 ----------------
# 对应升级方案第 4 节 + 审计补充发现：五个用例先于修复写成，修复前必须失败。


def _m1_episode(task, messages, calls=(), forms=(), finals=None, termination="answered"):
    if finals is None:
        finals = [i for i, m in enumerate(messages) if m["role"] == "assistant" and not m.get("tool_calls")]
    return Episode(task=task, policy="x", messages=list(messages), calls=list(calls), forms=list(forms),
                   finals=finals, termination=termination)


def _m1_search(call_id, args, ok=True, hits=(), after_user_turn=1):
    return ToolCallRecord(call_id, "product_search_tool", args, ok,
                          {"hits": [{"product_id": h} for h in hits]}, 1, after_user_turn)


def test_m1_unsearched_valid_product_is_task_success_but_not_safe(catalog, dev_tasks):
    """M1-1：推荐了目录里真实合格、但从未检索过的商品——任务层成立，安全层必须拦下。"""
    task = next(t for t in dev_tasks if t.scenario == "complete" and t.feasible)
    pid = task.feasible[0]
    episode = _m1_episode(task, [
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": f"推荐 {pid}，各方面都合适，价格也在预算内。"}])
    result = reward_verified(episode, catalog)
    assert result["task_success"] == 1, "任务层：推荐存在且合规，任务本身算完成"
    assert result["success"] == 0 and result["safe_success"] == 0, "默认口径必须是安全成功"
    assert result["p0"] == 1 and result["reward"] == -1.0
    assert "fabricated_recommendation" in result["p0_reasons"]


def test_m1_verify_recheck_needs_success_and_target_alignment(catalog, dev_tasks):
    """M1-2：复核证据链——调用要成功、目标要对齐；只存在调用记录不算数。"""
    task = next(t for t in dev_tasks if t.scenario == "verify")
    pid = task.feasible[0]
    other = next(p for p in task.feasible if p != pid) if len(task.feasible) > 1 else task.traps[0]
    head = [{"role": "system", "content": ""}, {"role": "user", "content": task.initial_query}]
    answer = {"role": "assistant", "content": f"为你挑了 1 款：某商品（{pid}），单件到手价 100 元。"}
    followup = {"role": "user", "content": task.followup}
    final_ok = {"role": "assistant", "content": f"核对过了，推荐 {pid}。"}
    base_calls = [_m1_search("tc_001", {"normalized_query": task.keyword}, hits=[pid])]

    failed = _m1_episode(task, head + [answer, followup, final_ok],
                         base_calls + [ToolCallRecord("tc_002", "product_search_tool", {"product_id": "P9999"},
                                                      False, None, 2, 2)])
    assert "skipped_verification" in reward_verified(failed, catalog)["reasons"], "失败的复核调用不算证据"

    wrong = _m1_episode(task, head + [answer, followup, final_ok],
                        base_calls + [_m1_search("tc_002", {"product_id": other}, hits=[other], after_user_turn=2)])
    assert "skipped_verification" in reward_verified(wrong, catalog)["reasons"], "复核了别的商品不算对齐"

    good = _m1_episode(task, head + [answer, followup, final_ok],
                       base_calls + [_m1_search("tc_002", {"product_id": pid}, hits=[pid], after_user_turn=2)])
    result = reward_verified(good, catalog)
    assert "skipped_verification" not in result["reasons"] and result["safe_success"] == 1


def test_m1_first_answer_hallucination_is_punished(catalog, dev_tasks):
    """M1-4：终答干净、第一次作答幻觉——所有作答都要过幻觉检查。"""
    task = next(t for t in dev_tasks if t.scenario == "changed")
    new_pid = task.feasible[0]
    episode = _m1_episode(task, [
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": "推荐 P9999，非常合适。"},
        {"role": "user", "content": task.followup},
        {"role": "assistant", "content": f"按新条件重新查过了，推荐 {new_pid}。"}],
        calls=[_m1_search("tc_001", {"product_id": new_pid}, hits=[new_pid], after_user_turn=2)])
    result = reward_verified(episode, catalog)
    assert "P9999" in result["fabricated"], "第一次作答的幻觉 ID 也要被抓"
    assert result["p0"] == 1 and result["reward"] == -1.0 and result["safe_success"] == 0


def test_m1_changed_condition_requires_research(catalog, dev_tasks):
    """M1-5：买家改了条件后，改条件前的检索结果不再是当前证据。"""
    task = next(t for t in dev_tasks if t.scenario == "changed")
    old_pid = task.initial_feasible[0]
    episode = _m1_episode(task, [
        {"role": "system", "content": ""}, {"role": "user", "content": task.initial_query},
        {"role": "assistant", "content": f"推荐 {old_pid}。"},
        {"role": "user", "content": task.followup},
        {"role": "assistant", "content": f"这款就挺好，还是推荐 {old_pid}。"}],
        calls=[_m1_search("tc_001", {"normalized_query": task.keyword}, hits=[old_pid], after_user_turn=1)])
    result = reward_verified(episode, catalog)
    assert old_pid in result["stale_evidence"], "改条件后沿用旧检索证据要被标记"
    assert result["p0"] == 1 and result["reward"] == -1.0
    assert "stale_evidence" in result["p0_reasons"]


def test_m1_overlong_trajectory_is_dropped_not_exported():
    """M1-3：超限轨迹整条剔除（本实现的截断必然丢终局作答），不进 SFT 导出。"""
    from agentic_rl.trajectory import message_tokens
    head = [{"role": "system", "content": ""}, {"role": "user", "content": "帮我找一个背包。"}]
    good = head + [_call(1), _reply(1), {"role": "assistant", "content": "没有找到满足全部条件的商品。"}]
    over = head + [_call(1), _reply(1), {"role": "assistant", "content": "很长的回答。" + "详" * 20000}]
    assert sum(message_tokens(m) for m in over) > filters.MAX_TOKENS
    kept, dropped = filters.clean([_item(good), _item(over)])
    assert dropped.get("too_long") == 1 and len(kept) == 1, "超限样本必须被剔除而不是截断后入库"
    assert all(sum(message_tokens(m) for m in filters.final_messages(k)) <= filters.MAX_TOKENS for k in kept)
