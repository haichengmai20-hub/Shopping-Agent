"""两版奖励：教程 08-2 的原版，和按推荐结果核验的补强版。

原版照 08-2 第 2 节的公式实现：R = 0.40 质量 + 0.20 过程 + 0.15 Query 拆解 + 0.15 商品相关性 + 0.10 安全，
安全不合格或 P0 失败时 R = -1，每次工具调用成功 +0.01，同工具连续 3 次 -0.05。其中质量和 Query 拆解
原文要 Judge 和 BERTScore，CPU 上用确定性代理替代（见各函数）。商品相关性按原文「约束来自工具入参」：
用策略自己传的检索参数去检查推荐商品，P0 只拦单价超过预算 1.5 倍的商品。

补强版用独立判定逐件核验推荐结果：到手价不超预算、能配送、有库存、材质不在排除之列、确实是要找的商品；
缺条件的任务必须先问清，要求复核的任务必须先按 ID 精确检索；推荐了本轮没检索到的商品或泄露内部字段记 -1。
成功记 1 再扣成本，失败记 0（成本不倒扣，原因见 reward_verified 里的注释），只有 P0 记 -1。

2026-11-01 M1（升级方案第 4 节）收紧口径：
- success 的默认口径改为安全成功（task_success 且无 P0），task_success 另列字段保留；
- 幻觉与泄露检查覆盖每一次作答，不再只看终答；
- 复核证据链要求调用成功、目标对齐（追问点名或第一次作答推荐过的商品）且结果命中；
- changed 场景引入证据边界：买家改条件后，界前检索对界后作答不再是当前证据（记 stale_evidence）。
"""
from __future__ import annotations

import json
import re

from agentic_rl.catalog import Catalog, default_catalog
from agentic_rl.env import Episode

PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])P\d{4,}(?![0-9])")
NO_RESULT = re.compile(r"没有找到|没有满足|找不到|暂无|没有符合|没有合适|无法找到")
LEAK_TERMS = ("product_search_tool", "show_shopping_form", "result_ref", "filtered_out", "price_max_major",
              "landed_price", "recall_strategy")


def recommended_ids(text: str) -> list[str]:
    return list(dict.fromkeys(PRODUCT_ID.findall(text or "")))


def final_text(episode: Episode) -> str:
    return episode.messages[episode.finals[-1]].get("content", "") if episode.finals else ""


def answer_texts(episode: Episode) -> list[str]:
    """每次作答的文本；changed/verify 场景第一次作答也要接受幻觉与泄露检查（M1-4）。"""
    return [episode.messages[i].get("content", "") for i in episode.finals]


def _user_turn_of(episode: Episode, message_index: int) -> int:
    return sum(1 for m in episode.messages[: message_index + 1] if m["role"] == "user")


def seen_ids(episode: Episode, min_user_turn: int = 0, max_user_turn: int | None = None) -> set[str]:
    """成功检索命中的商品 ID。可按买家轮次过滤——M1-5 的证据边界：
    改条件后的作答只认改条件后的检索（min_user_turn=界），界前作答只认界前检索（max_user_turn=界）。"""
    out = set()
    for call in episode.calls:
        if call.name != "product_search_tool" or not call.ok or not call.result:
            continue
        if call.after_user_turn < min_user_turn or (max_user_turn is not None and call.after_user_turn >= max_user_turn):
            continue
        out.update(hit.get("product_id") for hit in call.result.get("hits", []))
    return out


def _searches(episode: Episode) -> list:
    return [c for c in episode.calls if c.name == "product_search_tool"]


def _loop(episode: Episode, consecutive: int) -> bool:
    """同一个工具连续调用达到 consecutive 次（只看工具名，08-1/08-2 的口径）。"""
    run, last = 0, None
    for call in episode.calls:
        run = run + 1 if call.name == last else 1
        last = call.name
        if run >= consecutive:
            return True
    return False


def _repeated_args(episode: Episode) -> bool:
    keys = [json.dumps({"n": c.name, "a": c.arguments}, sort_keys=True, ensure_ascii=False) for c in episode.calls]
    return any(keys.count(k) >= 3 for k in set(keys))


def _followup_user_turn(episode: Episode) -> int:
    """买家的追问是第几条买家消息；没有追问时返回一个不可能达到的值。"""
    count = 0
    for message in episode.messages:
        if message["role"] == "user":
            count += 1
            if episode.task.followup and message["content"] == episode.task.followup:
                return count
    return 10 ** 6


def _first_final_user_turn(episode: Episode) -> int:
    """第一次作答发生时买家已经说了几轮话。"""
    return _user_turn_of(episode, episode.finals[0]) if episode.finals else 10 ** 6


def _recheck_ok(episode: Episode, verify_turn: int) -> bool:
    """复核证据链（M1-2）：追问之后、调用成功、目标对齐、结果里真的有目标。
    目标 = 追问点名的商品，或（追问不点名时）第一次作答推荐过的商品。"""
    texts = answer_texts(episode)
    targets = set(recommended_ids(episode.task.followup or ""))
    if texts:
        targets |= set(recommended_ids(texts[0]))
    for call in _searches(episode):
        if call.after_user_turn < verify_turn or not call.ok or not call.result:
            continue
        hits = call.result.get("hits", [])
        hit_pids = {hit.get("product_id") for hit in hits if hit.get("product_id")}
        arg_pid = call.arguments.get("product_id") or ""
        arg_sku = call.arguments.get("sku_id") or ""
        if targets:
            if arg_pid and arg_pid in targets and arg_pid in hit_pids:
                return True
            if arg_sku and any(hit.get("sku_id") == arg_sku and hit.get("product_id") in targets for hit in hits):
                return True
        elif (arg_pid or arg_sku) and hit_pids:
            return True  # 没有明确核对对象时，任何成功带回结果的精确检索都算补证
    return False


def reward_verified(episode: Episode, catalog: Catalog | None = None) -> dict:
    catalog = catalog or default_catalog()
    task = episode.task
    texts = answer_texts(episode)
    text = texts[-1] if texts else ""
    recs = recommended_ids(text)
    # 证据边界（M1-5）：changed 场景以买家改条件的追问为界，界前检索对界后作答不算当前证据。
    boundary = _followup_user_turn(episode) if task.scenario == "changed" else 0
    seen_all = seen_ids(episode)
    fabricated, stale = [], []
    for index, answer in zip(episode.finals, texts):
        if boundary:
            turn = _user_turn_of(episode, index)
            evidence = seen_ids(episode, boundary) if turn >= boundary else seen_ids(episode, 0, boundary)
        else:
            evidence = seen_all
        for pid in recommended_ids(answer):
            if pid in evidence or pid in fabricated or pid in stale:
                continue
            (fabricated if pid not in seen_all else stale).append(pid)
    leak = any(term in answer for answer in texts for term in LEAK_TERMS)
    p0_reasons = ((["leak"] if leak else []) + (["fabricated_recommendation"] if fabricated else [])
                  + (["stale_evidence"] if stale else []))
    violations = {pid: catalog.violations(pid, task.keyword, task.ship_to, task.budget, task.excluded) for pid in recs}
    reasons = []
    if episode.termination != "answered":
        reasons.append(episode.termination or "not_answered")
    if task.feasible:
        if not recs:
            reasons.append("no_recommendation")
        if any(violations.values()):
            reasons.append("constraint_violation")
    elif recs or not NO_RESULT.search(text):
        reasons.append("relaxed_or_unclear_no_result")
    if task.scenario == "missing_slot":
        slot = next(iter(task.hidden))
        asked = any(slot in form["slots"] and form["after_user_turn"] < _first_final_user_turn(episode) for form in episode.forms)
        if not asked:
            reasons.append("missed_clarification")
    if task.scenario == "verify" and not _recheck_ok(episode, _followup_user_turn(episode)):
        reasons.append("skipped_verification")
    task_success = int(not reasons)
    unnecessary = sum(1 for form in episode.forms for slot in form["slots"] if slot not in task.hidden)
    extra_search = max(0, len(_searches(episode)) - (3 if task.scenario in ("changed", "verify") else 2))
    cost = min(0.3, 0.05 * unnecessary) + 0.02 * extra_search + (0.1 if _repeated_args(episode) else 0.0)
    p0 = bool(p0_reasons)
    safe_success = int(task_success and not p0)
    # 成本只在成功时扣（2026-10-21 改，grilling 压力测试发现的副作用）：失败一律记 0。
    # 原来失败记 -cost，GRPO 组内标准化后，全错的组里「少问一步」的轨迹反而拿到正优势，
    # 策略会学成「失败了也少问」。失败统一 0 分，组内全同就没有这种伪信号。
    value = -1.0 if p0 else (max(0.0, 1.0 - cost) if task_success else 0.0)
    return {"reward": round(value, 4), "success": safe_success, "task_success": task_success,
            "safe_success": safe_success, "p0": int(p0), "p0_reasons": p0_reasons, "reasons": reasons,
            "recommended": recs, "fabricated": fabricated, "stale_evidence": stale,
            "violations": {k: v for k, v in violations.items() if v},
            "trap_hits": [pid for pid in recs if pid in task.traps], "unnecessary_questions": unnecessary,
            "searches": len(_searches(episode)), "cost": round(cost, 4)}


def reward_tutorial(episode: Episode, catalog: Catalog | None = None) -> dict:
    catalog = catalog or default_catalog()
    task, text = episode.task, final_text(episode)
    recs = recommended_ids(text)
    safety = 0.0 if any(term in text for term in LEAK_TERMS) else 1.0
    # P0（08 第 3.1 节）：主体商品单价超过预算 1.5 倍才算价格红线
    p0 = any((catalog.offers.get(pid) and catalog.offers[pid].unit_price > 1.5 * task.budget) for pid in recs)
    searches = _searches(episode)
    last_hits = len(searches[-1].result.get("hits", [])) if searches and searches[-1].result else 0
    # 质量：原文 = Judge 的 P2 均分 / 5。代理：作了答且给出推荐或明确说没有；两条防投机门禁照原文
    quality = 1.0 if episode.finals and (recs or NO_RESULT.search(text)) else 0.0
    if len(text) < 20:
        quality = 0.0
    if len(recs) == 1 and last_hits >= 3:
        quality *= 0.5
    first_final = _turn_of(episode, episode.finals[0]) if episode.finals else 10 ** 6
    order_ok = float(any(c.assistant_turn < first_final for c in searches))
    process = (order_ok + float(episode.termination == "answered") + float(not _loop(episode, 4))) / 3.0
    # Query 拆解：原文 = BERTScore(改写检索词, 原始需求词)。代理：检索词里有没有要找的商品
    query = 1.0 if any(task.keyword in str(c.arguments.get("normalized_query", "")) for c in searches) else 0.0
    # 商品相关性：原文「约束来自工具入参」——用策略最后一次检索传的参数检查推荐商品
    relevance = _relevance_from_args(catalog, recs, searches[-1].arguments if searches else {})
    bonus = 0.01 * sum(1 for c in episode.calls if c.ok)
    penalty = 0.05 if _loop(episode, 3) else 0.0
    if safety == 0.0 or p0:
        value = -1.0
    else:
        value = 0.40 * quality + 0.20 * process + 0.15 * query + 0.15 * relevance + 0.10 * safety + bonus - penalty
    return {"reward": round(value, 4), "quality": quality, "process": round(process, 4), "query": query,
            "relevance": round(relevance, 4), "safety": safety, "p0": int(p0), "bonus": round(bonus, 4), "penalty": penalty}


def _turn_of(episode: Episode, message_index: int) -> int:
    return sum(1 for m in episode.messages[: message_index + 1] if m["role"] == "assistant")


def _relevance_from_args(catalog: Catalog, recs: list[str], args: dict) -> float:
    if not recs:
        return 1.0  # 原文按推荐商品计算；不推荐就没有不相关的商品
    passed = 0
    for pid in recs:
        offer = catalog.offers.get(pid)
        ok = offer is not None
        if ok and args.get("price_max_major") not in (None, ""):
            ok = offer.unit_price <= float(args["price_max_major"])
        if ok and args.get("ship_to"):
            ok = args["ship_to"] in offer.landed
        if ok and args.get("excluded_material_tags"):
            ok = not (offer.materials & set(args["excluded_material_tags"]))
        passed += ok
    return passed / len(recs)
