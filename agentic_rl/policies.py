"""规则基线策略：《项目定位与其他AI须知》要求的「问完几个槽位，再搜一次」规则基线，以及三个对照变体。

解析词典只覆盖训练集的 A 套说法（国家名、材质标签、阿拉伯数字和中文数字），这是规则方案的真实处境：
写规则的人只见过训练集。四个策略共用同一套解析和作答逻辑，只在三处有差别：
- careful：缺配送地或预算就先问，检索后按到手价逐件核对再推荐。
- trust_filter：相信检索工具的价格上限，不核对到手价（会推荐陷阱商品）。
- always_ask：不管条件是否给齐，开局把配送地、预算、材质全问一遍。
- never_ask：从不追问，缺配送地默认寄中国，缺预算就不设上限。
"""
from __future__ import annotations

import json
import re

COUNTRY = {"中国": "CN", "国内": "CN", "美国": "US", "日本": "JP", "新加坡": "SG", "德国": "EU", "欧洲": "EU", "法国": "EU"}
COUNTRY_NAME = {"CN": "中国", "US": "美国", "JP": "日本", "SG": "新加坡", "EU": "欧洲"}
MATERIALS = ("合成聚合物", "金属", "天然纤维", "玻璃", "陶瓷", "纸", "天然材料")
OPENINGS = ("帮我找一个", "想买个", "推荐一款")
VERIFY_WORDS = re.compile(r"核对|还有货|变了没|再查")
_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_BUDGET_CUE = re.compile(r"预算|以内|不超过|顶多|别超过|控制在|元|块|花")


def chinese_number(text: str) -> int | None:
    total, digit = 0, None
    for ch in text:
        if ch in _CN_DIGIT:
            digit = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            total += (1 if digit is None else digit) * _CN_UNIT[ch]
            digit = None
        else:
            return None
    total += digit or 0
    return total or None


def parse_budget(text: str) -> int | None:
    for segment in re.split(r"[，。；,;]", text):
        if not _BUDGET_CUE.search(segment):
            continue
        number = re.search(r"\d+", segment)
        if number:
            return int(number.group())
        chinese = re.search(r"[零一二两三四五六七八九十百千]{2,}", segment)
        if chinese and (value := chinese_number(chinese.group())):
            return value
    return None


def parse_dest(text: str) -> str | None:
    for word, code in COUNTRY.items():
        if word in text:
            return code
    return None


def parse_excluded(text: str) -> list[str]:
    return [m for m in MATERIALS if m in text and re.search(r"不要|别|还是", text)]


def parse_keyword(text: str) -> str:
    segment = re.split(r"[，。]", text)[0]
    for opening in OPENINGS:
        if segment.startswith(opening):
            return segment[len(opening):]
    return segment


def _tool_json(message: dict) -> dict | None:
    try:
        value = json.loads(message.get("content", ""))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def analyze(messages: list) -> dict:
    """从对话里复原当前条件、最近一次检索、是否在等表单、是否要求核对。规则策略不保留其他状态。"""
    users = [(i, m["content"]) for i, m in enumerate(messages) if m["role"] == "user"]
    first = users[0][1]
    cond = {"keyword": parse_keyword(first), "ship_to": parse_dest(first), "budget": parse_budget(first),
            "excluded": parse_excluded(first)}
    relevant, verify_at, asked = users[0][0], None, set()
    for index, text in users[1:]:
        if text.startswith("已提交表单"):
            cond["ship_to"] = parse_dest(text) or cond["ship_to"]
            cond["budget"] = parse_budget(text) or cond["budget"]
            cond["excluded"] = sorted(set(cond["excluded"]) | set(parse_excluded(text)))
            relevant = index
        elif VERIFY_WORDS.search(text):
            verify_at = index
        else:  # 买家改了条件
            cond["ship_to"] = parse_dest(text) or cond["ship_to"]
            cond["budget"] = parse_budget(text) or cond["budget"]
            cond["excluded"] = sorted(set(cond["excluded"]) | set(parse_excluded(text)))
            relevant = index
    calls, last_search, exact_after_verify, finals, waiting = {}, None, False, [], False
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            for call in m.get("tool_calls", []) or []:
                calls[call["id"]] = (i, call["function"]["name"], json.loads(call["function"]["arguments"]))
                if call["function"]["name"] == "show_shopping_form":
                    asked.update(q.get("id") for q in json.loads(call["function"]["arguments"]).get("questions", []))
            if not m.get("tool_calls"):
                if waiting:
                    waiting = False
                else:
                    finals.append(i)
        elif m["role"] == "tool" and m["tool_call_id"] in calls:
            at, name, args = calls[m["tool_call_id"]]
            waiting = name == "show_shopping_form"
            if name == "product_search_tool":
                result = _tool_json(m) or {}
                if args.get("product_id") or args.get("sku_id"):
                    if verify_at is not None and at > verify_at:
                        exact_after_verify = True
                else:
                    last_search = (at, args, result)
    last = messages[-1]
    return {"cond": cond, "relevant": relevant, "verify_at": verify_at, "asked": asked, "last_search": last_search,
            "exact_after_verify": exact_after_verify, "finals": finals, "waiting_form": last["role"] == "tool" and waiting}


def _form(slots: list[str]) -> dict:
    labels = {"ship_to": ("配送国家", "single_select"), "budget": ("预算（到手价，人民币）", "number"),
              "excluded": ("不想要的材质", "text")}
    return {"content": "", "tool_calls": [{"name": "show_shopping_form", "arguments": {
        "title": "补充选购条件", "description": "这些条件会改变候选商品，请补充后我再检索。",
        "questions": [{"id": s, "type": labels[s][1], "label": labels[s][0]} for s in slots]}}]}


def _search(cond: dict) -> dict:
    args = {"normalized_query": cond["keyword"], "target_currency": "CNY", "top_k": 10}
    if cond["ship_to"]:
        args["ship_to"] = cond["ship_to"]
    if cond["budget"]:
        args["price_max_major"] = cond["budget"]
    if cond["excluded"]:
        args["excluded_material_tags"] = cond["excluded"]
    return {"content": "", "tool_calls": [{"name": "product_search_tool", "arguments": args}]}


def _landed(hit: dict) -> float | None:
    value = (hit.get("landed_price") or {}).get("landed_total_major")
    return float(value) if value is not None else None


def _answer(cond: dict, hits: list, check_landed: bool) -> dict:
    picks = []
    for hit in hits:
        if cond["keyword"] not in hit.get("title", ""):
            continue
        landed = _landed(hit)
        if check_landed and (landed is None or (cond["budget"] and landed > cond["budget"])):
            continue
        picks.append((hit, landed))
    picks = picks[:3]
    dest = COUNTRY_NAME.get(cond["ship_to"], "未指定地区")
    head = f"按你的条件（{cond['keyword']}，寄到{dest}" + (f"，到手价不超过 {cond['budget']} 元" if cond["budget"] else "")
    head += (f"，不要{'、'.join(cond['excluded'])}材质" if cond["excluded"] else "") + "）"
    if not picks:
        return {"content": head + "没有找到满足全部条件的商品。可以考虑放宽预算或换一个收货地，我再帮你查。"}
    lines = [head + f"，为你挑了 {len(picks)} 款："]
    for n, (hit, landed) in enumerate(picks, 1):
        price = f"单件到手价 {landed:.0f} 元" if (check_landed and landed is not None) else f"单件价 {hit.get('price_major')} 元"
        lines.append(f"{n}. {hit.get('title')}（{hit.get('product_id')}），{price}")
    return {"content": "\n".join(lines)}


def _rule(messages: list, *, check_landed: bool, ask: str) -> dict:
    st = analyze(messages)
    cond = dict(st["cond"])
    if st["waiting_form"]:
        return {"content": "请先在表单里补充这些信息，我再继续为你挑选。"}
    first_turn = not any(m["role"] == "assistant" for m in messages)
    if ask == "always" and first_turn:
        return _form(["ship_to", "budget", "excluded"])
    missing = [s for s in ("ship_to", "budget") if not cond[s]]
    if ask == "missing" and missing and not set(missing) & st["asked"]:
        return _form(missing)
    if ask == "never" and not cond["ship_to"]:
        cond["ship_to"] = "CN"
    if st["verify_at"] is not None and not st["exact_after_verify"] and st["finals"]:
        previous = re.findall(r"（(P\d{4,})）", messages[st["finals"][-1]]["content"])
        if previous:
            return {"content": "", "tool_calls": [{"name": "product_search_tool", "arguments": {
                "product_id": previous[0], "ship_to": cond["ship_to"] or "CN", "target_currency": "CNY"}}]}
    if st["last_search"] is None or st["last_search"][0] < st["relevant"]:
        return _search(cond)
    return _answer(cond, st["last_search"][2].get("hits", []), check_landed)


def careful(messages, view=None):
    return _rule(messages, check_landed=True, ask="missing")


def trust_filter(messages, view=None):
    return _rule(messages, check_landed=False, ask="missing")


def always_ask(messages, view=None):
    return _rule(messages, check_landed=True, ask="always")


def never_ask(messages, view=None):
    return _rule(messages, check_landed=True, ask="never")


POLICIES = {"careful": careful, "trust_filter": trust_filter, "always_ask": always_ask, "never_ask": never_ask}
