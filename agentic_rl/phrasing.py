"""买家说法库：训练和开发集用 A 套说法，held-out 用 B 套说法。

B 套故意换成训练集没出现过的表达（城市名代替国家、口语开场白、「塑料」代替材质标签、更多中文数字），
用来检验策略是理解了需求，还是只记住了训练集的句式。2026-10-19 实测，规则词典真正处理不了的是城市名和开场白；
中文数字预算它能解析。
"""
from __future__ import annotations

DESTINATIONS = {
    "CN": {"A": ["寄到中国", "寄回国内"], "B": ["寄到上海", "送到北京家里"]},
    "US": {"A": ["寄到美国", "寄美国"], "B": ["寄到纽约", "送到洛杉矶"]},
    "JP": {"A": ["寄到日本", "寄日本"], "B": ["寄到东京", "送到大阪"]},
    "SG": {"A": ["寄到新加坡"], "B": ["送到狮城"]},
    "EU": {"A": ["寄到德国", "寄到欧洲"], "B": ["寄到巴黎", "送到柏林"]},
}

MATERIALS = {
    "合成聚合物": {"A": ["不要合成聚合物材质的"], "B": ["别要塑料的"]},
    "金属": {"A": ["不要金属材质的"], "B": ["不要铁的、不锈钢的那种"]},
    "玻璃": {"A": ["不要玻璃材质的"], "B": ["别是玻璃做的"]},
    "陶瓷": {"A": ["不要陶瓷材质的"], "B": ["瓷的就算了"]},
    "天然纤维": {"A": ["不要天然纤维材质的"], "B": ["棉麻的不要"]},
}

OPENINGS = {"A": ["帮我找一个", "想买个", "推荐一款"], "B": ["有没有合适的", "我在挑", "给我看看"]}

BUDGET = {
    "A": ["预算 {n} 元", "{n} 元以内", "不超过 {n} 块"],
    "B": ["{cn}块钱以内", "顶多花 {n}", "别超过{cn}"],
}

CHANGE_BUDGET = {"A": ["预算改成 {n} 元吧", "还是控制在 {n} 元以内"], "B": ["再便宜点，{cn}以内", "钱有点紧，顶多 {n}"]}
CHANGE_DEST = {"A": ["改成{dest}吧"], "B": ["地址变了，{dest}"]}
CHANGE_MATERIAL = {"A": ["还是{mat}"], "B": ["想了想，{mat}"]}
VERIFY = {"A": ["第一个推荐现在的价格和库存，帮我再核对一下"], "B": ["你说的第一个，现在还有货吗？价钱变了没"]}

_DIGITS = "零一二三四五六七八九"


def to_chinese(n: int) -> str:
    """把 1–9999 的整数写成中文数字，如 230 → 二百三十、305 → 三百零五。"""
    if not 0 < n < 10000:
        raise ValueError("只支持 1–9999")
    units = [(1000, "千"), (100, "百"), (10, "十"), (1, "")]
    out, zero = [], False
    for value, name in units:
        digit = n // value % 10
        if digit:
            if zero and out:
                out.append("零")
            out.append(("二" if digit == 2 and value != 10 else _DIGITS[digit]) + name)
            zero = False
        elif out:
            zero = True
    text = "".join(out)
    return text[1:] if text.startswith("一十") else text


def pick(rng, table: dict, style: str) -> str:
    return rng.choice(table[style])


def budget_phrase(rng, style: str, budget: int) -> str:
    return pick(rng, BUDGET, style).format(n=budget, cn=to_chinese(budget))
