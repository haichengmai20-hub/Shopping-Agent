"""任务生成器：带隐藏条件的选购任务，标准答案由独立判定算出。

五类场景对应《项目定位与其他AI须知》列的情形：
- complete：条件都给齐了，继续追问只增加成本。
- missing_slot：缺一个会改变候选集合的条件（配送地或预算），应先问清。
- no_result：没有商品满足条件，应如实说明，不能擅自放宽硬约束。
- changed：买家中途改了条件，要淘汰旧候选，不能沿用过期条件。
- verify：买家要求核对某件商品的当前价格和库存，应先补取证据再回答。

预算按到手价理解。一半的可行任务特意挑能制造陷阱商品的预算（单件价能通过线上价格上限过滤，到手价却超预算），
其余随机取的预算也常常碰上陷阱；2026-10-19 生成的训练集里，288 道可行题有 225 道带陷阱。
训练集和开发集用一组商品关键词和 A 套说法；held-out 换一组关键词和 B 套说法，不是训练句子的同义改写。
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_rl import phrasing
from agentic_rl.catalog import Catalog, default_catalog

TRAIN_KEYWORDS = ("背包", "保温杯", "收纳盒", "扩展坞", "餐盒", "颈枕", "露营")
HELDOUT_KEYWORDS = ("洗护套装", "台灯", "耳机", "笔记本", "收纳包")
SCENARIO_MIX = (("complete", 0.30), ("missing_slot", 0.25), ("no_result", 0.10), ("changed", 0.20), ("verify", 0.15))
SPLITS = {"train": (320, "A", TRAIN_KEYWORDS, 11), "dev": (80, "A", TRAIN_KEYWORDS, 23), "heldout": (120, "B", HELDOUT_KEYWORDS, 37)}


@dataclass
class Task:
    task_id: str
    split: str
    style: str
    scenario: str
    keyword: str
    ship_to: str
    budget: int
    excluded: list = field(default_factory=list)
    initial_query: str = ""
    hidden: dict = field(default_factory=dict)  # 槽位名 → 被问到时的回答
    followup: str = ""  # changed / verify 场景在第一次作答后的买家追问
    change: dict = field(default_factory=dict)  # changed 场景：改了哪个条件、改成什么
    feasible: list = field(default_factory=list)  # 最终条件下满足要求的商品
    traps: list = field(default_factory=list)  # 最终条件下的陷阱商品
    initial_feasible: list = field(default_factory=list)  # changed 场景改条件之前的可行集

    def final_constraints(self) -> dict:
        return {"keyword": self.keyword, "ship_to": self.ship_to, "budget": self.budget, "excluded": list(self.excluded)}


def _round_up(value: float, step: int = 10) -> int:
    return int(math.ceil(value / step) * step)


def _scenarios(n: int, rng: random.Random) -> list[str]:
    counts = [int(round(n * share)) for _, share in SCENARIO_MIX]
    counts[0] += n - sum(counts)
    names = [name for (name, _), count in zip(SCENARIO_MIX, counts) for _ in range(count)]
    rng.shuffle(names)
    return names


def _trap_budget(catalog: Catalog, keyword, dest, excluded, rng) -> int | None:
    """找一个预算：可行集非空，且至少有一件陷阱商品。"""
    options = []
    for pid in catalog.eligible(keyword, dest, excluded):
        offer = catalog.offers[pid]
        low, high = _round_up(offer.unit_price), offer.landed[dest]
        options += [b for b in range(low, int(high), 10) if b < high]
    rng.shuffle(options)
    for budget in options:
        if catalog.feasible(keyword, dest, budget, excluded) and catalog.traps(keyword, dest, budget, excluded):
            return budget
    return None


def _plain_budget(catalog: Catalog, keyword, dest, excluded, rng) -> int:
    landed = sorted(catalog.offers[pid].landed[dest] for pid in catalog.eligible(keyword, dest, excluded))
    anchor = landed[min(len(landed) - 1, rng.randrange(max(1, len(landed) // 2 + 1)))]
    return _round_up(anchor) + rng.choice((0, 10, 20, 40))


def _no_result_budget(catalog: Catalog, keyword, dest, excluded) -> int | None:
    landed = [catalog.offers[pid].landed[dest] for pid in catalog.eligible(keyword, dest, excluded)]
    budget = int(min(landed) // 10 * 10) if landed else 0
    while budget >= 30 and catalog.feasible(keyword, dest, budget, excluded):
        budget -= 10
    return budget if budget >= 30 else None


def _query(rng, style, keyword, dest_text, budget_text, material_text) -> str:
    parts = [phrasing.pick(rng, phrasing.OPENINGS, style) + keyword] + [p for p in (dest_text, budget_text, material_text) if p]
    return "，".join(parts) + "。"


def make_task(catalog: Catalog, split: str, index: int, scenario: str, style: str, keywords, rng: random.Random) -> Task | None:
    keyword = rng.choice(keywords)
    dests = [d for d in phrasing.DESTINATIONS if len(catalog.eligible(keyword, d)) >= 3]
    if not dests:
        return None
    dest = rng.choice(dests)
    excluded = []
    present = sorted({m for pid in catalog.eligible(keyword, dest) for m in catalog.offers[pid].materials} & set(phrasing.MATERIALS))
    if present and rng.random() < 0.3:
        candidate = [rng.choice(present)]
        if len(catalog.eligible(keyword, dest, candidate)) >= 2:
            excluded = candidate
    if scenario == "no_result":
        budget = _no_result_budget(catalog, keyword, dest, excluded)
    elif rng.random() < 0.5:
        budget = _trap_budget(catalog, keyword, dest, excluded, rng) or _plain_budget(catalog, keyword, dest, excluded, rng)
    else:
        budget = _plain_budget(catalog, keyword, dest, excluded, rng)
    if not budget or (scenario != "no_result" and not catalog.feasible(keyword, dest, budget, excluded)):
        return None
    task = Task(task_id=f"{split}-{index:04d}", split=split, style=style, scenario=scenario, keyword=keyword,
                ship_to=dest, budget=budget, excluded=list(excluded))
    dest_text = phrasing.pick(rng, phrasing.DESTINATIONS[dest], style)
    budget_text = phrasing.budget_phrase(rng, style, budget)
    material_text = phrasing.pick(rng, phrasing.MATERIALS[excluded[0]], style) if excluded else ""
    if scenario == "missing_slot":
        slot = rng.choice(("ship_to", "budget"))
        task.hidden = {slot: dest_text if slot == "ship_to" else budget_text}
        dest_text, budget_text = ("", budget_text) if slot == "ship_to" else (dest_text, "")
    task.initial_query = _query(rng, style, keyword, dest_text, budget_text, material_text)
    if scenario == "changed" and not _apply_change(catalog, task, rng):
        return None
    if scenario == "verify":
        task.followup = phrasing.pick(rng, phrasing.VERIFY, style)
    task.feasible = catalog.feasible(task.keyword, task.ship_to, task.budget, task.excluded)
    task.traps = catalog.traps(task.keyword, task.ship_to, task.budget, task.excluded)
    return task


def _apply_change(catalog: Catalog, task: Task, rng: random.Random) -> bool:
    """在第一次作答后改一个条件，并保证旧的可行集里至少有一件在新条件下不再可行。"""
    before = catalog.feasible(task.keyword, task.ship_to, task.budget, task.excluded)
    kinds = ["budget", "material", "dest"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "budget":
            for new in range(task.budget - 10, 20, -10):
                after = catalog.feasible(task.keyword, task.ship_to, new, task.excluded)
                if after and set(before) - set(after):
                    text = rng.choice(phrasing.CHANGE_BUDGET[task.style]).format(n=new, cn=phrasing.to_chinese(new))
                    task.change, task.budget = {"slot": "budget", "from": task.budget, "to": new}, new
                    break
            else:
                continue
        elif kind == "material":
            options = sorted({m for pid in before for m in catalog.offers[pid].materials} & set(phrasing.MATERIALS) - set(task.excluded))
            options = [m for m in options if catalog.feasible(task.keyword, task.ship_to, task.budget, task.excluded + [m])]
            if not options:
                continue
            mat = rng.choice(options)
            text = rng.choice(phrasing.CHANGE_MATERIAL[task.style]).format(mat=phrasing.pick(rng, phrasing.MATERIALS[mat], task.style))
            task.change, task.excluded = {"slot": "excluded", "add": mat}, task.excluded + [mat]
        else:
            options = [d for d in phrasing.DESTINATIONS if d != task.ship_to
                       and catalog.feasible(task.keyword, d, task.budget, task.excluded)
                       and set(before) - set(catalog.feasible(task.keyword, d, task.budget, task.excluded))]
            if not options:
                continue
            dest = rng.choice(options)
            text = rng.choice(phrasing.CHANGE_DEST[task.style]).format(dest=phrasing.pick(rng, phrasing.DESTINATIONS[dest], task.style))
            task.change, task.ship_to = {"slot": "ship_to", "from": task.ship_to, "to": dest}, dest
        task.initial_feasible, task.followup = before, text
        return True
    return False


def generate_split(split: str, catalog: Catalog | None = None) -> list[Task]:
    catalog = catalog or default_catalog()
    size, style, keywords, seed = SPLITS[split]
    rng = random.Random(seed)
    tasks, index = [], 0
    for scenario in _scenarios(size, random.Random(seed + 1)):
        task = None
        while task is None:
            task = make_task(catalog, split, index, scenario, style, keywords, rng)
        tasks.append(task)
        index += 1
    return tasks


def generate_all(catalog: Catalog | None = None) -> dict[str, list[Task]]:
    return {split: generate_split(split, catalog) for split in SPLITS}


def save_tasks(tasks: dict[str, list[Task]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for split, items in tasks.items():
        with open(directory / f"tasks_{split}.jsonl", "w", encoding="utf-8") as fh:
            for task in items:
                fh.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> list[Task]:
    with open(path, encoding="utf-8") as fh:
        return [Task(**json.loads(line)) for line in fh if line.strip()]
