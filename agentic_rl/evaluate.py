"""评测：同一批任务上比较策略，按场景报告成功率和行为指标，并对照两版奖励。"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from agentic_rl.env import Episode
from agentic_rl.reward import reward_tutorial, reward_verified


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def score(episodes: list[Episode]) -> list[dict]:
    return [{"episode": ep, "verified": reward_verified(ep), "tutorial": reward_tutorial(ep)} for ep in episodes]


def _rate(items, predicate) -> float | None:
    items = list(items)
    return round(sum(1 for x in items if predicate(x)) / len(items), 4) if items else None


def summarize(scored: list[dict]) -> dict:
    n = len(scored)
    successes = sum(s["verified"]["success"] for s in scored)
    by_scenario = defaultdict(list)
    for s in scored:
        by_scenario[s["episode"].task.scenario].append(s)
    feasible = [s for s in scored if s["episode"].task.feasible]
    given = [s for s in scored if s["episode"].task.scenario != "missing_slot"]
    v = [s["verified"]["reward"] for s in scored]
    t = [s["tutorial"]["reward"] for s in scored]
    corr = None
    if n > 2 and statistics.pstdev(v) > 0 and statistics.pstdev(t) > 0:
        corr = round(statistics.correlation(v, t), 4)
    return {
        "n": n,
        "success": round(successes / n, 4) if n else None,
        "success_ci95": wilson(successes, n),
        "verified_reward_mean": round(statistics.fmean(v), 4) if n else None,
        "tutorial_reward_mean": round(statistics.fmean(t), 4) if n else None,
        "reward_correlation": corr,
        "p0_rate": _rate(scored, lambda s: s["verified"]["p0"]),
        "violation_rate": _rate(scored, lambda s: bool(s["verified"]["violations"])),
        "trap_rate": _rate(feasible, lambda s: bool(s["verified"]["trap_hits"])),
        "unnecessary_question_rate": _rate(given, lambda s: s["verified"]["unnecessary_questions"] > 0),
        "missed_clarification_rate": _rate(by_scenario["missing_slot"], lambda s: "missed_clarification" in s["verified"]["reasons"]),
        "no_result_honesty": _rate(by_scenario["no_result"], lambda s: s["verified"]["success"]),
        "verify_compliance": _rate(by_scenario["verify"], lambda s: "skipped_verification" not in s["verified"]["reasons"]),
        "avg_assistant_turns": round(statistics.fmean(s["episode"].assistant_turns for s in scored), 3) if n else None,
        "avg_tool_calls": round(statistics.fmean(len(s["episode"].calls) for s in scored), 3) if n else None,
        "terminations": dict(Counter(s["episode"].termination for s in scored)),
        "failure_reasons": dict(Counter(r for s in scored for r in s["verified"]["reasons"])),
        "by_scenario": {name: round(sum(s["verified"]["success"] for s in items) / len(items), 4)
                        for name, items in sorted(by_scenario.items())},
    }


def markdown_table(results: dict, split: str) -> str:
    """results: 策略名 → summarize() 的结果。"""
    cols = [("成功率", "success"), ("95% 区间", "success_ci95"), ("补强版奖励", "verified_reward_mean"),
            ("原版奖励", "tutorial_reward_mean"), ("两版相关", "reward_correlation"), ("陷阱推荐率", "trap_rate"),
            ("多余追问率", "unnecessary_question_rate"), ("漏问率", "missed_clarification_rate"),
            ("平均助手轮数", "avg_assistant_turns")]
    lines = [f"| 策略（{split}） | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, summary in results.items():
        cells = []
        for _, key in cols:
            value = summary.get(key)
            cells.append(f"{value[0]:.3f}–{value[1]:.3f}" if isinstance(value, tuple) or isinstance(value, list) else ("—" if value is None else f"{value}"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
