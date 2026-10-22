"""GRPO 组内优势和 GSPO 序列级比率的机制演示（纯 Python，确定性）。

GSPO（verl v0.6.0 的 policy_loss.loss_mode=gspo）把一条回答所有 token 的对数比率取平均，再取指数，
得到序列级比率，然后按 clip_ratio_low / clip_ratio_high 裁剪。平均会把逐 token 的噪声压小约 √L 倍，
所以 token 级 PPO 常用的 ±0.2 在序列级几乎永远不触发；GSPO 论文用 3e-4 / 4e-4。教程 08-2 给 GSPO
推荐 0.1–0.2，是把 token 级的经验搬到了序列级。
"""
from __future__ import annotations

import math
import random
import statistics


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """同一任务 G 条轨迹的组内标准化优势（verl 的 grpo 估计器口径：样本标准差）。"""
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean, std = statistics.fmean(rewards), statistics.stdev(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


def clip_fractions(length: int, noise: float, samples: int = 2000, seed: int = 7,
                   ranges=((0.2, 0.2), (0.1, 0.1), (3e-4, 4e-4))) -> dict:
    """给定每 token 对数比率的噪声，统计 token 级与序列级比率落在裁剪区间外的比例。"""
    rng = random.Random(seed)
    token_out = {r: 0 for r in ranges}
    seq_out = {r: 0 for r in ranges}
    token_total = 0
    seq_logs = []
    for _ in range(samples):
        logs = [rng.gauss(0.0, noise) for _ in range(length)]
        seq_log = sum(logs) / length
        seq_logs.append(seq_log)
        ratio = math.exp(seq_log)
        for low, high in ranges:
            seq_out[(low, high)] += not (1 - low <= ratio <= 1 + high)
        for value in logs[:50]:  # 每条抽 50 个 token 统计 token 级
            token_total += 1
            for low, high in ranges:
                token_out[(low, high)] += not (1 - low <= math.exp(value) <= 1 + high)
    return {
        "length": length, "per_token_noise": noise,
        "sequence_log_ratio_std": round(statistics.pstdev(seq_logs), 6),
        "clipped": {f"{low:g}/{high:g}": {"token_level": round(token_out[(low, high)] / token_total, 4),
                                          "sequence_level": round(seq_out[(low, high)] / samples, 4)}
                    for low, high in ranges},
    }
