"""训练推演生成器：两轮训练（SFT 冷启动 + GSPO）的全部产物，确定性生成。

这是推演，不是实测。真实的部分只有 agentic_rl real 跑出的东西：任务、规则基线、规则示范 SFT 数据、
GSPO 裁剪比例演示，全部来自 eval/verification/agentic-rl-env-20261019/。训练曲线、评测数字、
控制台日志由本文件里显式写出的参数模型生成，逐项写进 PROVENANCE.md。

两轮的故事：
- run-01：SFT 用规则示范冷启动，GSPO 用教程 08-2 的原版奖励，clip 0.2。训练奖励一路涨到 1.0 附近，
  但补强版核验的 held-out 成功率只从 0.10 涨到 0.31。归因有三条：原版奖励看不见陷阱商品（奖励漏洞）；
  clip 0.2 在序列级几乎不触发（等于没 clip）；SFT 示范全来自 A 套说法（分布窄）。
- run-02：换补强版奖励，clip 3e-4/4e-4，SFT 数据加入 C 风格改写（与 A、B 都不重叠）的示范。held-out 0.31 → 0.58。
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
GPU = "4×A800 80GB"
PRICE_PER_GPU_HOUR = 10.0
CARDS = 4


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _logistic(step: int, total: int, start: float, end: float, mid: float, width: float) -> float:
    x = (step / total - mid) / width
    return start + (end - start) / (1 + math.exp(-x))


def _curve(seed: int, total: int, start: float, end: float, mid: float, width: float, noise: float) -> list[float]:
    rng = _rng(seed)
    return [round(_logistic(s, total, start, end, mid, width) + rng.gauss(0, noise), 4) for s in range(1, total + 1)]


RUNS = {
    "run-01": {
        "seed": 101, "steps": 150, "sec_per_step": 610,
        "reward": "tutorial", "clip": (0.2, 0.2), "sft_data": "rule_demos_A",
        "tutorial_reward": (0.62, 1.03, 0.30, 0.10, 0.015),
        "verified_success_train": (0.55, 0.78, 0.40, 0.12, 0.02),
        "trap_rate": (0.30, 0.34, 0.50, 0.20, 0.015),
        "kl": (0.0, 0.09, 0.45, 0.15, 0.004),
        "eval": {"dev_success": 0.86, "heldout_success": 0.31, "heldout_trap": 0.33, "heldout_unneeded_q": 0.42,
                 "dev_ci": (0.77, 0.92), "heldout_ci": (0.23, 0.40)},
        "pg_clipfrac": 0.0,
    },
    "run-02": {
        "seed": 202, "steps": 150, "sec_per_step": 640,
        "reward": "verified", "clip": (3e-4, 4e-4), "sft_data": "rule_demos_A+C_rewrites",
        "tutorial_reward": (0.60, 0.98, 0.35, 0.12, 0.015),
        "verified_success_train": (0.52, 0.88, 0.45, 0.13, 0.02),
        "trap_rate": (0.28, 0.05, 0.40, 0.12, 0.01),
        "kl": (0.0, 0.05, 0.45, 0.15, 0.003),
        "eval": {"dev_success": 0.93, "heldout_success": 0.58, "heldout_trap": 0.06, "heldout_unneeded_q": 0.12,
                 "dev_ci": (0.85, 0.97), "heldout_ci": (0.49, 0.67)},
        # 推演假设：真实训练每步的逐 token 对数比率漂移约 0.007（比 gspo_toy 的压力设定 0.05 小一个量级），
        # 800 token 时序列级标准差约 2.4e-4，落在 [1-3e-4, 1+4e-4] 外的比例约 0.15。见 attribution.md 原因 2。
        "pg_clipfrac": 0.15,
    },
}

BASELINE_HELDOUT = {"careful_rule": 0.10, "qwen3_8b_prompt_only": 0.19, "sft_only_A": 0.24}

# 单因素消融（推演假设，不是任何真实运行）：只改一处、其余同 run-01，说明三处为什么都要改。
# 真跑三组各约 ¥1,068，共约 ¥3,200，超出单轮预算；可先只跑「只换奖励」验证主因。
ABLATIONS = {
    "只换奖励（补强版）": {"heldout": 0.45, "trap": 0.09},
    "只换 clip（3e-4/4e-4）": {"heldout": 0.34, "trap": 0.31},
    "只加 C 改写": {"heldout": 0.38, "trap": 0.30},
}


def _config(name: str, spec: dict) -> str:
    low, high = spec["clip"]
    reward_fn = "agentic_rl.reward.reward_tutorial" if spec["reward"] == "tutorial" else "agentic_rl.reward.reward_verified"
    return "\n".join([
        "#!/usr/bin/env bash",
        f"# {name}：推演配置，没有真实运行。键名对齐 verl v0.6.0（policy_loss.loss_mode=gspo 在 v0.6.0 才注册）。",
        "set -euo pipefail",
        "python3 -m verl.trainer.main_ppo \\",
        "  algorithm.adv_estimator=grpo \\",
        "  algorithm.use_kl_in_reward=False \\",
        f"  data.train_files=agentic_rl/runs/data/{name}_train.parquet \\",
        "  data.train_batch_size=32 data.max_prompt_length=2048 data.max_response_length=6144 \\",
        "  data.return_raw_chat=True \\",
        f"  actor_rollout_ref.model.path=checkpoints/sft-{spec['sft_data']} \\",
        "  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \\",
        "  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \\",
        f"  actor_rollout_ref.actor.clip_ratio_low={low} actor_rollout_ref.actor.clip_ratio_high={high} \\",
        "  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.02 \\",
        "  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\",
        "  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_epochs=2 \\",
        "  actor_rollout_ref.actor.ppo_mini_batch_size=8 \\",
        "  actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async \\",
        "  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \\",
        "  actor_rollout_ref.rollout.multi_turn.enable=True \\",
        "  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15 \\",
        "  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \\",
        "  actor_rollout_ref.rollout.multi_turn.tool_config_path=agentic_rl/runs/config/tools.yaml \\",
        "  actor_rollout_ref.rollout.multi_turn.interaction_config_path=agentic_rl/runs/config/user_sim.yaml \\",
        f"  custom_reward_function.path=agentic_rl/reward.py custom_reward_function.name={reward_fn.rsplit('.', 1)[1]} \\",
        f"  trainer.n_gpus_per_node={CARDS} trainer.nnodes=1 trainer.total_training_steps={spec['steps']} \\",
        "  trainer.test_freq=25 trainer.save_freq=50 trainer.logger='[\"console\",\"wandb\"]' \\",
        f"  trainer.project_name=globex-agentic-rl trainer.experiment_name={name}",
        "",
    ])


def _metrics(name: str, spec: dict) -> list[dict]:
    total, seed = spec["steps"], spec["seed"]
    tut = _curve(seed + 1, total, *spec["tutorial_reward"])
    ver = _curve(seed + 2, total, *spec["verified_success_train"])
    trap = _curve(seed + 3, total, *spec["trap_rate"])
    kl = _curve(seed + 4, total, *spec["kl"])
    rng = _rng(seed + 5)
    rows = []
    for i in range(total):
        rows.append({
            "step": i + 1,
            "critic/score/mean": tut[i] if spec["reward"] == "tutorial" else ver[i],
            "monitor/tutorial_reward_mean": tut[i],
            "monitor/verified_success_mean": max(0.0, min(1.0, ver[i])),
            "monitor/trap_recommend_rate": max(0.0, min(1.0, trap[i])),
            "actor/kl_loss": max(0.0, kl[i]),
            "actor/pg_clipfrac": round(max(0.0, spec["pg_clipfrac"] + rng.gauss(0, 0.01)) if spec["pg_clipfrac"] else 0.0, 4),
            "actor/entropy": round(0.62 - 0.25 * (i / total) + rng.gauss(0, 0.01), 4),
            "response_length/mean": round(780 + 140 * math.sin(i / 25) + rng.gauss(0, 20), 1),
            "env/assistant_turns_mean": round(3.6 - 0.4 * (i / total) + rng.gauss(0, 0.05), 3),
            "timing_s/step": round(spec["sec_per_step"] + rng.gauss(0, 25), 1),
        })
    return rows


def _svg(rows: list[dict], keys: list[str], title: str) -> str:
    width, height, pad = 640, 320, 40
    series = {k: [r[k] for r in rows] for k in keys}
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    span = hi - lo or 1.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             f'<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{pad}" y="20" font-size="14">{title}（推演，不是实测）</text>']
    for idx, (key, values) in enumerate(series.items()):
        pts = " ".join(f"{pad + (width - 2 * pad) * j / (len(values) - 1):.1f},"
                       f"{height - pad - (height - 2 * pad) * (v - lo) / span:.1f}" for j, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{colors[idx % 4]}" stroke-width="1.5" points="{pts}"/>')
        parts.append(f'<text x="{width - 220}" y="{40 + 16 * idx}" font-size="12" fill="{colors[idx % 4]}">{key}</text>')
    parts.append(f'<text x="{pad}" y="{height - 10}" font-size="11">step 1 → {len(rows)}；纵轴 {lo:.2f} → {hi:.2f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _hours(spec: dict) -> float:
    return round(spec["steps"] * spec["sec_per_step"] / 3600, 1)


def _cost(spec: dict) -> int:
    return int(round(_hours(spec) * CARDS * PRICE_PER_GPU_HOUR))


def _eval_md(evidence: dict) -> str:
    real = evidence["baselines"]["heldout"]
    lines = ["# 评测结果（固定协议：dev 80 题选 checkpoint，held-out 120 题只评最终 checkpoint 一次）", "",
             "成功率按补强版奖励核验（推荐商品逐件核对到手价、配送、库存、材质；缺条件先问；复核先查）。", "",
             "| 策略 | dev 成功率 | held-out 成功率 | held-out 95% 区间 | held-out 陷阱推荐率 | held-out 多余追问率 | 来源 |",
             "|---|---|---|---|---|---|---|"]
    lines.append(f"| careful 规则基线 | {evidence['baselines']['dev']['careful']['success']} | {real['careful']['success']} | "
                 f"{real['careful']['success_ci95'][0]}–{real['careful']['success_ci95'][1]} | {real['careful']['trap_rate']} | "
                 f"{real['careful']['unnecessary_question_rate']} | **真实运行** |")
    lines.append(f"| trust_filter 规则基线 | {evidence['baselines']['dev']['trust_filter']['success']} | {real['trust_filter']['success']} | "
                 f"{real['trust_filter']['success_ci95'][0]}–{real['trust_filter']['success_ci95'][1]} | {real['trust_filter']['trap_rate']} | "
                 f"{real['trust_filter']['unnecessary_question_rate']} | **真实运行** |")
    lines.append(f"| Qwen3-8B 只用提示词 | — | {BASELINE_HELDOUT['qwen3_8b_prompt_only']} | — | — | — | 推演 |")
    lines.append(f"| SFT（只用 A 套示范） | — | {BASELINE_HELDOUT['sft_only_A']} | — | — | — | 推演 |")
    for name, spec in RUNS.items():
        e = spec["eval"]
        lines.append(f"| {name} | {e['dev_success']} | {e['heldout_success']} | {e['heldout_ci'][0]}–{e['heldout_ci'][1]} | "
                     f"{e['heldout_trap']} | {e['heldout_unneeded_q']} | 推演 |")
    for name, spec in ABLATIONS.items():
        lines.append(f"| {name}（其余同 run-01） | — | {spec['heldout']} | — | {spec['trap']} | — | 推演（消融假设） |")
    lines += ["", "读法：", "- run-01 的训练奖励涨到 1.0 附近，held-out 只到 0.31：这是「结果一般」的那一轮，归因见 attribution.md。",
              "- run-02 对 run-01 的提升（0.31 → 0.58）区间不重叠；对规则基线的提升来自 held-out 换了说法后的泛化。",
              "- 三行消融是单因素假设，用来说明三处为什么都要改；不是任何真实运行，真实归因要真跑消融（三组约 ¥3,200）。",
              "- 规则基线两行是 `python -m agentic_rl real` 真实跑出来的；其余是推演参数模型的输出。"]
    return "\n".join(lines) + "\n"


ATTRIBUTION = """# run-01 归因：训练奖励很好看，held-out 成功率一般

## 现象
- 训练奖励（教程原版）从 0.62 涨到 1.03，看起来很好。
- 补强版核验的训练集成功率只从 0.55 涨到 0.78；held-out 成功率 0.31（区间 0.23–0.40）。
- 陷阱推荐率（单件价合格、到手价超预算）平滑后从 0.30 走到 0.34，逐步值在 0.28–0.37 之间波动，**没有下降**。
- `actor/pg_clipfrac` 全程为 0。

## 三条原因，每条都有证据
1. **奖励漏洞：原版奖励看不见陷阱商品。** 证据是真实运行：`eval/verification/agentic-rl-env-20261019/baselines/`
   里 trust_filter 推荐了 37% 的陷阱商品，原版奖励仍给 1.01，与补强版的相关系数接近 0。08-2 的「商品相关性」
   用策略自己传的检索参数去查推荐商品，价格上限本来就只比单件价，所以永远查不出到手价超支。策略学到的是
   「把预算填进 price_max_major」，而不是「核对到手价」。
2. **clip 等于没开。** 证据是真实运行的机制演示 `gspo_toy.json`：每 token 对数比率噪声 0.05、回答 800 token 时，
   序列级对数比率的标准差约 0.0018，±0.2 的区间外比例是 0（200、2000 token 时也是 0），所以 `pg_clipfrac` 恒为 0。
   GSPO 把整条回答的对数比率取平均，噪声被压小约 √L 倍；教程把 token 级的 0.1–0.2 搬到了序列级。GSPO 论文用 3e-4 / 4e-4。
3. **SFT 示范分布窄。** 证据是真实运行：careful 规则策略在 dev 上 100%、held-out 上 10%。held-out 换成 B 套说法后，
   规则词典认不出城市名（「寄到上海」解析不出收货地，每题都去追问，问完仍是未指定），也剥不掉 B 套开场白
   （「给我看看收纳包」整句被当成商品词，按标题过滤后一件不剩）；中文数字预算它其实能解析。多余追问率 1.0；
   120 题里 108 题最后没有给出推荐，答对的 12 题全是「本来就没有合格商品、应如实说没有」的场景。SFT 只用了这 319 条 A 套示范，RL 的起点在 held-out 上就弱；
   run-01 在 held-out 上的多余追问率 0.42（推演），说明 RL 没把这个毛病练掉。

## 为什么不是别的原因
- 不是训练没收敛：KL 稳定在 0.09 附近，熵平缓下降，没有崩塌信号。
- 不是评测噪声：dev 0.86 与 held-out 0.31 的差距远超两个区间的宽度。
- 不是模型容量：同样的 8B 在 run-02 里到了 0.58。

## run-02 改了什么（一次只改有证据的三处）
| 改动 | 对应原因 | 代价 |
|---|---|---|
| 奖励换成补强版 `reward_verified`（逐件核对到手价，缺条件必须先问，复核必须先查） | 原因 1 | 奖励更稀疏，前 40 步上升慢 |
| clip 改成 3e-4 / 4e-4 | 原因 2 | 区间很窄，裁剪比例对每步策略漂移非常敏感，见下文 |
| SFT 示范加入 C 风格改写（同样的轨迹，买家消息换成和 A、B 都不重叠的第三套说法） | 原因 3 | 改写要人工审核，示范从 319 条增到约 640 条。改写不用 B 套：B 是评测集的说话方式，泄进训练，held-out 就不再是「没见过」 |

没改的：学习率、KL 系数、rollout 组大小、步数，避免一次改太多说不清是哪一处起作用。三处是一起改的，
0.58 拆不出各自的贡献；`eval/results.md` 末尾三行单因素消融是参数模型的假设，真实归因要真跑消融，
三组约 ¥3,200，可先只跑「只换奖励」一组验证主因。

### 新 clip 的裁剪比例要先核对
- **真实演示的结论**（`gspo_toy.json`）：按演示的压力设定（每 token 噪声 0.05），3e-4 / 4e-4 在序列级会裁掉大部分样本：
  200 token 时 93%，800 token 时 84%，2000 token 时 75%。如果真实训练的漂移有这么大，新 clip 几乎每步都在截断，
  学习会停滞。
- **推演的设定**：run-02 取 `pg_clipfrac` 约 0.15。按正态近似倒推，800 token 时要让 15% 的样本落在区间外，
  序列级标准差约 2.4e-4，也就是每 token 漂移约 0.007，比演示的 0.05 小一个数量级。这是假设，不是实测。
- **真实跑时怎样核对**：看前 10 步的 `actor/pg_clipfrac`。持续高于 0.5，说明区间对当前步长过窄，先把学习率减半；
  接近 0，说明又回到了「等于没开」，检查 `loss_mode` 是否真的是 gspo。
"""


def build(evidence_dir: Path, out: Path) -> None:
    evidence = {"baselines": json.loads((evidence_dir / "baselines" / "summary.json").read_text(encoding="utf-8")),
                "sft": json.loads((evidence_dir / "sft" / "stats.json").read_text(encoding="utf-8")),
                "gspo_toy": json.loads((evidence_dir / "gspo_toy.json").read_text(encoding="utf-8"))}
    for sub in ("config", "logs", "curves", "eval"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    summary = {"model": MODEL, "machine": GPU, "price_per_gpu_hour_cny": PRICE_PER_GPU_HOUR, "runs": {}}
    for name, spec in RUNS.items():
        (out / "config" / f"{name}.sh").write_text(_config(name, spec), encoding="utf-8")
        rows = _metrics(name, spec)
        with open(out / "logs" / f"{name}_metrics.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out / "curves" / f"{name}_reward.svg").write_text(
            _svg(rows, ["critic/score/mean", "monitor/verified_success_mean", "monitor/trap_recommend_rate"], f"{name} 奖励与核验"),
            encoding="utf-8")
        (out / "curves" / f"{name}_stability.svg").write_text(
            _svg(rows, ["actor/kl_loss", "actor/pg_clipfrac", "actor/entropy"], f"{name} 稳定性"), encoding="utf-8")
        summary["runs"][name] = {"steps": spec["steps"], "hours": _hours(spec), "cost_cny": _cost(spec),
                                 "reward_fn": spec["reward"], "clip": spec["clip"], "sft_data": spec["sft_data"],
                                 "final_train_reward": rows[-1]["critic/score/mean"],
                                 "final_trap_rate": rows[-1]["monitor/trap_recommend_rate"], "eval": spec["eval"]}
    (out / "eval" / "results.md").write_text(_eval_md(evidence), encoding="utf-8")
    (out / "attribution.md").write_text(ATTRIBUTION, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "PROVENANCE.md").write_text(_provenance(evidence), encoding="utf-8")
    print(json.dumps({k: {"hours": v["hours"], "cost_cny": v["cost_cny"], "heldout": v["eval"]["heldout_success"]}
                      for k, v in summary["runs"].items()}, ensure_ascii=False))


def _provenance(evidence: dict) -> str:
    sft = evidence["sft"]
    return f"""# 来源说明：哪些是真实运行，哪些是推演

**这是推演，不是实测。** 没有租卡、没有训练模型。面试时的说法是「我做过一次完整的训练推演」。

## 真实运行（`python -m agentic_rl real`，CPU，0 次模型调用）
- 任务：三个切分，train 320 / dev 80 / held-out 120，五类场景。
- 规则基线：四个策略在三个切分上的成功率、陷阱推荐率、两版奖励对照（eval/results.md 里标「真实运行」的行）。
- SFT 示范：careful 策略的训练集轨迹经 08-1 五条红线清洗，{sft['input']} 条留下 {sft['kept']} 条。
- GSPO 裁剪比例演示：`gspo_toy.json`。
- 证据目录：`eval/verification/agentic-rl-env-20261019/`，带源码指纹。

## 推演（`agentic_rl/scenario.py` 里显式的参数模型，确定性）
- 两轮训练的配置（键名对齐 verl v0.6.0，但没有用 verl 跑过）。
- 每步指标 `logs/*_metrics.jsonl`、曲线 `curves/*.svg`：logistic 曲线加固定种子的噪声。
- 评测里 Qwen3-8B 提示词基线、SFT 基线、run-01、run-02 的数字，以及末尾三行单因素消融（`scenario.py` 的 `ABLATIONS`，假设）。
- run-02 的 SFT 改写用与 A、B 都不重叠的 C 套说法（推演设定）；B 套只留评测，防止训练数据泄入评测说话方式。
- 时长与费用：每步秒数是假设，按 ¥{PRICE_PER_GPU_HOUR:.0f}/卡时 × {CARDS} 卡估算。

## 已知差距
- 环境复用线上真实检索工具，但 Agent 循环、模拟用户、奖励是平行模块，不经过线上编排器和 AG-UI。
- token 数是按字数估算的，真实训练要用 Qwen3 的 tokenizer 和对话模板重算 loss mask。
- 推演的数字只用来练习「看曲线、做归因、定下一轮」，不能当效果证据。
"""
