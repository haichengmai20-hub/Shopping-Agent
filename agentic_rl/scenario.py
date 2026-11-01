"""训练推演生成器：两轮训练（SFT 冷启动 + GSPO）的全部产物，确定性生成。

这是推演，不是实测。真实的部分只有 agentic_rl real 跑出的东西：任务、规则基线、规则示范 SFT 数据、
GSPO 裁剪比例演示，全部来自 eval/verification/agentic-rl-env-20261019/。训练曲线、评测数字、
控制台日志、环境清单、checkpoint 清单、wandb 目录由本文件里显式写出的参数模型生成，
哪些真实、哪些合成，逐项写在 PROVENANCE.md。

两轮的故事：
- run-01：SFT 用规则示范冷启动，GSPO 用教程 08-2 的原版奖励，clip 0.2。训练奖励一路涨到 1.0 附近，
  但补强版核验的 held-out 成功率只从 0.10 涨到 0.31。归因有三条：原版奖励看不见陷阱商品（奖励漏洞）；
  clip 0.2 在序列级几乎不触发（等于没 clip）；SFT 示范全来自 A 套说法（分布窄）。
- run-02：换补强版奖励，clip 3e-4/4e-4，SFT 数据加入 C 风格改写（与 A、B 都不重叠）的示范。held-out 0.31 → 0.58。

产物形态对齐 Code-R1 的 runs 目录：env/、config/（含 resolved 与 tools/user_sim）、logs/（指标、
控制台头、validation）、curves/（每指标一张 PNG）、checkpoints/、wandb/、rollouts/、data/、eval/、
README 与 PROVENANCE。数字之间的一致性由 agentic_rl/check_scenario.py 核对。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

MODEL = "Qwen/Qwen3-8B"
# 产物在本仓库的目录名；配置里引用 tools.yaml / user_sim.yaml 时要带上，保证真跑时路径与仓库一致
RUNS_DIR_NAME = "qwen3-8b-gspo-20261020"
GPU = "4×A800 80GB"
PRICE_PER_GPU_HOUR = 10.0
CARDS = 4
NODE = "a800-4x-node12"
SAVE_FREQ, TEST_FREQ = 50, 25
RUN_START = {"run-01": datetime(2026, 10, 20, 8, 30, 12), "run-02": datetime(2026, 10, 21, 12, 0, 1)}
WANDB_RUN_IDS = {"run-01": "g7k2p9qx", "run-02": "m3t8w5rb"}
VAL_START = {"run-01": 0.62, "run-02": 0.70}  # step 0（训练前）的 dev 成功率：SFT 起点

CURVES = [("score", "critic/score/mean"), ("verified_success", "monitor/verified_success_mean"),
          ("trap_rate", "monitor/trap_recommend_rate"), ("kl_loss", "actor/kl_loss"),
          ("pg_clipfrac", "actor/pg_clipfrac"), ("entropy", "actor/entropy"),
          ("response_length", "response_length/mean"), ("assistant_turns", "env/assistant_turns_mean"),
          ("timing_step", "timing_s/step")]


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
                 "dev_ci": (0.84, 0.97), "heldout_ci": (0.49, 0.67)},
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


# ---------------------------------------------------------------- 命令与配置

def _command(name: str, spec: dict) -> str:
    low, high = spec["clip"]
    reward_fn = "reward_tutorial" if spec["reward"] == "tutorial" else "reward_verified"
    return " ".join([
        "python3 -m verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files=/data/{name}/train.parquet",
        "data.train_batch_size=32 data.max_prompt_length=2048 data.max_response_length=6144",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path=/root/models/sft-{spec['sft_data']}",
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        f"actor_rollout_ref.actor.clip_ratio_low={low} actor_rollout_ref.actor.clip_ratio_high={high}",
        "actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.02",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_epochs=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15",
        "actor_rollout_ref.rollout.multi_turn.max_user_turns=4",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path=/root/globex/agentic_rl/runs/{RUNS_DIR_NAME}/config/tools.yaml",
        f"actor_rollout_ref.rollout.multi_turn.interaction_config_path=/root/globex/agentic_rl/runs/{RUNS_DIR_NAME}/config/user_sim.yaml",
        f"custom_reward_function.path=/root/globex/agentic_rl/reward.py custom_reward_function.name={reward_fn}",
        f"trainer.n_gpus_per_node={CARDS} trainer.nnodes=1 trainer.total_training_steps={spec['steps']}",
        f"trainer.test_freq={TEST_FREQ} trainer.save_freq={SAVE_FREQ} trainer.logger=[console,wandb]",
        f"trainer.project_name=globex-agentic-rl trainer.experiment_name={name} trainer.default_local_dir=/data/ckpt/{name}",
    ])


def _config(name: str, spec: dict) -> str:
    low, high = spec["clip"]
    return "\n".join([
        "#!/usr/bin/env bash",
        f"# {name}：推演配置，没有真实运行。键名对齐 verl v0.6.0（policy_loss.loss_mode=gspo 在 v0.6.0 才注册）。",
        "# 节点上的工作目录是 /root/globex（本仓库克隆）；数据与 checkpoint 放 /data。",
        "set -euo pipefail",
        _command(name, spec),
        "",
    ])


def _resolved_yaml(name: str, spec: dict) -> str:
    low, high = spec["clip"]
    reward_fn = "reward_tutorial" if spec["reward"] == "tutorial" else "reward_verified"
    return f"""# {name} 的展开配置（推演）。键名与结构对齐 verl v0.6.0 的 ppo_trainer.yaml 及其子配置；
# 真实来源是节点上 `python -m verl.trainer.main_ppo ... --cfg job --resolve` 的输出，真跑时以它为准。
data:
  tokenizer: null
  use_shm: false
  train_files: /data/{name}/train.parquet
  val_files: /data/{name}/dev.parquet
  prompt_key: prompt
  reward_fn_key: data_source
  max_prompt_length: 2048
  max_response_length: 6144
  train_batch_size: 32
  shuffle: true
  return_raw_chat: true
  truncation: error
actor_rollout_ref:
  hybrid_engine: true
  model:
    path: /root/models/sft-{spec['sft_data']}
    enable_gradient_checkpointing: true
    use_remove_padding: true
  actor:
    strategy: fsdp
    policy_loss:
      loss_mode: gspo
    loss_agg_mode: seq-mean-token-mean
    clip_ratio: 0.2
    clip_ratio_low: {low}
    clip_ratio_high: {high}
    use_kl_loss: true
    kl_loss_coef: 0.02
    kl_loss_type: low_var_kl
    entropy_coeff: 0.0
    ppo_epochs: 2
    ppo_mini_batch_size: 8
    ppo_micro_batch_size_per_gpu: 2
    use_dynamic_bsz: false
    optim:
      lr: 1.0e-06
      lr_warmup_steps_ratio: 0.0
      total_training_steps: {spec['steps']}
      weight_decay: 0.01
  ref:
    strategy: fsdp
    fsdp_config:
      param_offload: true
      optimizer_offload: false
  rollout:
    name: sglang
    mode: async
    dtype: bfloat16
    n: 8
    temperature: 1.0
    top_p: 1.0
    gpu_memory_utilization: 0.45
    tensor_model_parallel_size: 1
    multi_turn:
      enable: true
      max_assistant_turns: 15
      max_user_turns: 4
      tool_config_path: /root/globex/agentic_rl/runs/{RUNS_DIR_NAME}/config/tools.yaml
      interaction_config_path: /root/globex/agentic_rl/runs/{RUNS_DIR_NAME}/config/user_sim.yaml
      use_inference_chat_template: false
      tokenization_sanity_check_mode: strict
      format: hermes
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
custom_reward_function:
  path: /root/globex/agentic_rl/reward.py
  name: {reward_fn}
trainer:
  n_gpus_per_node: {CARDS}
  nnodes: 1
  total_epochs: 15
  total_training_steps: {spec['steps']}
  save_freq: {SAVE_FREQ}
  test_freq: {TEST_FREQ}
  val_before_train: true
  default_local_dir: /data/ckpt/{name}
  logger: [console, wandb]
  project_name: globex-agentic-rl
  experiment_name: {name}
"""


TOOLS_YAML = """# verl 多轮工具声明（推演形态）。真跑前需要按 verl.tools.base_tool.BaseTool 实现
# agentic_rl/verl_tools.py 里的两个适配类（把工具结果包成 execute 的返回契约）；见 PROVENANCE「已知差距」。
# tool_schema 与 agentic_rl/env.py 的 TOOL_SCHEMAS 一致。
tools:
  - class_name: agentic_rl.verl_tools.ProductSearchTool
    config:
      backend: in_memory_catalog_v3
      session_prefix: rl
    tool_schema:
      type: function
      function:
        name: product_search_tool
        description: 检索商品目录；传 ship_to 时商品卡附单件到手价 landed_price。price_max_major 只比较单件商品价。
        parameters:
          type: object
          properties:
            normalized_query: {type: string}
            ship_to: {type: string, enum: [CN, US, JP, SG, EU]}
            price_max_major: {type: number}
            target_currency: {type: string}
            top_k: {type: integer}
            excluded_material_tags: {type: array, items: {type: string}}
            product_id: {type: string}
            sku_id: {type: string}
  - class_name: agentic_rl.verl_tools.ClarifyFormTool
    config:
      store: in_memory
    tool_schema:
      type: function
      function:
        name: show_shopping_form
        description: 向买家展示澄清表单；展示后结束本轮，等待买家提交，不得按默认值继续检索。
        parameters:
          type: object
          required: [title, questions]
          properties:
            title: {type: string}
            questions:
              type: array
              items:
                type: object
                properties:
                  id: {type: string}
                  type: {type: string}
                  label: {type: string}
"""

USER_SIM_YAML = """# verl 多轮的交互方声明（推演形态）：规则化模拟用户，隐藏条件被问到才回答。
# 适配类同样待实跑前实现；行为契约见 agentic_rl/user_sim.py。
interaction:
  - class_name: agentic_rl.verl_tools.SimulatedUserInteraction
    config:
      module: agentic_rl.user_sim
      max_user_turns: 4
      answer_style: per_task
"""


# ---------------------------------------------------------------- 指标、曲线与日志

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


def _validation(name: str, spec: dict) -> list[dict]:
    """每逢 test_freq 在 dev（A 套说法，80 题）上用 temperature=0 评一次；step 0 是训练前。"""
    total = spec["steps"]
    seed = spec["seed"] + 9
    rng = _rng(seed)
    rows = []
    for step in range(0, total + 1, TEST_FREQ):
        value = VAL_START[name] if step == 0 else _logistic(step, total, VAL_START[name], spec["eval"]["dev_success"], 0.45, 0.20)
        if step == total:
            value = spec["eval"]["dev_success"]  # 末行与最终评测一致，check_scenario 核对
        value = round(value + (0 if step in (0, total) else rng.gauss(0, 0.015)), 4)
        rows.append({"step": step, "val/reward/mean": value, "scenario/val_n_pass": round(value * 80),
                     "scenario/val_pass_rate": value})
    return rows


def _png_curves(out: Path, name: str, rows: list[dict], validation: list[dict]) -> list[str]:
    """每指标一张 PNG（推演标注在标题里）。matplotlib 不在时退回 SVG 并返回提示。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        for slug, key in CURVES:
            (out / "curves" / f"{name}_{slug}.svg").write_text(_svg(rows, [key], f"{name} {slug}"), encoding="utf-8")
        return ["matplotlib 未安装，曲线退回 SVG"]
    written = []
    for slug, key in CURVES:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        ax.plot([r["step"] for r in rows], [r[key] for r in rows], linewidth=1.2)
        ax.set_title(f"{name} {key} (rehearsal, not measured)", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "curves" / f"{name}_{slug}.png")
        plt.close(fig)
        written.append(f"{name}_{slug}.png")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.plot([r["step"] for r in validation], [r["val/reward/mean"] for r in validation], marker="o", markersize=3, linewidth=1.2)
    ax.set_title(f"{name} dev success, temperature=0 (rehearsal)", fontsize=9)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "curves" / f"{name}_val_dev.png")
    plt.close(fig)
    written.append(f"{name}_val_dev.png")
    return written


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


def _console_head(name: str, spec: dict, rows: list[dict]) -> str:
    start = RUN_START[name].strftime("%Y-%m-%d %H:%M:%S")
    lines = ["+ " + _command(name, spec),
             f"{start}\tINFO worker.py:1852 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265",
             f"TaskRunner hostname: {NODE}, PID: 48071",
             "initial_seed is set to 42 as `+seed` is null. Please make sure the seed is not used elsewhere.",
             "Max *outstanding* Rollout Requests: 32. To adjust, set `async_dt_worker.max_outstanding_requests`.",
             f"... (配置 pprint 与各 worker 初始化日志从略；完整配置见 ../config/{name}.resolved.yaml) ..."]
    for row in rows[:5] + [rows[-1]]:
        pairs = " - ".join(f"{key}:{value}" for key, value in row.items() if key != "step")
        lines.append(f"step:{row['step']} - {pairs}")
    lines.append(f"... (中间与结尾的 step 行与 logs/{name}/metrics.jsonl 一一对应，从略) ...")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 环境、数据与 checkpoint 清单

VERSIONS = """# 训练节点（推演设定：2026-10-20 08:21 记录，run-01 启动前。真跑时在租到的节点上重录本文件）
host            : {node}（云租用，单节点）
os              : Ubuntu 22.04.4 LTS, kernel 5.15.0-113-generic
cpu             : 2 × Intel Xeon Platinum 8358 (32C/64T each) → 128 逻辑核
memory          : 512 GiB
gpu             : 4 × NVIDIA A800-SXM4-80GB, NVLink 400 GB/s
driver / cuda   : 550.90.07 / 12.4
disk            : /dev/nvme0n1 300G (root, 系统)   /dev/nvme1n1 3.5T (/data：训练数据、rollout、checkpoint)
python          : 3.10.12
torch           : 2.6.0+cu124
sglang          : 0.4.6（rollout 引擎；多轮 async 模式）
transformers    : 4.51.3
flash_attn      : 2.7.4.post1
ray             : 2.43.0
hydra-core      : 1.3.2
wandb           : 0.19.8 (WANDB_MODE=offline)
verl            : v0.6.0 (tag, pip install -e .)，gsupo 走 policy_loss.loss_mode=gspo
model           : Qwen/Qwen3-8B（后训练版，自带工具调用能力），本地目录 /root/models/Qwen3-8B，
                  5 个 safetensors 分片共 16.4 GB；SFT 起点 checkpoint 另存 /root/models/sft-rule_demos_A[C]
本地代码        : /root/globex（Shopping-Agent 克隆）；环境与奖励代码在 agentic_rl/，不改动 app/
""".replace("gsupo", "GSPO")

NVIDIA_SMI = """Tue Oct 20 08:21:40 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI: 550.90.07    Driver Version: 550.90.07    CUDA Version: 12.4                 |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================+
|   0  NVIDIA A800-SXM4-80GB           On  |  00000000:1A:00.0 Off |                    0 |
| N/A   31C    P0             62W / 400W |       1MiB / 81920MiB |      0%      Default |
|                                         |                        |               Disabled |
|   1  NVIDIA A800-SXM4-80GB           On  |  00000000:1B:00.0 Off |                    0 |
| N/A   32C    P0             58W / 400W |       1MiB / 81920MiB |      0%      Default |
|                                         |                        |               Disabled |
|   2  NVIDIA A800-SXM4-80GB           On  |  00000000:1C:00.0 Off |                    0 |
| N/A   31C    P0             60W / 400W |       1MiB / 81920MiB |      0%      Default |
|                                         |                        |               Disabled |
|   3  NVIDIA A800-SXM4-80GB           On  |  00000000:1D:00.0 Off |                    0 |
| N/A   33C    P0             61W / 400W |       1MiB / 81920MiB |      0%      Default |
|                                         |                        |               Disabled |
+-----------------------------------------------------------------------------------------+

No running processes found
"""

PIP_FREEZE = """# 训练节点关键依赖（推演设定；真跑时以 pip freeze 完整输出为准）
accelerate==1.5.0
aiohttp==3.11.13
datasets==3.2.0
deepspeed==0.16.4
flash-attn==2.7.4.post1
hydra-core==1.3.2
numpy==1.26.4
omegaconf==2.3.0
pytorch-lightning==2.4.0
ray==2.43.0
sglang==0.4.6
sentencepiece==0.2.0
tensordict==0.6.2
torch==2.6.0+cu124
transformers==4.51.3
verl @ file:///root/verl
wandb==0.19.8
# -e verl（v0.6.0）；agentic_rl 以仓库路径 import，不单独安装
""".strip()

_HF_FILES = [(726, "huggingface/config.json"), (239, "huggingface/generation_config.json"),
             (11422654, "huggingface/tokenizer.json"), (9732, "huggingface/tokenizer_config.json"),
             (2776833, "huggingface/vocab.json"), (1671853, "huggingface/merges.txt"),
             (613, "huggingface/special_tokens_map.json"), (707, "huggingface/added_tokens.json")]


def _checkpoint_manifest(name: str, spec: dict) -> str:
    start = RUN_START[name]
    lines = [f"# find /data/ckpt/{name} -type f -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' | sort -k3",
             f"# checkpoint 放数据盘（Code-R1 的教训：根盘 300 GB 放两份就满）"]
    total_bytes = 0
    for step in range(SAVE_FREQ, spec["steps"] + 1, SAVE_FREQ):
        when = (start + timedelta(seconds=step * spec["sec_per_step"])).strftime("%Y-%m-%d %H:%M")
        base = f"/data/ckpt/{name}/global_step_{step}/actor"
        entries = []
        for rank in range(CARDS):
            entries += [(8191969927 + 17 * rank, f"{base}/model_world_size_4_rank_{rank}.pt"),
                        (16383816398 + 13 * rank, f"{base}/optim_world_size_4_rank_{rank}.pt"),
                        (15874 + rank, f"{base}/extra_state_world_size_4_rank_{rank}.pt")]
        entries += [(45, f"{base}/fsdp_config.json")] + [(size, f"{base}/{path}") for size, path in _HF_FILES]
        entries += [(2174, f"/data/ckpt/{name}/global_step_{step}/data.pt")]
        total_bytes += sum(size for size, _ in entries)
        lines += [f"{size}\t{when}\t{path}" for size, path in entries]
    lines.append(f"3\t{(start + timedelta(seconds=spec['steps'] * spec['sec_per_step'])).strftime('%Y-%m-%d %H:%M')}\t/data/ckpt/{name}/latest_checkpointed_iteration.txt")
    lines.append(f"# 合计 {total_bytes / 1e9:.1f} GB，每份 checkpoint 约 {total_bytes / 3 / 1e9:.1f} GB（模型 fp32 分片 32.8 GB + 优化器 65.5 GB）")
    return "\n".join(lines) + "\n"


def _data_manifest(evidence_dir: Path, out: Path) -> str:
    lines = ["# 训练数据清单（任务文件是真实生成的，带 sha256；节点侧的 parquet 转换待实跑，不伪造哈希）",
             "# 本地生成：python -m agentic_rl real --out eval/verification/agentic-rl-env-20261019"]
    for path in sorted((evidence_dir / "tasks").glob("tasks_*.jsonl")) + [evidence_dir / "sft" / "rule_demonstrations.jsonl"]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(evidence_dir)}  # {sum(1 for _ in open(path, encoding='utf-8'))} 行")
    lines += ["",
              "# 节点侧（待实跑）：tasks_train.jsonl → train.parquet（列 prompt / data_source / extra_info，",
              "# extra_info 里带任务视图与标准答案，供 reward_verified 在判分端独立核验）；",
              "# run-02 的 SFT 数据 = 319 条 A 套示范 + C 套改写（改写词表真跑前另写，与 A、B 都不重叠）。"]
    return "\n".join(lines) + "\n"


def _probe_log() -> str:
    """真实探针：在本机 CPU 上用 careful 规则策略跑 2 条任务，证明环境链路可执行。"""
    lines = ["# 真实运行（生成推演时在本机执行；python -m agentic_rl scenario 内嵌）",
             "# 目的：证明「策略循环 → 工具（线上真实检索）→ 奖励」链路可执行，与训练推演共用同一套代码。"]
    try:
        from agentic_rl.env import SearchBackend, run_episode
        from agentic_rl.policies import POLICIES
        from agentic_rl.reward import reward_verified
        from agentic_rl.tasks import load_tasks

        tasks = load_tasks(Path(__file__).resolve().parents[1] / "eval" / "verification" / "agentic-rl-env-20261019" / "tasks" / "tasks_dev.jsonl")
        backend = SearchBackend()
        import asyncio

        for task in tasks[:2]:
            episode = asyncio.run(run_episode(POLICIES["careful"], task, backend, "careful"))
            reward = reward_verified(episode)
            lines.append(f"{task.task_id} scenario={task.scenario} turns={episode.assistant_turns} "
                         f"calls={len(episode.calls)} termination={episode.termination} "
                         f"verified_reward={reward['reward']} reasons={reward['reasons']}")
        lines.append("OK: 2/2 episodes completed")
    except Exception as err:  # 探针失败不阻塞生成，但要如实记录
        lines.append(f"PROBE FAILED: {type(err).__name__}: {err}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- wandb、rollouts 与逐题评测

def _wandb(out: Path, name: str, spec: dict, rows: list[dict]) -> None:
    run_dir = out / "wandb" / f"offline-run-{RUN_START[name].strftime('%Y%m%d_%H%M%S')}-{WANDB_RUN_IDS[name]}" / "files"
    run_dir.mkdir(parents=True, exist_ok=True)
    low, high = spec["clip"]
    config = {"algorithm": {"adv_estimator": {"value": "grpo"}},
              "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": {"value": "gspo"}},
                                              "clip_ratio_low": {"value": low}, "clip_ratio_high": {"value": high},
                                              "optim": {"lr": {"value": 1.0e-06}}},
                                    "rollout": {"name": {"value": "sglang"}, "mode": {"value": "async"}, "n": {"value": 8}}},
              "trainer": {"total_training_steps": {"value": spec["steps"]}, "experiment_name": {"value": name}}}
    (run_dir / "config.yaml").write_text(_yaml_dump(config) + "\n", encoding="utf-8")
    summary = dict(rows[-1])
    summary["training/global_step"] = spec["steps"]
    (run_dir / "wandb-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(run_dir / "history.export.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"_step": row["step"], **{k: v for k, v in row.items() if k != "step"}},
                                ensure_ascii=False) + "\n")
    # git.commit 留 null：推演时间线里不存在那个时刻的真实提交，写具体 hash 会自相矛盾（wandb 在无 git
    # 信息的环境里也记 null）；真跑时 wandb 会自动记录当时的 HEAD。
    metadata = {"args": _command(name, spec).split(), "host": NODE, "startedAt": RUN_START[name].isoformat(),
                "executable": "/usr/bin/python3", "cpu_count": 128, "gpu_count": CARDS,
                "git": {"remote": "https://github.com/haichengmai20-hub/Shopping-Agent", "commit": None}}
    (run_dir / "wandb-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _yaml_dump(value, indent: int = 0) -> str:
    """极简 YAML 序列化，够 wandb config 的形态即可。"""
    pad = "  " * indent
    if isinstance(value, dict):
        return "\n".join(f"{pad}{key}:\n{_yaml_dump(item, indent + 1)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(f"{pad}- {item}" for item in value)
    if isinstance(value, dict) is False and isinstance(value, (int, float, str)):
        return f"{pad}{value}"
    return f"{pad}{value}"


def _rollouts(out: Path, name: str, spec: dict, rows: list[dict], train_ids: list[str]) -> None:
    """三个 step 各落一小批样本（合成，格式与 trajectory.to_record 对齐）。"""
    rng = _rng(spec["seed"] + 21)
    for step in (1, 75, spec["steps"]):
        lines = []
        for index, task_id in enumerate(rng.sample(train_ids, 6)):
            reward = max(-1.0, min(1.0, rows[step - 1]["critic/score/mean"] + rng.gauss(0, 0.25)))
            lines.append(json.dumps({
                "task_id": task_id, "sample_id": f"{task_id}#s{index}", "step": step,
                "policy": name, "reward": round(reward, 4), "reward_fn": spec["reward"],
                "termination": "answered", "assistant_turns": rng.randrange(2, 6),
                "tokens_policy_est": rng.randrange(120, 420),
                "messages": "…（该样本的多轮 messages，格式见 trajectory.to_record；此处从略）"}, ensure_ascii=False))
        with gzip.open(out / "rollouts" / name / f"step{step}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def _per_task(path: Path, task_ids: list[str], success_rate: float, trap_rate: float, uq_rate: float, seed: int) -> None:
    """逐题记录（合成）：把聚合比例落成逐题布尔值，计数由 check_scenario 核对。"""
    ids = list(task_ids)
    rng = _rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    k, t, u = round(success_rate * n), round(trap_rate * n), round(uq_rate * n)
    success = set(ids[:k])
    traps = set(ids[k:k + t])  # 陷阱命中必然失败，从失败侧选
    uq = set(rng.sample(ids, u))
    with open(path, "w", encoding="utf-8") as fh:
        for task_id in ids:
            fh.write(json.dumps({"task_id": task_id, "success": int(task_id in success),
                                 "trap_hit": int(task_id in traps), "unnecessary_question": int(task_id in uq)},
                                ensure_ascii=False) + "\n")


def _eval_protocol() -> str:
    return """# 评测协议（训练前冻结）

- 切分：dev 80 题（A 套说法）只在 test_freq 的验证和 checkpoint 选择用；held-out 120 题（B 套说法：
  城市名、口语开场白、「塑料」等）只在每轮最终 checkpoint 上评一次，不参与任何选择。
- 解码：评测一律 temperature=0；训练中验证（validation.jsonl）与离线终评（eval/heldout/*.jsonl）都按此。
- 成功判定：补强版奖励 `reward_verified` 的 success——推荐商品逐件核对到手价、配送、库存、材质；
  缺条件的任务必须先问清；要求复核的任务必须先按 ID 精确检索；没有合格商品时如实说明。
  判定需要的标准答案来自任务生成时的独立判定（catalog.violations），不依赖策略自己的检索参数。
- 配对比较：同一任务在两个策略下的成败按任务配对；区间是 Wilson 95%。逐题文件在 eval/dev/、eval/heldout/。
- 边界：dev 与 held-out 差距大是本任务集的预期现象（B 套说法没进训练）；评测代码与奖励代码同一套，
  奖励换版时评测口径不变。
"""


# ---------------------------------------------------------------- 汇总文本

def _hours(spec: dict) -> float:
    return round(spec["steps"] * spec["sec_per_step"] / 3600, 1)


def _cost(spec: dict) -> int:
    return int(round(_hours(spec) * CARDS * PRICE_PER_GPU_HOUR))


def _eval_md(evidence: dict) -> str:
    real = evidence["baselines"]["heldout"]
    lines = ["# 评测结果（协议见 protocol.md：dev 80 题选 checkpoint，held-out 120 题只评最终 checkpoint 一次）", "",
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
              "- 规则基线两行是 `python -m agentic_rl real` 真实跑出来的；其余是推演参数模型的输出。",
              "- 逐题记录在 dev/、heldout/ 下（合成），计数与上表一致，`python -m agentic_rl check` 核对。"]
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


def _read_ids(evidence_dir: Path, split: str) -> list[str]:
    path = evidence_dir / "tasks" / f"tasks_{split}.jsonl"
    return [json.loads(line)["task_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(evidence_dir: Path, out: Path) -> None:
    evidence = {"baselines": json.loads((evidence_dir / "baselines" / "summary.json").read_text(encoding="utf-8")),
                "sft": json.loads((evidence_dir / "sft" / "stats.json").read_text(encoding="utf-8")),
                "gspo_toy": json.loads((evidence_dir / "gspo_toy.json").read_text(encoding="utf-8"))}
    for sub in ("config", "curves", "eval/dev", "eval/heldout", "data", "env"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "env" / "versions.txt").write_text(VERSIONS.format(node=NODE), encoding="utf-8")
    (out / "env" / "nvidia-smi.txt").write_text(NVIDIA_SMI, encoding="utf-8")
    (out / "env" / "pip_freeze.txt").write_text(PIP_FREEZE + "\n", encoding="utf-8")
    (out / "env" / "probe_policy_env.log").write_text(_probe_log(), encoding="utf-8")
    (out / "data" / "manifest.txt").write_text(_data_manifest(evidence_dir, out), encoding="utf-8")
    (out / "config" / "tools.yaml").write_text(TOOLS_YAML, encoding="utf-8")
    (out / "config" / "user_sim.yaml").write_text(USER_SIM_YAML, encoding="utf-8")
    (out / "eval" / "protocol.md").write_text(_eval_protocol(), encoding="utf-8")

    dev_ids, heldout_ids, train_ids = (_read_ids(evidence_dir, s) for s in ("dev", "heldout", "train"))
    summary = {"model": MODEL, "machine": GPU, "node": NODE, "price_per_gpu_hour_cny": PRICE_PER_GPU_HOUR, "runs": {}}
    curve_notes = []
    for name, spec in RUNS.items():
        run_log = out / "logs" / name
        run_log.mkdir(parents=True, exist_ok=True)
        (out / "rollouts" / name).mkdir(parents=True, exist_ok=True)
        (out / "checkpoints" / name).mkdir(parents=True, exist_ok=True)
        (out / "config" / f"{name}.sh").write_text(_config(name, spec), encoding="utf-8")
        (out / "config" / f"{name}.resolved.yaml").write_text(_resolved_yaml(name, spec), encoding="utf-8")
        rows = _metrics(name, spec)
        validation = _validation(name, spec)
        with open(run_log / "metrics.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(run_log / "validation.jsonl", "w", encoding="utf-8") as fh:
            for row in validation:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (run_log / "console_head.log").write_text(_console_head(name, spec, rows), encoding="utf-8")
        (out / "checkpoints" / name / "MANIFEST.txt").write_text(_checkpoint_manifest(name, spec), encoding="utf-8")
        (out / "checkpoints" / name / "latest_checkpointed_iteration.txt").write_text(f"{spec['steps']}\n", encoding="utf-8")
        _wandb(out, name, spec, rows)
        _rollouts(out, name, spec, rows, train_ids)
        curve_notes += _png_curves(out, name, rows, validation)
        e = spec["eval"]
        _per_task(out / "eval" / "dev" / f"{name}.jsonl", dev_ids, e["dev_success"], 0.0, 0.0, spec["seed"] + 31)
        _per_task(out / "eval" / "heldout" / f"{name}.jsonl", heldout_ids, e["heldout_success"], e["heldout_trap"],
                  e["heldout_unneeded_q"], spec["seed"] + 32)
        summary["runs"][name] = {"steps": spec["steps"], "hours": _hours(spec), "cost_cny": _cost(spec),
                                 "reward_fn": spec["reward"], "clip": spec["clip"], "sft_data": spec["sft_data"],
                                 "final_train_reward": rows[-1]["critic/score/mean"],
                                 "final_trap_rate": rows[-1]["monitor/trap_recommend_rate"],
                                 "final_dev_success": validation[-1]["val/reward/mean"], "eval": spec["eval"]}
    _per_task(out / "eval" / "heldout" / "prompt_only.jsonl", heldout_ids, BASELINE_HELDOUT["qwen3_8b_prompt_only"], 0.30, 0.35, 901)
    _per_task(out / "eval" / "heldout" / "sft_only_A.jsonl", heldout_ids, BASELINE_HELDOUT["sft_only_A"], 0.28, 0.30, 902)
    (out / "eval" / "results.md").write_text(_eval_md(evidence), encoding="utf-8")
    eval_summary = {"protocol": "eval/protocol.md", "n_dev": len(dev_ids), "n_heldout": len(heldout_ids),
                    "real": {"careful": evidence["baselines"]["heldout"]["careful"],
                             "trust_filter": evidence["baselines"]["heldout"]["trust_filter"]},
                    "rehearsal": {"qwen3_8b_prompt_only": BASELINE_HELDOUT["qwen3_8b_prompt_only"],
                                  "sft_only_A": BASELINE_HELDOUT["sft_only_A"],
                                  **{name: spec["eval"] for name, spec in RUNS.items()}},
                    "ablations_assumed": ABLATIONS}
    (out / "eval" / "summary.json").write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "attribution.md").write_text(ATTRIBUTION, encoding="utf-8")
    summary["curve_notes"] = curve_notes
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(_runs_readme(), encoding="utf-8")
    (out / "PROVENANCE.md").write_text(_provenance(evidence), encoding="utf-8")
    print(json.dumps({k: {"hours": v["hours"], "cost_cny": v["cost_cny"], "heldout": v["eval"]["heldout_success"]}
                      for k, v in summary["runs"].items()} | {"curves": len(curve_notes)}, ensure_ascii=False))


def _runs_readme() -> str:
    return """# 两轮 GSPO 训练推演（Qwen3-8B，澄清/检索/补证任务）

**这是推演，不是实测。** 哪些文件来自真实运行、哪些是参数模型合成，逐项见 [PROVENANCE.md](PROVENANCE.md)。

## 目录

| 位置 | 内容 |
|---|---|
| `config/` | 两轮启动脚本与展开配置（键名对齐 verl v0.6.0），`tools.yaml` / `user_sim.yaml` 是多轮工具与模拟用户的声明 |
| `logs/<run>/` | `metrics.jsonl` 每步指标；`validation.jsonl` 每逢 test_freq 的 dev 验证；`console_head.log` 控制台开头 |
| `curves/` | 每指标一张 PNG（标题带「推演」），含 dev 验证曲线 |
| `checkpoints/<run>/` | checkpoint 清单与最后迭代号（真跑时是 `find` 输出） |
| `wandb/` | 每轮一个离线 run 目录：config、逐 step history、summary、metadata |
| `rollouts/<run>/` | 三个 step 各一小批样本（合成，格式说明在文件内） |
| `data/manifest.txt` | 任务与 SFT 数据的真实 sha256 清单；节点侧 parquet 待实跑 |
| `eval/` | 冻结的评测协议、汇总表、`summary.json`、dev 与 held-out 的逐题记录 |
| `env/` | 训练节点环境（推演设定）+ 一份**真实**的本机策略链路探针日志 |
| `attribution.md` | run-01 结果一般的归因与 run-02 的三处改动 |

## 再生成与核对

~~~bash
LLM_API_KEY=placeholder .venv/bin/python -m agentic_rl real       # 真实基线（证据目录）
LLM_API_KEY=placeholder .venv/bin/python -m agentic_rl scenario   # 重建本目录（确定性）
.venv/bin/python -m agentic_rl check                              # 核对叙述、汇总与逐文件数字一致
~~~
"""


def _provenance(evidence: dict) -> str:
    sft = evidence["sft"]
    return f"""# 来源说明：哪些是真实运行，哪些是推演

**这是推演，不是实测。** 没有租卡、没有训练模型。面试时的说法是「我做过一次完整的训练推演」。

## 真实运行（`python -m agentic_rl real`，CPU，0 次模型调用）
- 任务：三个切分，train 320 / dev 80 / held-out 120，五类场景；`data/manifest.txt` 里的 sha256 是真实文件的。
- 规则基线：四个策略在三个切分上的成功率、陷阱推荐率、两版奖励对照（eval/results.md 里标「真实运行」的行）。
- SFT 示范：careful 策略的训练集轨迹经 08-1 五条红线清洗，{sft['input']} 条留下 {sft['kept']} 条。
- GSPO 裁剪比例演示：`eval/verification/agentic-rl-env-20261019/gspo_toy.json`。
- `env/probe_policy_env.log`：生成推演时在本机真实执行的策略链路探针（careful 规则策略跑 2 条任务）。
- 逐题文件里的 task_id 全部来自真实生成的任务文件。
- 证据目录：`eval/verification/agentic-rl-env-20261019/`，带源码指纹。

## 推演（`agentic_rl/scenario.py` 里显式的参数模型，确定性）
- 两轮训练的配置与展开 YAML（键名对齐 verl v0.6.0，但没有用 verl 跑过）；`tools.yaml` / `user_sim.yaml`
  是配置形态，`verl_tools.py` 适配类待实跑前实现。
- 每步指标 `logs/*/metrics.jsonl`、验证 `logs/*/validation.jsonl`、曲线 `curves/*.png`：logistic 曲线加固定种子噪声。
- 控制台 `logs/*/console_head.log`、checkpoint 清单、wandb 目录、rollouts 样本：按真实运行会留下的形态合成。
- 评测里 Qwen3-8B 提示词基线、SFT 基线、run-01、run-02 的数字与逐题布尔值，以及末尾三行单因素消融（`ABLATIONS`，假设）。
- run-02 的 SFT 改写用与 A、B 都不重叠的 C 套说法（推演设定）；B 套只留评测，防止训练数据泄入评测说话方式。
- `env/versions.txt`、`env/nvidia-smi.txt`、`env/pip_freeze.txt`：训练节点的推演设定，真跑时在节点上重录。
- 时长与费用：每步秒数是假设，按 ¥{PRICE_PER_GPU_HOUR:.0f}/卡时 × {CARDS} 卡估算。

## 已知差距
- 环境复用线上真实检索工具，但 Agent 循环、模拟用户、奖励是平行模块，不经过线上编排器和 AG-UI；
  verl 侧的 `verl_tools.py` 适配类还没写。
- token 数是按字数估算的，真实训练要用 Qwen3 的 tokenizer 和对话模板重算 loss mask。
- 曲线重生成需要 matplotlib（`uv pip install matplotlib`）；PNG 已入库，不装也能看。
- 推演的数字只用来练习「看曲线、做归因、定下一轮」，不能当效果证据。
"""
