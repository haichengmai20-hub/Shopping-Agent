"""核对推演产物之间的数字一致性（Code-R1 的 check_scenario_docs 对应物）。

检查：summary 与 metrics 末行一致；validation 末行等于 dev 成功率；checkpoint 步数、最后迭代号与
时间戳推算（启动时间 + 步数 × 每步秒数）；resolved.yaml 关键取值与参数模型一致；控制台首行与
启动脚本一致；wandb 目录名时间戳、summary 与 metrics 末行一致；逐题文件计数（含两个推演基线）
与汇总比例一致；tools.yaml 的 tool_schema 与 env.py 的 TOOL_SCHEMAS 一致；data/manifest.txt 的
sha256 与真实证据文件相符；results.md / attribution.md / 06 文档里的关键数字存在；
rollouts 可解压且字段齐全；曲线文件齐全。
用法：python -m agentic_rl check [--runs agentic_rl/runs/qwen3-8b-gspo-20261020]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path

from agentic_rl.scenario import BASELINE_HELDOUT, CURVES, RUN_START, RUNS, SAVE_FREQ, TEST_FREQ

EXPECTED_06 = ["0.31", "0.58", "0.15"]  # 06 与归因里必须能对上的关键数字


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(runs_dir: Path, repo: Path) -> list[str]:
    failures = []
    summary = json.loads((runs_dir / "summary.json").read_text(encoding="utf-8"))

    def need(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for name, spec in RUNS.items():
        rows = _read_jsonl(runs_dir / "logs" / name / "metrics.jsonl")
        validation = _read_jsonl(runs_dir / "logs" / name / "validation.jsonl")
        run_summary = summary["runs"][name]
        need(run_summary["steps"] == len(rows) == spec["steps"], f"{name}: metrics 行数 {len(rows)} != {spec['steps']}")
        need(run_summary["final_train_reward"] == rows[-1]["critic/score/mean"], f"{name}: summary 末步奖励不一致")
        need(run_summary["final_trap_rate"] == rows[-1]["monitor/trap_recommend_rate"], f"{name}: summary 末步陷阱率不一致")
        need(run_summary["final_dev_success"] == validation[-1]["val/reward/mean"], f"{name}: summary dev 与 validation 不一致")
        need(validation[-1]["val/reward/mean"] == spec["eval"]["dev_success"], f"{name}: validation 末行 != dev_success")
        need([row["step"] for row in validation] == list(range(0, spec["steps"] + 1, TEST_FREQ)),
             f"{name}: validation 步数不是 0..{spec['steps']} 间隔 {TEST_FREQ}")

        manifest = (runs_dir / "checkpoints" / name / "MANIFEST.txt").read_text(encoding="utf-8")
        for step in range(SAVE_FREQ, spec["steps"] + 1, SAVE_FREQ):
            need(f"global_step_{step}/" in manifest, f"{name}: MANIFEST 缺 global_step_{step}")
            # 落盘分钟必须等于「启动时间 + 步数 × 每步秒数」的推算值
            when = (RUN_START[name] + timedelta(seconds=step * spec["sec_per_step"])).strftime("%Y-%m-%d %H:%M")
            need(f"\t{when}\t/data/ckpt/{name}/global_step_{step}/actor/model_world_size_4_rank_0.pt" in manifest,
                 f"{name}: global_step_{step} 的时间戳与启动时间/每步秒数推算不符（应为 {when}）")
        need((runs_dir / "checkpoints" / name / "latest_checkpointed_iteration.txt").read_text().strip() == str(spec["steps"]),
             f"{name}: latest_checkpointed_iteration != {spec['steps']}")

        # 展开配置的关键取值必须与参数模型一致（clip、lr、n、步数、save/test_freq、路径）
        resolved = (runs_dir / "config" / f"{name}.resolved.yaml").read_text(encoding="utf-8")
        low, high = spec["clip"]
        for token in (f"clip_ratio_low: {low}", f"clip_ratio_high: {high}", "lr: 1.0e-06", "n: 8",
                      f"total_training_steps: {spec['steps']}", f"save_freq: {SAVE_FREQ}", f"test_freq: {TEST_FREQ}",
                      f"/root/models/sft-{spec['sft_data']}", f"/data/ckpt/{name}"):
            need(token in resolved, f"{name}: resolved.yaml 缺或与参数模型不一致的取值 {token!r}")

        script = (runs_dir / "config" / f"{name}.sh").read_text(encoding="utf-8").splitlines()
        command = " ".join(line.strip() for line in script if line.strip().startswith(("python3", "algorithm.", "data.", "actor_", "custom_", "trainer.")))
        console_first = (runs_dir / "logs" / name / "console_head.log").read_text(encoding="utf-8").splitlines()[0]
        need(console_first == "+ " + command, f"{name}: console 首行与启动脚本的命令不一致")
        for row in rows[:5]:
            need(f"step:{row['step']} - " in "\n".join((runs_dir / "logs" / name / "console_head.log").read_text(encoding="utf-8").splitlines()),
                 f"{name}: console 缺 step {row['step']} 的打印行")

        wandb_dirs = sorted((runs_dir / "wandb").glob("offline-run-*"))
        need(len(wandb_dirs) == len(RUNS), f"wandb 目录数 {len(wandb_dirs)} != {len(RUNS)}")
        # 目录名形如 offline-run-<YYYYMMDD_HHMMSS>-<run_id>，时间戳必须等于该轮的启动时间
        run_dirs = {d.name.split("-")[-1]: d for d in wandb_dirs}
        stamp = RUN_START[name].strftime("%Y%m%d_%H%M%S")
        mine = run_dirs.get({"run-01": "g7k2p9qx", "run-02": "m3t8w5rb"}[name])
        need(mine is not None, f"{name}: 找不到对应 run id 的 wandb 目录")
        if mine is not None:
            need(mine.name.startswith(f"offline-run-{stamp}-"),
                 f"{name}: wandb 目录名 {mine.name} 的时间戳与启动时间 {stamp} 不一致")
            wandb_summary = json.loads((mine / "files" / "wandb-summary.json").read_text(encoding="utf-8"))
            for key in ("critic/score/mean", "monitor/trap_recommend_rate", "actor/pg_clipfrac"):
                need(wandb_summary.get(key) == rows[-1][key], f"{name}: wandb summary 的 {key} 与 metrics 末行不一致")
            history = _read_jsonl(mine / "files" / "history.export.jsonl")
            need(len(history) == spec["steps"] and history[-1]["_step"] == spec["steps"],
                 f"{name}: wandb history 行数或末步不一致")

        manifest = (runs_dir / "checkpoints" / name / "MANIFEST.txt").read_text(encoding="utf-8")
        # checkpoint 的时间戳必须等于「启动时间 + 步数 × 每步秒数」推算的落盘分钟
        for step in range(SAVE_FREQ, spec["steps"] + 1, SAVE_FREQ):
            when = (RUN_START[name] + timedelta(seconds=step * spec["sec_per_step"])).strftime("%Y-%m-%d %H:%M")
            need(f"\t{when}\t/data/ckpt/{name}/global_step_{step}/actor/model_world_size_4_rank_0.pt" in manifest,
                 f"{name}: global_step_{step} 的时间戳与启动时间/每步秒数推算不符（应为 {when}）")

        for step_file in sorted((runs_dir / "rollouts" / name).glob("step*.jsonl.gz")):
            samples = [json.loads(line) for line in gzip.open(step_file, "rt", encoding="utf-8")]
            need(samples and all(set(s) >= {"task_id", "reward", "termination", "messages"} for s in samples),
                 f"{name}: {step_file.name} 样本字段不全")

        for slug, _ in CURVES:
            need((runs_dir / "curves" / f"{name}_{slug}.png").exists() or (runs_dir / "curves" / f"{name}_{slug}.svg").exists(),
                 f"{name}: 缺曲线 {slug}")
        need((runs_dir / "curves" / f"{name}_val_dev.png").exists() or (runs_dir / "curves" / f"{name}_val_dev.svg").exists(),
             f"{name}: 缺 dev 验证曲线")

        heldout = _read_jsonl(runs_dir / "eval" / "heldout" / f"{name}.jsonl")
        n = len(heldout)
        e = spec["eval"]
        need(sum(r["success"] for r in heldout) == round(e["heldout_success"] * n), f"{name}: heldout 成功数不一致")
        need(sum(r["trap_hit"] for r in heldout) == round(e["heldout_trap"] * n), f"{name}: heldout 陷阱数不一致")
        need(sum(r["unnecessary_question"] for r in heldout) == round(e["heldout_unneeded_q"] * n),
             f"{name}: heldout 多余追问数不一致")
        need(all(r["trap_hit"] == 0 or r["success"] == 0 for r in heldout),
             f"{name}: 出现「陷阱命中却成功」的逐题记录")
        dev = _read_jsonl(runs_dir / "eval" / "dev" / f"{name}.jsonl")
        need(sum(r["success"] for r in dev) == round(e["dev_success"] * len(dev)), f"{name}: dev 成功数不一致")

    # 两个推演基线的逐题计数（成功数与比例一致、陷阱命中必然失败）
    baseline_rates = {"prompt_only.jsonl": (BASELINE_HELDOUT["qwen3_8b_prompt_only"], 0.30, 0.35),
                      "sft_only_A.jsonl": (BASELINE_HELDOUT["sft_only_A"], 0.28, 0.30)}
    for fname, (succ_rate, trap_rate, uq_rate) in baseline_rates.items():
        rows = _read_jsonl(runs_dir / "eval" / "heldout" / fname)
        n = len(rows)
        need(sum(r["success"] for r in rows) == round(succ_rate * n), f"{fname}: 成功数与 {succ_rate} 不一致")
        need(sum(r["trap_hit"] for r in rows) == round(trap_rate * n), f"{fname}: 陷阱数与 {trap_rate} 不一致")
        need(sum(r["unnecessary_question"] for r in rows) == round(uq_rate * n), f"{fname}: 多余追问数与 {uq_rate} 不一致")
        need(all(r["trap_hit"] == 0 or r["success"] == 0 for r in rows), f"{fname}: 出现「陷阱命中却成功」的记录")

    # tools.yaml 的 tool_schema 必须与训练环境代码 TOOL_SCHEMAS 逐字段一致
    import yaml
    from agentic_rl.env import TOOL_SCHEMAS
    tools_cfg = yaml.safe_load((runs_dir / "config" / "tools.yaml").read_text(encoding="utf-8"))
    declared = [tool["tool_schema"] for tool in tools_cfg["tools"]]
    need(json.loads(json.dumps(declared, ensure_ascii=False)) == json.loads(json.dumps(TOOL_SCHEMAS, ensure_ascii=False)),
         "tools.yaml 的 tool_schema 与 agentic_rl/env.py 的 TOOL_SCHEMAS 不一致")

    # data/manifest.txt 里的 sha256 声称是真实文件的，逐个重算核对
    evidence = repo / "eval" / "verification" / "agentic-rl-env-20261019"
    for line in (runs_dir / "data" / "manifest.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or len(fields[0]) != 64 or not fields[0].isalnum():
            continue
        digest, rel = fields[0], fields[1]
        path = evidence / rel
        need(path.exists(), f"data/manifest.txt 引用的文件不存在：{rel}")
        if path.exists():
            need(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                 f"data/manifest.txt 里 {rel} 的 sha256 与真实文件不符")

    results = (runs_dir / "eval" / "results.md").read_text(encoding="utf-8")
    for name, spec in RUNS.items():
        need(str(spec["eval"]["heldout_success"]) in results, f"results.md 缺 {name} 的 held-out 数字")
    attribution = (runs_dir / "attribution.md").read_text(encoding="utf-8")
    for token in EXPECTED_06:
        need(token in attribution, f"attribution.md 缺关键数字 {token}")
    doc06 = repo / "docs" / "学习路线" / "06_训练推演_澄清检索补证.md"
    if doc06.exists():
        text = doc06.read_text(encoding="utf-8")
        for token in ("0.31", "0.58"):
            need(token in text, f"06 缺关键数字 {token}")
    else:
        failures.append("找不到 06 文档")
    provenance = (runs_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    need("真实运行" in provenance and "推演" in provenance, "PROVENANCE 缺真实/推演分节")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_rl check")
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "agentic_rl" / "runs" / "qwen3-8b-gspo-20261020")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    failures = check(args.runs.resolve(), repo)
    if failures:
        print(f"FAIL {len(failures)} 项：")
        for item in failures:
            print("  -", item)
        return 1
    print("OK：summary / metrics / validation / checkpoints 时间戳 / resolved / console / wandb / rollouts / curves / "
          "逐题计数 / tools.yaml / manifest 哈希 / 文档关键数字全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
