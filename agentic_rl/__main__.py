"""命令行入口。

python -m agentic_rl real --out eval/verification/agentic-rl-env-20261019
    生成三个切分的任务；在每个切分上跑四个规则策略并用两版奖励打分；用 careful 的训练集轨迹
    按 08-1 红线清洗出规则示范 SFT 数据；跑 GSPO 裁剪比例的机制演示。全部在 CPU 上真实运行，不调模型。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fingerprint() -> dict:
    files = sorted((ROOT / "agentic_rl").glob("*.py")) + [ROOT / "data" / "catalog-v3.jsonl"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + path.read_bytes())
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"sha256": digest.hexdigest(), "files": [str(p.relative_to(ROOT)) for p in files], "git_head": head}


def run_real(out: Path) -> dict:
    from agentic_rl import evaluate, filters, gspo, trajectory
    from agentic_rl.env import SearchBackend, run_many
    from agentic_rl.policies import POLICIES
    from agentic_rl.tasks import generate_all, save_tasks

    out.mkdir(parents=True, exist_ok=True)
    started, timings = time.time(), {}
    tasks = generate_all()
    save_tasks(tasks, out / "tasks")
    timings["tasks_s"] = round(time.time() - started, 2)
    task_summary = {split: {"n": len(items),
                            "scenarios": {s: sum(t.scenario == s for t in items) for s in sorted({t.scenario for t in items})},
                            "feasible_tasks": sum(bool(t.feasible) for t in items),
                            "feasible_with_trap": sum(bool(t.feasible and t.traps) for t in items),
                            "keywords": sorted({t.keyword for t in items}), "style": items[0].style}
                    for split, items in tasks.items()}
    (out / "tasks" / "summary.json").write_text(json.dumps(task_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    backend, results, sft_scored = SearchBackend(), {}, None
    (out / "baselines" / "samples").mkdir(parents=True, exist_ok=True)
    for split, items in tasks.items():
        results[split] = {}
        for name, policy in POLICIES.items():
            begin = time.time()
            scored = evaluate.score(run_many(policy, items, backend, name))
            timings[f"{split}/{name}_s"] = round(time.time() - begin, 2)
            results[split][name] = evaluate.summarize(scored)
            records = [trajectory.to_record(s["episode"], {"verified": s["verified"], "tutorial": s["tutorial"]})
                       for s in scored[:8]]
            trajectory.write_jsonl(records, out / "baselines" / "samples" / f"{name}_{split}.jsonl")
            if split == "train" and name == "careful":
                sft_scored = scored
    (out / "baselines" / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tables = "\n\n".join(evaluate.markdown_table(results[split], split) for split in results)
    (out / "baselines" / "tables.md").write_text(tables + "\n", encoding="utf-8")

    kept, dropped = filters.clean(sft_scored)
    (out / "sft").mkdir(exist_ok=True)
    # M1-3：SFT 导出与 token 统计统一消费 filters.final_messages 的同一版本消息
    trajectory.write_jsonl([trajectory.to_record(k["episode"], {"verified": k["verified"]},
                                                 messages=filters.final_messages(k)) for k in kept],
                           out / "sft" / "rule_demonstrations.jsonl")
    tokens = [trajectory.to_record(k["episode"], messages=filters.final_messages(k))["tokens_total"] for k in kept]
    sft_stats = {"source": "careful 规则策略在训练集上的真实轨迹", "input": len(sft_scored), "kept": len(kept),
                 "dropped": dropped, "kept_by_scenario": {s: sum(k["episode"].task.scenario == s for k in kept)
                                                          for s in sorted({k["episode"].task.scenario for k in kept})},
                 "tokens_total_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
                 "tokens_total_max": max(tokens) if tokens else None}
    (out / "sft" / "stats.json").write_text(json.dumps(sft_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    toy = [gspo.clip_fractions(length, 0.05) for length in (200, 800, 2000)]
    (out / "gspo_toy.json").write_text(json.dumps(toy, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {"command": "python -m agentic_rl real --out " + str(out.relative_to(ROOT)), "source": _fingerprint(),
                "python": sys.version.split()[0], "platform": platform.platform(), "timings": timings,
                "total_s": round(time.time() - started, 1), "model_calls": 0}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"tasks": task_summary, "baselines": results, "sft": sft_stats, "manifest": manifest}


def run_orders(out: Path) -> dict:
    """M2（2026-11-01）：E1 任务准入 + 见证轨迹 + 重放一致性（≥100 对）+ 8 副本并发隔离。CPU、0 模型调用。"""
    import hashlib
    import time as _time
    import zlib

    from agentic_rl import order_tasks
    from agentic_rl.catalog import default_catalog
    from agentic_rl.env import SearchBackend
    from agentic_rl.order_env import OrderCarefulPolicy, OrderEnvironment, run_order_episode, verify_order_episode

    out.mkdir(parents=True, exist_ok=True)
    started = _time.time()
    catalog = default_catalog()
    backend = SearchBackend(catalog)

    records, rejected = order_tasks.generate_order_pool(catalog)
    admission = order_tasks.save_pool(records, rejected, out)

    witness_rows = []
    for item in records:
        task, snapshot = item["task"], item["snapshot"]
        env = run_order_episode(task, snapshot, catalog, seed=zlib.crc32(task.task_id.encode()) % 1000, backend=backend)
        verdict = verify_order_episode(env)
        witness_rows.append({"task_id": task.task_id, "family": task.family, "passed": verdict["passed"],
                             "steps": env.steps, "final_inventory": env.inventory, "orders": env.orders})
    with open(out / "witness.jsonl", "w", encoding="utf-8") as fh:
        for row in witness_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    replay = {"pairs": 0, "consistent": 0, "mismatches": []}
    for item in records[:30]:
        for seed in (0, 1, 2, 3):
            runs = []
            for _ in range(2):
                env = run_order_episode(item["task"], item["snapshot"], catalog, seed=seed, backend=backend)
                digest = hashlib.sha256(
                    json.dumps(env.messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                state = json.dumps({"inventory": env.inventory, "orders": env.orders}, sort_keys=True)
                runs.append((digest, state))
            replay["pairs"] += 1
            if runs[0] == runs[1]:
                replay["consistent"] += 1
            else:
                replay["mismatches"].append({"task_id": item["task"].task_id, "seed": seed})
    (out / "replay_report.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")

    item = records[0]
    envs = [OrderEnvironment(catalog, backend) for _ in range(8)]
    policies = [OrderCarefulPolicy(item["task"]) for _ in range(8)]
    for env in envs:
        env.reset(item["task"], item["snapshot"], seed=7)
    done = [False] * 8
    while not all(done):
        for i, (env, policy) in enumerate(zip(envs, policies)):
            if not done[i]:
                env.step(policy.act(env.messages))
                done[i] = env.terminated or env.truncated
    states = [json.dumps(env.state_for_verifier(), sort_keys=True, default=str) for env in envs]
    concurrency = {"copies": 8, "task_id": item["task"].task_id,
                   "identical_final_state": len(set(states)) == 1,
                   "invariants_clean": all(not env.invariant_checks() for env in envs)}
    (out / "concurrency_report.json").write_text(json.dumps(concurrency, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"admission": admission,
               "witness": {"total": len(witness_rows), "passed": sum(r["passed"] for r in witness_rows)},
               "replay": replay, "concurrency": concurrency}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"command": "python -m agentic_rl orders --out " + str(out.relative_to(ROOT)),
                "source": _fingerprint(), "python": sys.version.split()[0], "platform": platform.platform(),
                "total_s": round(_time.time() - started, 1), "model_calls": 0}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_rl")
    sub = parser.add_subparsers(dest="command", required=True)
    real = sub.add_parser("real", help="真实运行：任务、规则基线、规则示范 SFT 数据、GSPO 机制演示")
    real.add_argument("--out", type=Path, default=ROOT / "eval" / "verification" / "agentic-rl-env-20261019")
    scenario = sub.add_parser("scenario", help="生成训练推演产物（需要先跑 real）")
    scenario.add_argument("--evidence", type=Path, default=ROOT / "eval" / "verification" / "agentic-rl-env-20261019")
    scenario.add_argument("--out", type=Path, default=ROOT / "agentic_rl" / "runs" / "qwen3-8b-gspo-20261020")
    check = sub.add_parser("check", help="核对推演产物之间的数字一致性（改过生成器或叙述后必跑）")
    check.add_argument("--runs", type=Path, default=ROOT / "agentic_rl" / "runs" / "qwen3-8b-gspo-20261020")
    orders = sub.add_parser("orders", help="E1 下单环境：任务准入、见证轨迹、重放与并发验收（CPU，0 模型调用）")
    orders.add_argument("--out", type=Path, default=ROOT / "eval" / "verification" / "agentic-rl-m2-20261101")
    args = parser.parse_args(argv)
    if args.command == "real":
        summary = run_real(args.out.resolve())
        print(json.dumps({"total_s": summary["manifest"]["total_s"], "sft_kept": summary["sft"]["kept"]}, ensure_ascii=False))
    elif args.command == "scenario":
        from agentic_rl.scenario import build
        build(args.evidence.resolve(), args.out.resolve())
    elif args.command == "check":
        from agentic_rl.check_scenario import main as check_main
        return check_main(["--runs", str(args.runs.resolve())])
    elif args.command == "orders":
        summary = run_orders(args.out.resolve())
        print(json.dumps({"admitted": summary["admission"]["admitted"],
                          "replay_consistent": summary["replay"]["consistent"],
                          "replay_pairs": summary["replay"]["pairs"],
                          "concurrency_identical": summary["concurrency"]["identical_final_state"]},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
