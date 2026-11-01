# 两轮 GSPO 训练推演（Qwen3-8B，澄清/检索/补证任务）

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
