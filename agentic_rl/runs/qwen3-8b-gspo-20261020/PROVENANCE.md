# 来源说明：哪些是真实运行，哪些是推演

**这是推演，不是实测。** 没有租卡、没有训练模型。面试时的说法是「我做过一次完整的训练推演」。

## 真实运行（`python -m agentic_rl real`，CPU，0 次模型调用）
- 任务：三个切分，train 320 / dev 80 / held-out 120，五类场景；`data/manifest.txt` 里的 sha256 是真实文件的。
- 规则基线：四个策略在三个切分上的成功率、陷阱推荐率、两版奖励对照（eval/results.md 里标「真实运行」的行）。
- SFT 示范：careful 策略的训练集轨迹经 08-1 五条红线清洗，320 条留下 319 条。
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
- 时长与费用：每步秒数是假设，按 ¥10/卡时 × 4 卡估算。

## 已知差距
- 环境复用线上真实检索工具，但 Agent 循环、模拟用户、奖励是平行模块，不经过线上编排器和 AG-UI；
  verl 侧的 `verl_tools.py` 适配类还没写。
- token 数是按字数估算的，真实训练要用 Qwen3 的 tokenizer 和对话模板重算 loss mask。
- 曲线重生成需要 matplotlib（`uv pip install matplotlib`）；PNG 已入库，不装也能看。
- 推演的数字只用来练习「看曲线、做归因、定下一轮」，不能当效果证据。
