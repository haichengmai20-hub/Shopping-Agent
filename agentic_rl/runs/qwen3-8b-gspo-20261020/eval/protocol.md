# 评测协议（训练前冻结）

- 切分：dev 80 题（A 套说法）只在 test_freq 的验证和 checkpoint 选择用；held-out 120 题（B 套说法：
  城市名、口语开场白、「塑料」等）只在每轮最终 checkpoint 上评一次，不参与任何选择。
- 解码：评测一律 temperature=0；训练中验证（validation.jsonl）与离线终评（eval/heldout/*.jsonl）都按此。
- 成功判定：补强版奖励 `reward_verified` 的 success——推荐商品逐件核对到手价、配送、库存、材质；
  缺条件的任务必须先问清；要求复核的任务必须先按 ID 精确检索；没有合格商品时如实说明。
  判定需要的标准答案来自任务生成时的独立判定（catalog.violations），不依赖策略自己的检索参数。
- 配对比较：同一任务在两个策略下的成败按任务配对；区间是 Wilson 95%。逐题文件在 eval/dev/、eval/heldout/。
- 边界：dev 与 held-out 差距大是本任务集的预期现象（B 套说法没进训练）；评测代码与奖励代码同一套，
  奖励换版时评测口径不变。
