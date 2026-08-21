Qwen3-8B × ALFWorld：ReAct / Reflexion / ReflexGrad 复现规格

更新时间：2026-07-29用途：交给 Copilot 实现 AE 项目的三个已发表 baseline 锚点。

0. 先统一名称

这三个数字不能视为同一实验协议下的横向结果：

代码配置名

论文中的名称

Qwen3-8B 结果

真正含义

react_reflact_anchor

ReAct

65.7%（约 88/134）

ReflAct 附录中的 1-shot ReAct，关闭 Qwen thinking，MPO 风格环境

reflexion_only_reflexgrad_v4

Reflexion-only

42.5 ± 2.2%

ReflexGrad v4 的 zero-shot、single-episode 慢过程消融；不是原始 Reflexion

reflexgrad_v4

ReflexGrad

75.4 ± 2.2%

zero-shot、single-episode、15 环境步、TextGrad + stall-triggered Reflexion

original_reflexion

Reflexion

无 Qwen3-8B 数字

原论文的跨 trial 方法：失败后反思、重置环境、最多 12 trials

代码、配置、结果文件和论文表格中都必须使用上述全名。不要把 42.5% 写成未加限定词的 “Reflexion”。

1. 共用环境与记录口径

1.1 Benchmark

环境：ALFWorld 0.3.3。

split：eval_out_of_distribution，也常被称为 valid-unseen。

任务：完整 134 个 episode，覆盖六类任务。

一个 task ID 必须只对应一个固定初始环境；不要在不同方法间悄悄换任务列表。

成功只读环境完成信号，例如 info["won"]；LLM evaluator 不能决定最终 success。

指标：success_rate = solved / 134。

失败包括：达到环境动作上限、环境终止但未完成、解析失败耗尽预算、运行异常。

1.2 Qwen

checkpoint：Qwen/Qwen3-8B，必须把最终解析出的 revision/commit 写进 manifest。

ReAct 的 published-anchor 模式必须关闭 Qwen3 thinking：

chat_template_kwargs = {"enable_thinking": False}

如果推理后端使用别的 API，采用等价开关，并在第一条原始响应中断言没有 <think> 内容。

不要混用 Qwen3-8B-Base、量化版或第三方微调版。

保存 tokenizer、Transformers、vLLM/推理服务器、CUDA 和 ALFWorld 版本。

1.3 每个 episode 必须写 JSONL

至少记录：

{
  "method": "reflexgrad_v4",
  "task_id": "...",
  "task_type": "...",
  "seed": 42,
  "model": "Qwen/Qwen3-8B",
  "model_revision": "...",
  "thinking_enabled": false,
  "max_env_steps": 15,
  "success": true,
  "terminate_reason": "env_success",
  "env_steps": 11,
  "llm_calls_by_role": {
    "actor": 11,
    "evaluator": 11,
    "decomposer": 1,
    "loss": 3,
    "gradient": 3,
    "optimizer": 3,
    "trajectory_analyzer": 1,
    "causal_diagnoser": 1,
    "plan_generator": 1
  },
  "input_tokens": 0,
  "output_tokens": 0,
  "trajectory": []
}

每一步 trajectory 还要保存原始模型输出、解析后的 action、前后 observation、环境 reward/done/won、evaluator score、router 分支、policy、active TODO、cooldown 和所有中间文本。

2. ReAct：复现 65.7% 锚点

2.1 论文能确认的设置

来源：ReflAct v2 Appendix G。

模型：Qwen3-8B。

Qwen thinking / inference-time scaling：关闭。

ALFWorld：相同的 134-task test set。

ICL：1-shot。

每个环境步由模型联合输出可选 Thought 和一个 Action。

Success：环境二值完成信号。

论文只给出 65.7%，没有报告多 seed 方差；这应视为一个确定性单次锚点。

2.2 按论文引用的 MPO 实现还原

ReflAct 明确说明 ALFWorld 基于 MPO 官方实现。MPO 的公开配置给出：

method: react_reflact_anchor
model: Qwen/Qwen3-8B
thinking: false
temperature: 0.0
max_completion_tokens: 512
max_env_steps: 30
num_demos: 1
demo_selection: first_demo_matching_task_type
icl_format: first
num_trials: 1

其中 max_env_steps=30、temperature=0 和 max_completion_tokens=512 来自 MPO 公开配置，不是 ReflAct Appendix G 自己重新列出的参数。因此 manifest 中标注：

protocol_source:
  paper_explicit:
    - model
    - thinking_disabled
    - 1_shot
    - 134_tasks
    - success_rate_65_7
  inherited_from_mpo_repo:
    - max_env_steps_30
    - temperature_0
    - max_completion_tokens_512

2.3 Prompt 与 action parser

不要让 Copilot自己改写 prompt。直接读取以下官方文件：

Instruction：https://github.com/WeiminXiong/MPO/blob/main/prompt/instructions/alfworld_inst.txt

1-shot 轨迹：https://github.com/WeiminXiong/MPO/blob/main/prompt/icl_examples/alfworld_icl.json

Prompt 拼接：https://github.com/WeiminXiong/MPO/blob/main/prompt/templates.py

环境封装和 parser：https://github.com/WeiminXiong/MPO/blob/main/envs/alfworld_env.py

关键行为：

根据当前任务类型选对应 demonstration 列表的第一条。

初始 prompt 是 instruction + 完整 1-shot trajectory + 当前任务。

允许 sparse thought：模型可以输出 Thought 后接 Action，也可以只输出 Action。

parser 使用第一个 Action: 后的全部文本作为动作。

put X in Y 和 put X on Y 规范化为 ALFWorld 的 put X in/on Y。

非法格式返回 parser error，并占用一个环境步。

每个 step 一次 actor LLM call；不额外调用 evaluator、critic 或反思模型。

成功即停止；失败不重试、不跨 episode 保留 memory。

2.4 ReAct 验收

134 个任务全部跑完。

每题至多 30 个环境动作。

每步 actor call 数等于 agent 尝试步数。

prompt 中恰有一条同任务类型 demo。

没有 Reflexion、TextGrad、TODO decomposer 或 LLM evaluator 调用。

结果先按整数报告 solved/134，再报告百分比。

65.7% 对应约 88/134，这是由四舍五入反推；不要把“必须恰好 88”写成论文明确要求。

3. Reflexion-only：复现 42.5 ± 2.2% 锚点

3.1 这不是原始 Reflexion

ReflexGrad v4 Table 2 的 Reflexion-only 是：

Qwen/Qwen3-8B；

zero-shot；

single episode / single trial；

每题最多 15 个环境动作；

134 个任务；

10 个固定 seeds；

只保留 ReflexGrad 的 slow-process verbal diagnosis，禁用 TextGrad fast process；

结果为 42.5 ± 2.2%，其中 ± 是 10 个 seed 成功率的 sample standard deviation。

它没有“失败后重置环境并再跑一整条轨迹”的经典 Reflexion loop。

3.2 配置

method: reflexion_only_reflexgrad_v4
model: Qwen/Qwen3-8B
demos: 0
num_trials: 1
max_env_steps: 15
working_memory_size: 10
slow_window_m: 5
low_progress_threshold: 4
low_rule: score_strictly_less_than_threshold
cooldown_steps: 5
textgrad_enabled: false
reflexion_enabled: true
seeds: [42, 123, 456, 789, 1024, 1337, 2025, 3141, 5926, 7531]

每步仍需：

actor 根据 task、当前 observation、active TODO、当前自然语言 policy 和 working memory 生成 action；

执行 action；

evaluator 根据 (task, o_t, a_t, o_t+1) 输出 0–10 的一个整数；

把 transition 和 score 写入最近 10 条 working memory；

若不在 cooldown，且最近 5 个 score 全部严格 < 4，触发 slow process；

slow process 依次调用 trajectory analyzer、causal diagnoser、plan generator，生成 1–3 个纠正 subgoals；

把 plan 以最高优先级合并进 policy，并进入 5-step cooldown；cooldown 期间不再触发 slow/fast 更新。

slow process 的 prompt 使用 ReflexGrad v4 Appendix E 原文，不要自行缩成一次“请反思”的调用：

https://arxiv.org/html/2511.14584v4#Sx19

3.3 未完全披露之处

论文把 Table 2 描述为“移除一个组件、其余超参数相同”，据此最合理的实现是保留 task decomposer、TODO manager、evaluator、working memory 和 policy merge，只关闭 TextGrad。可是当前 GitHub main 的 --pure_ablation 行为与 v4 论文并不完全透明。

因此实现时：

把 use_task_decomposer 做成显式配置；

published-anchor 默认设为 true，并在结果中披露这是依照“只移除 TextGrad”的论文表述；

额外跑一个小规模 false sanity check，但不能无标注地拿它替换主配置；

若作者发布与 v4 对应的 tag/commit，优先以该 revision 为准。

4. ReflexGrad：复现 75.4 ± 2.2% 锚点

4.1 固定设置

method: reflexgrad_v4
benchmark: alfworld
alfworld_version: 0.3.3
split: eval_out_of_distribution
num_tasks: 134
model: Qwen/Qwen3-8B
demos: 0
num_trials: 1
max_env_steps: 15
gradient_cadence_k: 3
slow_window_m: 5
low_progress_threshold: 4
cooldown_steps: 5
working_memory_size: 10
seeds: [42, 123, 456, 789, 1024, 1337, 2025, 3141, 5926, 7531]

论文说使用后端“默认 deterministic decoding”，但没有给出一个跨后端唯一的 generation config。实现必须把实际 do_sample、temperature、top-p、top-k、max tokens 写入 manifest。

不要因为目标数字是 75.4 而反复换 decoding 参数。先固定一套确定性配置，再按 seed 列表跑。

一个 A100 80GB 是论文报告的 Qwen 实验硬件；这不是算法约束。

4.2 初始化

每个 episode 重置所有状态，不跨任务共享记忆：

Task decomposer 用 1 次 LLM call，将 task 和初始 observation 分成 3–8 个高层、顺序 subgoals。

TODO 状态为 pending/active/done/failed，记录尝试次数、最近 action 和失败原因。

初始化自然语言 policy pi_0。

working memory 为空，容量 10。

score window 为空，cooldown 为 0。

4.3 每步状态机

for t in range(1, 16):
    action = actor(task, observation, active_todo, policy, memory)
    next_observation, env_success = env.step(action)

    score = evaluator(task, observation, action, next_observation)  # int 0..10
    memory.append((observation, action, next_observation, score))
    memory.keep_last(10)

    if env_success:
        break

    if cooldown > 0:
        cooldown -= 1
        route = "cooldown"

    elif len(scores) >= 5 and all(s < 4 for s in scores[-5:]):
        analysis = trajectory_analyzer(last_5_tuples)
        cause = causal_diagnoser(analysis, policy)
        plan = plan_generator(cause, policy)  # 1..3 corrective subgoals
        policy = merge(policy, plan=plan)
        cooldown = 5
        route = "slow"

    elif t % 3 == 0:
        loss = llm_as_loss(policy, last_3_tuples)
        gradient = llm_as_gradient(loss, policy)
        policy = llm_as_optimizer(policy, gradient)
        route = "fast"

    else:
        route = "base"

    observation = next_observation

优先级必须是：

active slow plan > TextGrad gradient > existing base policy

不能把 slow plan 和 gradient 做平均；触发 slow 时，本步不再同时触发 fast。

4.4 各 LLM 角色的输入输出合同

不要在实现中合并这些角色，先按论文拆开记录：

角色

输入

输出

decomposer

task, initial observation

3–8 个高层 TODO

actor

task, observation, active TODO, policy, memory

一个合法 ALFWorld action

evaluator

task, previous obs, action, next obs

仅一个 0–10 整数

loss

policy, 最近 3 个带分数 transition

具体 mismatch 列表

gradient

loss, policy

可执行的局部 policy 修改建议

optimizer

policy, gradient

更新后的完整自然语言 policy

trajectory analyzer

最近 5 个带分数 transition

failed-action 列表

causal diagnoser

analysis, policy

一个具体 broken assumption

plan generator

cause, policy

1–3 个可执行纠正 subgoals

论文的完整模板在 Appendix E：

https://arxiv.org/html/2511.14584v4#Sx19

4.5 统计

对每个 seed 跑完整 134 题：

per_seed_sr = solved_per_seed / 134 * 100
mean = np.mean(per_seed_sr)
sample_std = np.std(per_seed_sr, ddof=1)

报告：

每个 seed 的 solved/134；

mean ± sample std；

每类任务 success；

env steps/task；

LLM calls/task，按角色拆分；

input/output tokens/task；

slow activation rate、fast activation count、evaluator parse failure rate。

论文只展示了三个代表性 Qwen counts：Reflexion-only 54/57/60，ReflexGrad 98/101/104；它们不是完整的 10-seed 顺序表，不能拿来逐 seed 对齐。

4.6 当前仓库不能直接当 v4 实现

截至 2026-07-29，ReflexGrad GitHub main 已出现与 v4 论文不同的控制逻辑，例如：

当前 core 默认 stall_threshold=2，论文为 m=5；

当前 core 默认 low_score_cutoff=5，论文是 score 严格 <4；

当前代码还包含 text/status routing、2-step cooldown 和后续论文注释；

某些 trial 路径使用 55 steps，而 v4 headline 是 15。

仓库：https://github.com/qpiai/reflexgrad论文 v4：https://arxiv.org/html/2511.14584v4

因此：

不要直接 git clone 后运行 main 就声称复现 v4；

优先寻找作者与 arXiv v4 对应的 tag/commit；

找不到时按本文状态机独立实现，并保存所用 commit SHA；

把 repo-based 和 paper-spec 两种结果分开命名。

另一个需要披露的问题：论文 Table 3 报告 ReflexGrad 约 100 calls/task，但 Appendix A 的最小状态机、15 步预算和 Appendix E 列出的调用角色无法自然推导出这个均值。不要人为添加调用去凑 100；完整记录真实调用数，并把差异列为 reproduction gap。

5. 原始 Reflexion：若项目需要“方法本体”，单独实现

原始 Reflexion ALFWorld 实验不是 Qwen3-8B，不能用来对齐 42.5%。它的协议是：

Actor：ReAct；

模型：论文写 GPT-3；

134 个 ALFWorld 环境；

2 条 domain-specific few-shot trajectories；

一个 trial 内若相同 action + 相同 response 重复超过 3 次，或动作数超过 30，则判定应反思；

失败后调用 self-reflection LLM，总结 trajectory 和失败信号；

将反思加入长期 memory，最多保留最近 3 条；

reset 同一环境，从头开始下一个 trial；

最多 12 个连续 trials；

指标是 12 trials 内曾经成功的累计任务数，结果 130/134 = 97.0%。

如需 Qwen 版，应命名为 original_reflexion_qwen3_ours，并明确这是我们的迁移实验，不是 published Qwen anchor。

6. Copilot 可直接执行的任务说明

将下面内容与本文文件路径一起交给 Copilot：

在现有 AE 仓库中实现三个独立的 ALFWorld baseline 配置：react_reflact_anchor、reflexion_only_reflexgrad_v4、reflexgrad_v4。严格以 AE_Qwen3_ALFWorld_Baseline_Reproduction_Spec.md 为规范。

先检查现有环境 adapter、模型 adapter、action parser、logging 和 config 系统，不要另建一套重复框架。不要改 AE 方法本身。

实现顺序：

建立共享 EpisodeRunner，success 只读 ALFWorld 环境信号。

实现 ReAct 的 MPO prompt 加载、按 task type 选第一条 1-shot demo、Qwen thinking-off 和 30-step 单 trial。

实现 ReflexGrad 的 decomposer、TODO、working memory、evaluator、router、TextGrad 三阶段、Reflexion 三阶段、priority merge。

用 feature flags 生成 Reflexion-only；不要复制一套行为可能漂移的代码。

增加逐 episode JSONL、run manifest、resume、异常记录和聚合脚本。

增加单元测试和 2-task smoke test；smoke test 通过前不要发起 134×10 full run。

必须先给我：

计划修改的文件；

从本文每个配置字段到代码字段的 mapping；

所有仍无法从论文确定的选择。

完成后运行 formatter、单测和 smoke test，报告命令、结果、已知偏差。不要为了贴近 65.7/42.5/75.4 调参或筛 seed。

7. 必须有的自动化测试

test_success_comes_only_from_env：evaluator 输出 10 但环境未完成时，success 仍为 false。

test_low_rule_is_strict：窗口 [3,3,3,3,4] 不触发 slow；[3,3,3,3,3] 触发。

test_slow_preempts_fast：第 6 步同时满足 cadence 和 stall 时，只走 slow。

test_cooldown_exactly_five_steps：slow 后连续五步不触发 fast/slow。

test_fast_every_three_steps：无 stall 时，仅 3、6、9、12、15 步可 fast。

test_reflexion_only_disables_textgrad：fast 三阶段调用数恒为 0。

test_memory_keeps_last_ten：第 11 条写入后最早 tuple 被移除。

test_episode_state_reset：policy、TODO、memory、score window、cooldown 不泄漏到下一题。

test_react_uses_one_matching_demo：每类任务使用对应列表的第一条且恰好一条。

test_parser_failure_consumes_budget：非法输出被记录且步数加一。

test_no_qwen_think_trace_in_react_anchor：ReAct anchor 中 Qwen 原生 thinking 被关闭。

test_aggregate_uses_sample_std：标准差使用 ddof=1。

8. 推荐的运行阶段

Phase A：2-task smoke

每类挑 1–2 题；

检查 prompt、parser、env success、调用计数、router 和 resume；

固定输出完整 trace。

Phase B：单 seed 全 134

ReAct：先跑一次完整 134；

Reflexion-only/ReflexGrad：先用 seed 42 跑 134；

若与 published anchor 相差超过 10 个百分点，先审计协议，不要直接调 prompt。

Phase C：10 seeds

只在 Phase B 无明显实现错误后运行 Reflexion-only 和 ReflexGrad 的 10 seeds；

每个 seed 单独输出，不允许只保留最接近论文均值的 runs。

9. 主要来源

ReflAct v2，Qwen3-8B Appendix G 与实验细节：https://arxiv.org/html/2505.15182v2

MPO 官方实现：https://github.com/WeiminXiong/MPO

ReflexGrad v4，Table 2、算法、Appendix E/Q：https://arxiv.org/html/2511.14584v4

ReflexGrad 当前仓库：https://github.com/qpiai/reflexgrad

原始 Reflexion：https://arxiv.org/html/2303.11366

Reflexion 官方仓库：https://github.com/noahshinn024/reflexion

10. 复现判定

复现目标是“协议正确且结果处于合理区间”，不是不停修改直到数字相同：

ReAct：以 65.7% 为单次 published sanity anchor。

Reflexion-only：以 42.5 ± 2.2% 为 ReflexGrad-v4-ablation anchor。

ReflexGrad：以 75.4 ± 2.2% 为 10-seed anchor。

差 0–5pp：通常可接受，记录实现差异。

差 5–10pp：审计 checkpoint revision、chat template、parser、step 定义、prompt 和环境任务列表。

差 >10pp：暂停扩模型，先逐题对 trace、router 和 success 判定。

最终论文主表若要公平比较，应另建统一预算配置；不要把这三个 published-anchor 配置伪装成相同协议。
