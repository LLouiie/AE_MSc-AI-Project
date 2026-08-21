# AE 控制器只读代码审计

生成时间：2026-07-31。范围：只读审计，未修改任何文件，未提交Git。

## 仓库状态

- 路径：`~/projects/AE3`
- 分支：`feature/intra-episode-ae`
- 最近一次commit：`1d4bb66 feat(ae): add initial stateful affect controller`
- **工作区有未提交修改**（`git status --short`）：`ae/baselines/ae_full.py`、`ae/baselines/react.py`、`ae/baselines/reflexion.py`、`ae/controllers/signals.py`、`ae/controllers/stateful_controller.py`、`ae/controllers/tests/test_ae_controller.py`、`ae/runners/run_alfworld.py`、`alfworld_runs_ae/agents.py`、`alfworld_runs_ae/environment.py`、`hotpotqa_runs/llm.py`、两个slurm脚本，以及一批未追踪的新文件（`ae/controllers/canonical_action.py`及其测试、`ae/baselines/reflexgrad_v4*`系列、`ae/baselines/react_reflact_anchor.py`等）。
- **重要含义**：`ae/runners/runs/`下所有已存在的`ae_full`相关episode日志（`smoke5_ae_full*`、`paired20_ae_full*`）时间戳均为2026-07-25，早于当前未提交的`signals.py`/`stateful_controller.py`/`agents.py`改动。**没有任何一次真实运行是用当前代码状态产生的**——本审计对"是否真的进入正式运行路径"的判断基于静态代码追踪+单元测试执行，不是基于对当前代码的一次端到端观测。这一点本身就是本审计的一项发现（见D节）。

---

## A. 当前真实运行流程

调用链（`--baseline ae_full`）：

```
ae/runners/run_alfworld.py:main()
  -> load_ae_config(args.ae_config)                          [config.py:132, 默认路径 configs/controllers/ae_full.yaml]
  -> ae_full_baseline.run_episode(env, goal, task_type, act_llm, ae_config, ...)   [ae/baselines/ae_full.py:20]
       -> controller = StatefulController(config)             [stateful_controller.py:74]
       -> agent = ALFWorldAgent(act_llm, controller=controller, termination_policy=...)
       -> agent.run(env, goal, task_type, ...)                 [alfworld_runs_ae/agents.py:282]
            -> controller.reset()                              [stateful_controller.py:83]
            -> ob, info = env.reset()                          # 每episode一次，episode内不再reset
            for step in range(MAX_STEPS=50):
                -> _fit_action_prompt(...)                      [agents.py:143]  确定性截断
                -> directive = controller.active_directive()    [stateful_controller.py:375] 拼进prompt
                -> raw_generation = self.llm(prompt)             # 唯一一次真正的actor LLM调用
                -> parsed = parse_agent_output(raw_generation, admissible_before)  [output_parser.py:109]
                -> (重复动作提前终止检查——仅在 termination_policy=="legacy_early_stop" 时生效，见D节)
                -> obs_raw, reward_raw, done_raw, info = env.step([parsed_action])  # parsed_action，不是raw_generation
                -> controller.step(action=parsed_action, observation=..., admissible_before=..., ...)
                     -> signals = signal_extractor.extract(...)   [signals.py:147]
                     -> state = update_state(prev_state, signals, config)  [affect_state.py:66]
                     -> hysteresis.update(state, config)           [affect_state.py:124]
                     -> signature = compute_signature(...)         [affect_state.py:201]
                     -> candidate, reason = _select_intervention(signals, state, signature)  [stateful_controller.py:344]
                     -> edge detection (severity_upgrade / invalid_rising_edge / flag_rising_edge)
                     -> if is_event and not (warmup/cooldown/budget_exhausted): _arm_intervention(...)
                     -> 记录完整step record 到 self.step_log
                -> history.append((thought, action, observation))   # 只存进history，用于下一次_build_trajectory
                -> if done: return trajectory, won
       -> merged_step_log = zip(agent.step_log, controller.step_log)  # 逐步合并
       -> summary = controller.summary(success, env_steps, termination_reason)
  -> logger.write({...result})   # 追加到 episode_log.jsonl
```

关键点：
- **actor LLM每步恰好调用1次**（`agents.py:320`），是唯一真正产生动作的LLM调用。
- **REFLECT/REPLAN/VERIFY目前都不产生额外LLM调用**——它们只是把`intervention_renderer.render_directive()`返回的固定模板文本拼进下一次actor prompt（`agents.py:316-318`），`stateful_controller.py:390`的`summary()`把`intervention_llm_calls`硬编码为`0`并注释"v1: directives are fixed templates...not a separate LLM call"。
- **信号计算不调用LLM，不看admissible_commands之外的任何环境隐藏字段**（`signals.py`模块docstring已明确"No embeddings, no extra LLM evaluator calls"，且`info.keys()`只有`extra.gamefile`/`won`/`admissible_commands`）。
- **`won`/`reward`/`done`只用于日志字段，不参与signal计算或state更新**（`stateful_controller.py:154-156`把它们当参数收下，只在`record`字典里出现，`_select_intervention`/`update_state`都不读它们）。

---

## B. 逐文件审计表

| 文件:行号 | 当前行为 | 是否进入正式运行路径 | 与新设计对照 | 证据 |
|---|---|---|---|---|
| `ae/controllers/affect_state.py:22-38` | `AffectState`四字段dataclass：uncertainty/frustration/surprise/confidence，`clip01`裁剪到[0,1] | **是** | 与新设计四状态完全一致 | `initial_state()`/`update_state()`被`stateful_controller.py:86,175`直接调用 |
| `ae/controllers/affect_state.py:66-90` | `update_state`: 每状态独立执行 `clip(decay*prev + Σweight_i*signal_i, 0, 1)` | **是** | 与新设计"证据累积+旧证据衰减"部分一致；"有效进展后恢复"通过`progress_relief`权重（对uncertainty/frustration为负、对confidence为正）实现，见下 | `_weighted_delta()`的8项求和，`stateful_controller.py:175`每步调用一次 |
| `ae/controllers/affect_state.py:93-175` | `Mode`枚举+`HysteresisTracker`：4条独立enter/exit闭锁，`current_mode()`用固定优先级`_MODE_PRIORITY`折叠成单一Mode | **是，但current_mode()仅用于日志**（`stateful_controller.py:177`注释"logging only"） | 与新设计"仲裁优先级"部分相关，但这是**第二套**优先级（见D节，与`_select_intervention`的优先级不是同一份） | `compute_signature()`才是真正驱动决策的信号，见下 |
| `ae/controllers/affect_state.py:178-208` | `StateSignature`：5个布尔位（不折叠），`compute_signature()`计算 | **是**，`stateful_controller.py:180`每步调用，供`_select_intervention`使用 | 是新设计"多状态同时达到阈值"仲裁的真正输入 | — |
| `ae/controllers/signals.py:143-295` | `SignalExtractor.extract()`：9个信号（含2个未被状态更新使用的），全部由字符串/Jaccard/canonical_action/admissible_commands计算 | **是** | 与新设计"仅用可观测轨迹"完全一致；无embedding、无额外LLM | `stateful_controller.py:162`每步调用 |
| `ae/controllers/signals.py:120-121` `information_gain_proxy`/`local_state_change_proxy` | 计算并写入`StepSignals`/日志 | `local_state_change_proxy`**是**（驱动intervention outcome评估，`stateful_controller.py:185`）；`information_gain_proxy`**否**（`affect_state.py::_weighted_delta`未引用此字段，`config.py::StateWeights`也没有对应权重项） | `information_gain_proxy`是死信号——计算、记录、有独立单元测试（`test_signal_semantics.py:135`），但对四状态和intervention决策**零因果影响** | grep确认`information_gain_proxy`仅出现在`signals.py`自身和其测试文件里 |
| `ae/controllers/canonical_action.py:35-51` | 把`put/take/heat/cool/clean/go/open/close/toggle/look/inventory/think`解析成`(family, obj[, target])`，用于repeated_action的规范化匹配 | **是**（`signals.py:95`的`action_similarity()`调用） | **不认识`move X to Y`**——如果ALFWorld环境实际接受的是`move`而不是`put`（见D节重大风险），这里的`_PATTERNS`没有"move"条目，`canonicalize_action("move mug 1 to fridge 1")`返回`None`，会退化到Jaccard fallback | `_PATTERNS`列表第35-51行逐一核对，无move family |
| `ae/controllers/intervention_renderer.py:12-29` | 只有`VERIFY`/`REFLECT`/`REPLAN`三条固定模板文本；`CONTINUE`和不存在的`DIRECTIVE`/`STOP`没有对应文本 | **是**（`_arm_intervention`后`active_directive()`每步查询） | 与新设计的6类候选（`CONTINUE/DIRECTIVE/VERIFY/REFLECT/REPLAN/STOP`）**不一致**：当前只实现4类，且没有`DIRECTIVE`和`STOP` | `ae/core.py:44-48`的`InterventionType`枚举只有4个成员 |
| `ae/controllers/stateful_controller.py:344-372` `_select_intervention` | 4分支优先级规则树：`frustration_high&confidence_low->REPLAN` > `invalid or (frustration_medium&repeated)->REFLECT` > `uncertainty_high or surprise_high->VERIFY` > `CONTINUE` | **是** | 是新设计"仲裁优先级"的真正实现，但只覆盖4类intervention中的3类主动触发（CONTINUE是默认值） | 逐行读取确认 |
| `ae/controllers/stateful_controller.py:56-71` `_SEVERITY`/`_ESCALATION` | 第三套排序：`CONTINUE<VERIFY<REFLECT<REPLAN`，用于"severity_upgrade"边沿检测和unresolved后的升级路径 | **是**（`stateful_controller.py:182,209`） | 与`_select_intervention`的优先级、`_MODE_PRIORITY`是**三个不同但需保持一致的排序**（见D节） | — |
| `ae/controllers/stateful_controller.py:240-277` 预算/冷却/warmup | `warmup_steps`/`cooldown_steps`/`max_interventions`/`patch_duration_steps`/`recovery_grace_steps`全部来自`self.config.*`，全部真实生效（非硬编码） | **是** | 与新设计的budget机制一致 | `AEConfig`各字段在`config.py:132-187`从yaml读入，`stateful_controller.py`直接引用`self.config.warmup_steps`等，无本地覆盖 |
| `ae/controllers/stateful_controller.py:126-137` `consume_grace_step`/`note_final_exhausted` | 供`alfworld_runs_ae/agents.py`在"legacy_early_stop"策略下调用，用于抑制/放行重复动作提前终止 | **是，但只在`termination_policy=="legacy_early_stop"`时才会被触发**（默认策略`fixed_horizon`下这两个方法从不被调用） | 见D节重大发现——默认策略下ReAct的重复动作终止规则本身就被整体禁用，不是"AE用预算拦截"而是"两边都不终止" | `agents.py:371` `if repeated_action_detected and not fixed_horizon:` |
| `ae/baselines/ae_full.py:20-48` | 每episode新建`StatefulController`，跑完即弃；把`agent.step_log`和`controller.step_log`按索引zip合并 | **是** | 与新设计"仅单episode内可观测轨迹，无跨episode状态"完全一致 | 无任何持久化/pickle/数据库写入 |
| `ae/runners/run_alfworld.py:40` `IMPLEMENTED_BASELINES` | `{"react","reflexion","ae_full"}`真正可跑；`ae_no_hysteresis`等5个在`PLANNED_BASELINES`里，跑了会直接`SystemExit` | **ae_full是**；消融变体**否，纯预留** | — | `run_alfworld.py:78-82` |
| `ae/controllers/config.py` 全部字段 | 见下方专项表 | 见下 | — | — |
| `alfworld_runs_ae/output_parser.py:109-151` | 区分`empty_generation`/`truncated_no_action`/`unrecognised_bare_text`三类parse failure；已标注动作也做`_clean_action_text`去除">"和编号前缀（非语义纠正） | **是**，唯一parser，react/reflexion/ae_full共用 | — | `agents.py:352` 调用 |
| `alfworld_runs_ae/agents.py:356-382` 重复动作提前终止 | 见D节详细分析 | 见D节 | 与新设计"保留原ReAct终止规则"**部分冲突**（默认策略下规则本身不生效） | — |
| `ae/core.py:34-65` `AppraisalState`/`MetaController`/`StepContext`/`InterventionDecision` | 定义了一套完整的替代接口（progress/recovery/cooldown_remaining字段，Protocol类） | **否，纯死代码** | 与实际用的`AffectState`+`dict`返回值是两套不同的数据结构，容易被误认成"现役接口" | grep确认这4个符号只在`ae/core.py`自身出现，无任何其他文件import/实例化 |

### 配置文件逐项核对

| yaml字段 | 是否被`config.py::load_config`读取 | 是否被下游代码实际使用 | 备注 |
|---|---|---|---|
| `mode` | 是(`raw.get("mode",...)` line 186) | **否，未见任何代码分支读取`config.mode`来改变行为** | `AEConfig.mode`字段存在，但`stateful_controller.py`/`ae_full.py`/`run_alfworld.py`都没有`if config.mode == ...`的分支——mode目前只是个未消费的标签 |
| `initial.*`(4项) | 是 | 是（`initial_state()`） | — |
| `decay.*`(4项) | 是 | 是（`update_state()`） | 命名提示：`decay`在这里是**保留系数**（值越大衰减越慢），不是"衰减率"，容易望文生义搞反，纯文档/命名层面的潜在误解，非bug |
| `weights.*`每状态4-3项 | 是(`_weights()`辅助函数) | 是（`_weighted_delta()`） | `StateWeights`的`invalid_action`/`repeated_action`等8个字段名与`StepSignals`字段名同名对齐，一一对应 |
| `hysteresis.*`4组enter/exit | 是(`_band()`) | 是（`HysteresisTracker.update()`） | — |
| `controller.frustration_medium` | 是 | 是（`_select_intervention`第358行，`compute_signature`第204行） | — |
| `controller.warmup_steps/cooldown_steps/max_interventions/patch_duration_steps/recovery_grace_steps` | 是 | 是 | 全部在`stateful_controller.py`直接引用，无硬编码覆盖 |
| `signals.repetition_window` | 是 | 是（`signals.py:176,191,195`） | — |
| `signals.unexpected_similarity_threshold` | 是 | 是（`signals.py:251`，仅在"未知family"分支使用） | — |
| `signals.novelty_progress_threshold` | 是 | 是（`signals.py:263,278`） | — |
| `signals.repetition_trigger_threshold` | 是 | 是（`stateful_controller.py:349`，**不在signals.py内使用**，命名容易让人以为是信号提取参数，实际是controller判定规则的参数） | 见D节命名易混淆 |

**未使用/命名不一致/被硬编码覆盖的参数**：目前只发现`mode`一项被读取但未消费。没有发现yaml里存在但被代码里的硬编码值覆盖的情况——`config.py`里的`AEConfig`默认值与`ae_full.yaml`里的数字目前逐项一致（有意设计成"没传yaml就退化成同样的默认值"），不构成"覆盖"。

---

## C. 新设计—现有实现对照表

| 新设计要素 | 现有实现 | 一致程度 |
|---|---|---|
| 仅单episode可观测轨迹，无经验库/跨题记忆/参数更新/环境reset | `ae_full.py`每episode新建controller，`agents.py`只在episode开始reset一次env，无pickle/db | **一致** |
| 四状态：uncertainty/surprise/frustration/confidence | `AffectState`四字段完全同名 | **一致** |
| 失败证据累积、旧证据衰减、有效进展后恢复 | `update_state`的decay*prev+delta公式；`progress_relief`权重对uncertainty/frustration为负、对confidence为正 | **一致**（公式/权重/阈值本就"暂定"，当前值即"第一版工程常数"，配置文件顶部注释也这么写） |
| 6类候选：CONTINUE/DIRECTIVE/VERIFY/REFLECT/REPLAN/STOP | `InterventionType`只有CONTINUE/VERIFY/REFLECT/REPLAN 4类 | **缺失DIRECTIVE和STOP** |
| DIRECTIVE/VERIFY不额外调用LLM，REFLECT/REPLAN可以增加辅助LLM调用 | 当前VERIFY/REFLECT/REPLAN**全部**只是prompt patch，`intervention_llm_calls`硬编码0，REFLECT/REPLAN目前也不调用额外LLM | **部分冲突**——现状比新设计更保守（少了REFLECT/REPLAN该有的辅助LLM调用能力），需要新增代码路径 |
| 仲裁优先级 | `_select_intervention`的4分支规则树（REPLAN>REFLECT>VERIFY>CONTINUE） | **一致**（规则树形式，非显式数值优先级表，但效果等价） |
| 预算：intervention budget/辅助LLM call budget/replan上限/patch duration/grace-cooldown | `max_interventions`/`patch_duration_steps`/`cooldown_steps`/`recovery_grace_steps`全部实现且真实生效；**没有"辅助LLM call budget"或"replan上限"这两个独立预算**（REPLAN目前算作`max_interventions`里的一种，没有单独计数上限） | **部分实现**——通用intervention预算有，REFLECT/REPLAN专属的辅助LLM调用预算不存在（因为目前它们根本不调用LLM） |
| 重复动作拦截：保留ReAct原终止规则，AE只能用自身预算拦截/恢复，不能取消规则或无上限重试 | 见B/D节：**默认`termination_policy=fixed_horizon`下终止规则本身被整体禁用**（react和ae_full都不再因重复动作提前终止）；只有显式传`--termination-policy legacy_early_stop`时，规则才生效，且此时AE确实是"在终止前用`recovery_grace_steps`预算拦截"，不取消规则，不无上限（`test_ae_controller.py`的G1-G8已覆盖这条路径的正确性） | **默认路径冲突，非默认路径一致**——见E节必须澄清"以后到底跑哪个termination_policy作为AE-full的正式实验设置" |
| 成本日志（LLM调用次数/tokens/环境步数/终止原因/success等） | 环境步数/终止原因/success/intervention计数都有；`intervention_llm_calls`硬编码0；`input_output_tokens`硬编码`None`（"not tracked"）；actor侧真实token用量在`agent.step_log[].usage`里（依赖`AnyOpenAILLM`是否透传`response.usage`，需另查`ae/llm_client.py`） | **部分实现**——环境层面齐全，intervention/辅助角色的token成本目前完全没有真实数据源（现在也没有辅助LLM调用，无从统计） |

---

## D. 风险与歧义

1. **【最高优先级】默认`termination_policy=fixed_horizon`下，ReAct的"重复动作提前终止"规则对react和ae_full都完全不生效**（`agents.py:43` `DEFAULT_TERMINATION_POLICY = "fixed_horizon"`；`agents.py:371` `if repeated_action_detected and not fixed_horizon:`——`fixed_horizon=True`时这整段代码（含grace消耗）都不会执行）。这与新设计"保留原ReAct的重复动作终止规则"的要求**直接冲突**——不是"AE在终止前拦截"，而是"终止规则本身在当前默认配置下不存在"。`consume_grace_step`/`note_final_exhausted`/`recovery_grace_steps`这一整套机制在默认路径下是**完全死代码**（虽然有测试覆盖，测试是直接调用底层API绕过`run_alfworld.py`默认值来验证的，不代表默认CLI跑法会走到这段代码）。**必须先确认新一轮实验打算用哪个`termination_policy`作为正式设置**，再谈"AE如何拦截"。

2. **"repeated_action"在两个文件里有两个不同的定义**：
   - `signals.py::action_similarity`（状态更新用）：规范化family匹配或Jaccard，窗口`repetition_window=3`，连续值[0,1]。
   - `agents.py:363-366`（终止规则用，仅legacy_early_stop下生效）：`parsed_action == last_action`纯字符串相等，只看**上一步**，二元。
   这意味着"重复动作"这个概念在信号层和终止规则层**标准不同**——一个能识别"put X on Y"和"put X in Y"是同一动作，另一个不能。新设计如果要"用统一标准定义重复动作/循环"，这里需要先决定哪个是权威定义。

3. **三套不同的优先级/排序，服务于三个不同目的，容易被混淆**：
   - `affect_state.py:108` `_MODE_PRIORITY`：仅用于`current_mode()`日志展示。
   - `stateful_controller.py:344` `_select_intervention`的if-elif规则树：真正决定intervention。
   - `stateful_controller.py:57-62` `_SEVERITY`字典：驱动"severity_upgrade"边沿检测和`_ESCALATION`升级路径。
   三者目前**效果一致**（都隐含REPLAN>REFLECT>VERIFY的顺序），但分别用不同的数据结构手工维护，未来改动一处而漏改另一处会产生静默不一致，没有测试直接断言"三者的顺序必须一致"。

4. **`information_gain_proxy`是计算了但零因果作用的信号**：出现在`StepSignals`/日志/独立单元测试里，但`affect_state.py::_weighted_delta`和`config.py::StateWeights`都没有引用它，四状态和intervention决策完全不受它影响。如果之后有人看日志以为"information_gain_proxy高的时候confidence应该涨"，会得出错误结论。

5. **`ae/core.py`里的`AppraisalState`/`MetaController`/`StepContext`/`InterventionDecision`是完全未被使用的死代码**——只在自身文件内出现，`stateful_controller.py`实际用的是自己的`AffectState`+裸`dict`，字段也不同（`AppraisalState`有`progress`/`recovery`/`cooldown_remaining`，`AffectState`没有）。如果后续开发者看到`ae/core.py`就以为这是当前接口，会被误导去改一个没人调用的类。

6. **【重大风险，超出controller机制本身，但直接影响一切实验结果的可解释性】AE3的ALFWorld数据（`$ALFWORLD_DATA/json_2.1.1`）的`PutObject`指令模板实测确认为`"move {o} to {r}"`**（本次审计对`alfworld_tasks_suffix.json`前3个gamefile的`game.tw-pddl`内嵌grammar做了只读核实，见下方证据），**而`alfworld_3prompts.json`（`agents.py:16`引用的ICL示例）和`canonical_action.py`的`put`family（第37行）都还是"put X in/on Y"约定**。这与我们在另一项目（ReflAct R1/R2复现）里刚刚定位并修复过的同一个环境/prompt不匹配问题高度相似。`full_react_alfworld_practice`这份已有日志显示100题成功率只有5%——与"绝大多数需要放置动作的任务因为这个原因系统性失败"的假设吻合，但**本次未逐题重新统计"put数量/Nothing happens比例"来做完全确认**（现有日志的trajectory文本格式看起来还是旧版`_build_trajectory`格式，可能产生于更早的代码版本，不能100%代表当前代码——见仓库状态一节），只能说这是一个**有强证据支持、但未做逐题定量确认**的重大风险，会让"AE的干预能不能改善成功率"这个问题在修复这个环境兼容问题之前几乎无法有意义地回答（如果几乎所有需要放置的任务都因为动词不匹配必然失败，那么uncertainty/frustration/confidence的调优空间会被这个更底层的问题淹没）。

7. **REFLECT/REPLAN目前不调用额外LLM，与新设计"可以增加辅助LLM调用"的暂定方向不符**——这不是bug（当前是v1，注释里明确写了"directives are fixed templates...not a separate LLM call"），但如果新一轮设计确实要给REFLECT/REPLAN加辅助调用，`intervention_llm_calls`目前硬编码为0这件事必须先改，否则日志会一直显示0，掩盖新加的调用。

8. **success/done/won只用于日志，不参与信号和状态计算，未发现success oracle泄漏**——这是一个正面结论：`stateful_controller.py:154-156,304-306`把`reward`/`done`/`won`收进来只是为了写进`record`字典（供事后分析/summary用），`_select_intervention`/`update_state`/`SignalExtractor.extract`均不引用这三个参数。同样没有发现任何"偷看admissible_commands之外隐藏环境字段"的代码（`info.keys()`就只有那三项，signals.py模块docstring里也明确核实过这一点）。

---

## E. 最小修改计划（暂不执行，仅列出）

### 必须修改
- `ae/core.py`：`InterventionType`枚举补齐`DIRECTIVE`、`STOP`两个成员；决定`STOP`是否复用现有`env-terminal`/`max_steps`终止路径，还是需要一个新的"controller主动要求提前停止"信号（目前完全没有这个概念——`episode_termination_reason`只由环境侧的`done`/`max_steps`决定，controller从未有能力主动结束episode）。
- `ae/controllers/intervention_renderer.py`：为`DIRECTIVE`增加模板文本（当前只有VERIFY/REFLECT/REPLAN三条）；明确`DIRECTIVE`和现有`VERIFY`在语气/内容上的区别（新设计里两者都"不额外调用LLM"，需要有实际差异，否则退化成同一个东西）。
- `ae/controllers/stateful_controller.py`：
  - 决定并实现`STOP`的触发条件、以及它如何真正结束episode（目前`step()`只返回log record，不返回控制流决定，`agents.py`的主循环也没有检查"controller要求终止"的分支）。
  - 如果REFLECT/REPLAN要允许辅助LLM调用，需要新增一条与`intervention_renderer`平行的"辅助调用"路径，并且给它一个独立的预算字段（新设计提到"辅助 LLM call budget"，目前`AEConfig`里没有这个字段，`max_interventions`不能直接复用，因为它数的是"触发次数"不是"调用次数"）。
  - `summary()`里硬编码的`intervention_llm_calls: 0`需要改成真实计数。
- `alfworld_runs_ae/agents.py`：明确"重复动作提前终止规则"在新一轮实验里到底是不是`fixed_horizon`还是`legacy_early_stop`作为默认——如果要求"保留原ReAct终止规则"，当前`DEFAULT_TERMINATION_POLICY="fixed_horizon"`本身就需要重新审视（这不是AE controller自己的文件，但直接决定Q8的答案，必须先定下来才能改controller）。
- **在改任何controller代码之前，先独立确认第D6条的ALFWorld put/move兼容性风险**（只读检查即可，比如对现有`ae_full.yaml`config跑1-2个smoke episode，统计put动作的Nothing happens比例）——如果这个问题真实存在且未修复，任何controller层面的调优在这之上都是不可靠的基线。

### 建议修改
- `ae/controllers/canonical_action.py`：如果D6的put/move问题确认存在且需要修复，`_PATTERNS`需要加入`move`family（与另一项目`alfworld_action_normalize.py`的归一化逻辑保持概念一致，但这是两个独立代码库，不能直接共享文件，需要各自实现）。
- `ae/controllers/config.py`：给`AEConfig.mode`字段实际接上一个分支（目前读了但没用），或者干脆在文档/注释里明确注明"mode当前只是标签，暂不驱动任何行为分支"，避免后来者误以为改了yaml的mode就能切换行为。
- 统一"重复动作"的定义：要么让`agents.py`的终止检查复用`signals.py::action_similarity`（而不是自己的精确字符串比较），要么在文档里显式记录两者是有意设计成不同粒度的（终止规则用更严格的"完全相同"，信号层面用更宽松的"相似"）。
- 三套优先级排序（`_MODE_PRIORITY`/`_select_intervention`规则树/`_SEVERITY`）建议加一条测试直接断言三者蕴含的顺序互相一致，防止未来只改一处。
- `information_gain_proxy`：要么给它接一个`StateWeights`权重项让它真正参与状态更新，要么在docstring/日志字段说明里更醒目地标注"仅供人工分析，不影响状态"，避免被误读。

### 暂时不要修改
- `ae/controllers/affect_state.py`的四状态更新公式、衰减系数、hysteresis阈值——spec原文已明确"具体公式、权重和阈值尚未冻结"，属于后续调参范畴，不属于这次机制重新冻结要处理的结构性问题。
- `alfworld_runs_ae/legacy/run_practice.py`及其`RulePool`/`consolidate`——已经明确隔离在`legacy/`目录，不进入正式路径，删不删是另一个决定，不影响本次机制审计。
- `ae/baselines/reflexgrad_v4*`系列、`ae/baselines/react_reflact_anchor.py`——这些是未追踪的新文件，服务于不同的（published-anchor）实验线，与AE controller机制无关，不在本次审计范围内，不要顺手改动。

---

## 附：现有测试运行结果（只读执行，未修改任何文件）

```
ae/controllers/tests/test_ae_controller.py       44 passed, 0 failed
ae/controllers/tests/test_signal_semantics.py    11 passed, 0 failed
ae/controllers/tests/test_canonical_action.py    19 passed, 0 failed
ae/controllers/tests/test_intervention_outcome.py 23 passed, 0 failed
```

覆盖情况（Q12）：
- **已覆盖**：四状态bounds/decay/hysteresis无抖动/cooldown/max_interventions/warmup（`test_ae_controller.py`）；unexpected_outcome的家族区分逻辑、information_gain/local_state_change_proxy的独立行为（`test_signal_semantics.py`）；canonical action的介词/编号归一化、repeated_action的规范化匹配（`test_canonical_action.py`）；intervention outcome的recovered/unresolved/escalation/跨episode不泄漏（`test_intervention_outcome.py`）；grace窗口的8个场景（G1-G8，含"react模式完全不受影响"、"grace从不绕过真实环境终止"）。
- **完全没有测试**：
  - `config.py::load_config()`本身——没有一个测试直接加载`configs/controllers/ae_full.yaml`并断言字段被正确解析（现有测试都是直接构造`AEConfig(...)`或用默认值，不经过yaml这条路径）。
  - `canonical_action.py`的`move`family——因为它根本不存在，自然也没有测试。
  - `DIRECTIVE`/`STOP`——因为它们在`InterventionType`里不存在。
  - token/LLM调用成本统计的正确性——`intervention_llm_calls`硬编码为0，没有测试断言"辅助LLM调用发生时这个数字应该增加"（因为现在确实不会增加）。
  - `agents.py`里`termination_policy="fixed_horizon"`路径下"重复动作被记录但从不终止"这条**默认路径**本身——搜索`test_ae_controller.py`发现的G1-G8场景全部是在`legacy_early_stop`策略下测的（`test_recovery_grace_*`），没有一个测试专门断言"默认fixed_horizon策略下，两次完全相同的动作真的不会导致提前终止"这一现状本身（虽然这是从代码逻辑直接可推导的，但没有回归测试保护它不被未来改动打破）。
  - AE3自己的ALFWorld环境put/move兼容性——完全没有测试涉及，风险D6未被任何自动化检查覆盖。
