# REPLAN prompt-v2:进展后剩余窗口措辞修复报告

**未改动正式仓库,未运行GPU,未启动Gate或134题实验,未清理孤儿配置项。** 全部实施+测试在隔离副本
`/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_workdir/`完成。

## 〇、开始前只读核查:5-seed批任务状态

```
[2026-08-06 14:45:30] START version=old seed=46 run_prefix=ae_full_cooldown5_old_seed46
```
仍在运行(旧版REPLAN-unconditional阶段的最后一个种子),正式仓库文件未受影响。

## 一、修改了哪些隔离副本文件

**只有一个文件**:`ae/controllers/intervention_renderer.py`。`stateful_controller.py`和
`alfworld_runs_ae/agents.py`本轮**完全未touch**——本轮要求只改REPLAN措辞的条件渲染逻辑,B+的三个字段
(`last_nonprogress_action_text`/`last_nonprogress_observation_text`/`current_observation_text`)、
`local_state_change_proxy<0.5`门控、字段更新时序全部在`stateful_controller.py`里,本轮不需要也没有碰它。

## 二、精确diff(相对B+版本,隔离出本轮改动)

完整diff文件:`/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_progress_wording_diff_isolated.diff`(86行)

核心部分：

```diff
     current_obs_text = current_observation if current_observation else _NO_CURRENT_OBSERVATION
     ssmc = steps_since_meaningful_change if steps_since_meaningful_change is not None else 0
+
+    if ssmc > 0:
+        status_line = f"No meaningful progress has been made for {ssmc} steps."
+        guidance_line = "Discard the current ineffective strategy and create a different route."
+        plan_bullets = (
+            "- restate the exact goal;\n"
+            "- give a different 2-4-subgoal route using A -> B -> C;\n"
+            "- explain why the next action differs from the failed action."
+        )
+        subgoal_instruction = "the first subgoal"
+        bottom_prohibition = "Do not repeat the failed action."
+    else:
+        status_line = "New progress has been observed since REPLAN was activated."
+        guidance_line = "Continue from the latest observation and preserve the productive route."
+        plan_bullets = (
+            "- restate the exact goal;\n"
+            "- continue the current productive route with the next subgoal using A -> B -> C;\n"
+            "- explain why this next action follows from the latest observation, "
+            "not from the earlier no-progress action."
+        )
+        subgoal_instruction = "the next subgoal"
+        bottom_prohibition = "Do not return to the most recent action with no observed progress."
+
     return (
         "[ACTIVE CONTROL DIRECTIVE: REPLAN]\n"
         "\n"
         f"Original task: {task_description}\n"
-        f"No meaningful progress has been made for {ssmc} steps.\n"
+        f"{status_line}\n"
         f"Most recent action with no observed progress: {action_text}\n"
         f"Environment response to that action: {nonprogress_obs_text}\n"
         f"Current observation: {current_obs_text}\n"
         "\n"
+        f"{guidance_line}\n"
+        "\n"
         "In one concise Thought line:\n"
-        "- restate the exact goal;\n"
-        "- give a different 2-4-subgoal route using A -> B -> C;\n"
-        "- explain why the next action differs from the failed action.\n"
+        f"{plan_bullets}\n"
         "\n"
-        "Then output exactly one executable Action line and execute only the first subgoal.\n"
-        "Do not repeat the failed action.\n"
+        f"Then output exactly one executable Action line and execute only {subgoal_instruction}.\n"
+        f"{bottom_prohibition}\n"
         "Do not assume any state that has not been observed."
     )
```

## 三、两个动态措辞分支的最终完整文本

**分支A(`steps_since_meaningful_change > 0`,与之前一致)**：
```
[ACTIVE CONTROL DIRECTIVE: REPLAN]

Original task: {task_description}
No meaningful progress has been made for {ssmc} steps.
Most recent action with no observed progress: {action_text}
Environment response to that action: {nonprogress_obs_text}
Current observation: {current_obs_text}

Discard the current ineffective strategy and create a different route.

In one concise Thought line:
- restate the exact goal;
- give a different 2-4-subgoal route using A -> B -> C;
- explain why the next action differs from the failed action.

Then output exactly one executable Action line and execute only the first subgoal.
Do not repeat the failed action.
Do not assume any state that has not been observed.
```

**分支B(`steps_since_meaningful_change == 0`,本轮新增)**：
```
[ACTIVE CONTROL DIRECTIVE: REPLAN]

Original task: {task_description}
New progress has been observed since REPLAN was activated.
Most recent action with no observed progress: {action_text}
Environment response to that action: {nonprogress_obs_text}
Current observation: {current_obs_text}

Continue from the latest observation and preserve the productive route.

In one concise Thought line:
- restate the exact goal;
- continue the current productive route with the next subgoal using A -> B -> C;
- explain why this next action follows from the latest observation, not from the earlier no-progress action.

Then output exactly one executable Action line and execute only the next subgoal.
Do not return to the most recent action with no observed progress.
Do not assume any state that has not been observed.
```

**一处需要如实指出的措辞不一致**:分支A底部仍保留`"Do not repeat the failed action."`(沿用v1的措辞,
本轮未改动),而分支A的字段标签早已改成`"Most recent action with no observed progress"`(B+那一轮改的)——
这两处"failed"/"no observed progress"措辞不完全统一,是B+那一轮遗留、本轮范围之外的问题,之前的B+报告里
也提到过同样的不一致(当时判断超出"只改这一处路由判断"的范围,未处理)。本轮严格按你这次给的范围只改
"进展后措辞"这一处,没有顺带处理这个遗留的措辞不一致,如实告知。

## 四、分支判断使用的真实字段及渲染时序

判断条件是`_render_replan()`函数入参`steps_since_meaningful_change`(来自`controller.steps_since_meaningful_change`,`active_directive()`调用时实时读取,B+及更早就已存在的字段,本轮未新增/未改动这个字段本身的计算或门控逻辑)。`if ssmc > 0` / `else`两分支只影响四个局部变量(`status_line`/`guidance_line`/`plan_bullets`/`subgoal_instruction`/`bottom_prohibition`)最终拼进同一个返回模板,`Most recent action with no observed progress`/`Environment response to that action`/`Current observation`三行的取值逻辑(B+那一轮的三个字段)完全不变,两个分支共用。

**REPLAN仍按`patch_duration_steps=3`完整展示3步,本轮未提前结束或缩短REPLAN**——这只是改变了剩余窗口里
文字怎么写,不是新的“提前判定成功”的机制。

## 五、新增测试及结果

`ae/controllers/tests/test_replan_progress_wording.py`(新文件,45项检查),覆盖你列出的12类要求：

| 要求 | 测试 |
|---|---|
| 1. ssmc>0仍显示无进展+重新规划措辞 | `test_ssmc_positive_keeps_no_progress_wording`(4项) |
| 2/3/4/5/6. ssmc==0不出现"0 steps"/"Discard..."等,改为承认进展+继续路线+仍禁止退回 | `test_ssmc_zero_progress_aware_wording`(7项) |
| 7. "无进展→恢复→继续"窗口内directive随真实状态切换 | `test_three_step_window_switches_with_real_state`(3项) |
| 8. 再次无进展时切回重新规划分支 | `test_eight_switches_back_on_renewed_nonprogress`(3项) |
| 9. B+三个字段值/更新时序不变 | `test_b_plus_fields_and_timing_unaffected`(3项) |
| 10. REFLECT/VERIFY逐字不变 | `test_reflect_and_verify_directives_byte_identical_to_v1`(2项) |
| 11. patch duration仍是3步 | `test_patch_duration_still_three_steps`(2项) |

外加`test_replan_prompt_v2.py`里`safe-7`一项按新行为更新了断言(不再断言"for 0 steps"字面文本,改为断言ssmc缺失时正确默认0并选中分支B,不产生崩溃/None)。

全部用真实`StatefulController.step()`序列+真实`active_directive()`/`render_directive()`调用驱动,没有手写字符串断言、没有mock任何核心逻辑。

```
45 passed, 0 failed  (test_replan_progress_wording.py,本轮新增)
```

## 六、完整回归测试结果

```
test_ae_controller.py                          51 passed, 0 failed
test_ae_full_production_termination_gate.py    46 passed, 0 failed
test_canonical_action.py                       26 passed, 0 failed
test_intervention_outcome.py                   23 passed, 0 failed
test_pir_ablation_modes.py                     41 passed, 0 failed
test_post_intervention_exact_repeat.py         34 passed, 0 failed
test_replan_progress_wording.py(本轮新增)       45 passed, 0 failed
test_replan_prompt_v2.py(safe-7已更新)          43 passed, 0 failed
test_replan_sustained_evidence.py              20 passed, 0 failed
test_signal_semantics.py                       18 passed, 0 failed
------------------------------------------------------------------
controller测试合计                              347 passed, 0 failed

alfworld_runs_ae相关测试:
test_alfworld_action_normalize.py    42 passed, 0 failed
test_output_parser.py                38 passed, 0 failed
test_prompt_budget.py                18 passed, 0 failed
test_put_move_compat_integration.py  21 passed, 0 failed
test_environment_anchor.py            8 passed, 0 failed
------------------------------------------------------------------
合计                                 127 passed, 0 failed

总计474项检查,0失败。
```

## 七、对5个真实案例、7个矛盾directive的复核结果 —— **发现并修正了此前审计的一处计数错误**

用真实134题sustained-only日志逐offset重放(`last_nonprogress_action`/`last_nonprogress_observation`/
`current_observation`按B+规则从真实记录的signal序列累积重建,`steps_since_meaningful_change`用真实记录值),
对abs_idx=14/33/49/83/106这5个真实episode做了完整3步窗口重放,**新发现真实矛盾directive实例是10个,不是
此前`ae_replan_prompt_v2_b_plus_report.md`第九节报告的7个**——如实更正如下：

**此前7个数字的计算方法有误**:第九节当时只检查了窗口offset=2、3(用`ssmc_seq[k-2]`取"上一个offset的值"),
隐含假设"offset=1的directive一定显示武装时刻的ssmc(=patch_duration_steps,不可能是0)"。这个假设是错的:
路由判断读取的是**这一步自己的bookkeeping更新之前**的ssmc值,但"武装"(`_arm_intervention`真正设置
`active_patch_type`)发生在**这一步自己的bookkeeping更新之后**——如果武装那一步自己的动作恰好取得了真实
进展(3个真实案例里发生了,abs_idx=14/49/106),武装那一步自己记录的`steps_since_meaningful_change`就已经
是0,offset=1的directive因此会读到0,而不是3。

修正后的完整清单(10个矛盾directive实例,已用真实重放确认全部不再出现禁用措辞、全部正确显示分支B文本)：

| abs_idx | q_index | 出现矛盾的offset | 说明 |
|---|---|---|---|
| 14 | 15 | 1、2、3(全部3步) | 整个窗口全程都在推进任务(proxy全部1.0) |
| 33 | 34 | 2 | |
| 49 | 50 | 1、2、3(全部3步) | 同abs_idx=14,整个窗口全程推进 |
| 83 | 84 | 2 | |
| 106 | 7 | 1、3 | |

`test_real_log_replay_five_cases_seven_instances_no_longer_contradictory`(测试函数名沿用旧称呼,内容
已按10个实例校正)对全部10个实例逐一验证:无"for 0 steps"、无"Discard the current ineffective strategy"、
明确显示"New progress has been observed since REPLAN was activated."——**全部10个此前会矛盾的directive
实例,重放后确认已经修复。**

## 八、是否确认B+ bookkeeping和所有控制行为均未改变

**确认。** `stateful_controller.py`本轮零改动(用SHA256/diff双重核对,见下节)。B+的三个字段更新逻辑、
`local_state_change_proxy<0.5`门控、action/observation原子配对、`steps_since_meaningful_change`本身
的计算和路由门控、sustained-only路由公式、REPLAN武装/恢复/`_ESCALATION`升级逻辑、
`patch_duration_steps=3`、warmup/grace/cooldown/max_interventions、PIR及其消融模式、termination
policy、REFLECT/VERIFY文本——全部原样通过474项回归检查,0项需要修改预期值。本轮改动完全局限于
`intervention_renderer.py::_render_replan()`内部的措辞分支逻辑。

## 九、正式仓库SHA256核对

```
ae/controllers/stateful_controller.py:    35249c6949ab5e8fcd2d0292f34c4c68be78c78c57f15a92a088177dbd6f2f3f (与本轮开始前一致)
ae/controllers/intervention_renderer.py:  3b1f2039f5eb3d95e2470c694cd5a82ff39cd568f44273d825a2569abe0fb3a1 (与本轮开始前一致,仍是v1原始版本)
alfworld_runs_ae/agents.py:               811e2d5f8d417c6b756109a7f5d8e0eb4a5fc6600e459971dd192ef688915c05 (与本轮开始前一致)
```

**正式仓库未被本轮任何一次Edit/Write工具调用触及,全部改动只存在于隔离副本
`/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_workdir/`。**

## 十、是否确认没有运行GPU、Gate或134题实验

**确认。** 全程未运行任何GPU任务、未启动Gate、未启动134题正式实验、未清理
`reflect_first_encounter_repeat_enabled`孤儿配置项。第七节的"真实案例复核"是对已有历史日志的纯离线
重放(读取已存在的`episode_log.jsonl`,不涉及任何新的env/LLM调用)。

---

到此停止。不合入正式仓库,等待你审阅(包括第七节的10-vs-7计数修正、第三节指出的"failed"措辞遗留不一致)后再决定下一步。
