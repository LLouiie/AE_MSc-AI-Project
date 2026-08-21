# REPLAN prompt-v2 "B+"语义修复实施报告

**未改动正式仓库,未运行GPU,未启动Gate或134题实验,未清理孤儿配置项。** 全部实施+测试仍在隔离副本
`/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_workdir/`里完成。

## 〇、开始前只读核查:5-seed批任务状态

```
[2026-08-06 13:43:23] START version=old seed=45 run_prefix=ae_full_cooldown5_old_seed45
[2026-08-06 14:45:30] DONE  version=old seed=45 run_prefix=ae_full_cooldown5_old_seed45 rc_practice=0 rc_exam=0
[2026-08-06 14:45:30] START version=old seed=46 run_prefix=ae_full_cooldown5_old_seed46
```

**仍在运行**,当前处于"旧版REPLAN-unconditional"阶段的最后一个种子(seed=46),尚未进入"新版sustained-only"阶段。正式仓库`ae/controllers/stateful_controller.py`目前仍是旧版内容,本次全部改动只在隔离副本进行,未触碰正式仓库文件。

---

## 一、实际修改的文件

- `ae/controllers/intervention_renderer.py`
- `ae/controllers/stateful_controller.py`

`alfworld_runs_ae/agents.py`**本轮未修改**——已核对,B+只改了`stateful_controller.py`内部字段名/更新时机和`intervention_renderer.py`的字段名/措辞,`step()`/`active_directive()`对外的调用签名(参数名`display_action_text`/`display_observation_text`/`goal`)完全没变,`agents.py`那两处调用点不需要跟着改。

## 二、精确diff

```
/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_bplus_diff_controller.diff  (93行)
/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_bplus_diff_renderer.diff    (123行)
```

**`stateful_controller.py`核心改动**：

```diff
-        self.last_action_text: Optional[str] = None
-        self.last_observation_text: Optional[str] = None
+        self.last_nonprogress_action_text: Optional[str] = None
+        self.last_nonprogress_observation_text: Optional[str] = None
+        self.current_observation_text: Optional[str] = None
```

```diff
         self.step_index += 1

-        self.last_action_text = display_action_text if display_action_text is not None else action
-        self.last_observation_text = (
-            display_observation_text if display_observation_text is not None else observation
-        )
-
         signals = self.signal_extractor.extract(
             action=action,
             observation=observation,
             ...
         )

+        _display_action = display_action_text if display_action_text is not None else action
+        _display_observation = display_observation_text if display_observation_text is not None else observation
+        self.current_observation_text = _display_observation
+        if signals.local_state_change_proxy < 0.5:
+            self.last_nonprogress_action_text = _display_action
+            self.last_nonprogress_observation_text = _display_observation
```

```diff
         return render_directive(
             self.active_patch_type, goal=goal,
             steps_since_meaningful_change=self.steps_since_meaningful_change,
-            last_action=self.last_action_text,
-            last_observation=self.last_observation_text,
+            last_nonprogress_action=self.last_nonprogress_action_text,
+            last_nonprogress_observation=self.last_nonprogress_observation_text,
+            current_observation=self.current_observation_text,
         )
```

**`intervention_renderer.py`核心改动**(字段/措辞)：

```diff
-        f"Last failed action: {action_text}\n"
-        f"Environment response: {observation_text}\n"
-        f"Current observation: {observation_text}\n"
+        f"Most recent action with no observed progress: {action_text}\n"
+        f"Environment response to that action: {nonprogress_obs_text}\n"
+        f"Current observation: {current_obs_text}\n"
```
`_render_replan()`/`render_directive()`签名的`last_action`/`last_observation`两个参数拆成`last_nonprogress_action`/`last_nonprogress_observation`/`current_observation`三个,`_NO_PRIOR_ACTION`/`_NO_PRIOR_OBSERVATION`占位符改名并新增`_NO_CURRENT_OBSERVATION`,含义一一对应新字段。

## 三、字段重命名及数据来源

| 旧字段 | 新字段 | 数据来源 | 更新时机 |
|---|---|---|---|
| `last_action_text` | `last_nonprogress_action_text` | `display_action_text`(回退到`action`) | **仅当**`signals.local_state_change_proxy < 0.5`时更新,与`last_nonprogress_observation_text`原子地(同一个if块内)一起赋值 |
| `last_observation_text` | `last_nonprogress_observation_text` | `display_observation_text`(回退到`observation`) | 同上,与action同一个step、同一个if块 |
| (无) | `current_observation_text`(新增) | `display_observation_text`(回退到`observation`) | **每步无条件更新**,不受进展门控 |

`local_state_change_proxy`是**已有信号**(`signals.py`定义,`steps_since_meaningful_change`已经在用同一个信号做门控),本次**未新增第二套进展判定**,直接复用同一个来源。

## 四、更新发生在调用时序的哪个位置

赋值代码块现在位于`signals = self.signal_extractor.extract(...)`**之后**(此前是之前),因为需要读取这次
`extract()`算出的`signals.local_state_change_proxy`。这几行仍然在`_select_intervention()`路由判断、
`update_state()`状态更新、`_arm_intervention()`武装逻辑之前,不读取、不影响后续任何一步的决策——只是
利用已经算好的`signals`对象里的一个既有字段。

## 五、为什么成功动作不再被误标

新的赋值逻辑加了一个门控条件:`if signals.local_state_change_proxy < 0.5:`。当这一步**真的取得了可观测进展**(proxy>=0.5)时,这个if块不执行,`last_nonprogress_action_text`/`last_nonprogress_observation_text`**维持上一次记录的值不变**——也就是说,一个成功的动作永远不会被写进这两个字段,自然也不会出现在directive的"Most recent action with no observed progress"这一行里。`current_observation_text`则相反,不带这个门控,每步都无条件刷新,所以模型看到的"Current observation"永远是最新的真实观察,无论成功失败。

## 六、真实"失败→成功→再次失败"三步示例

用真实`StatefulController.step()`跑出来的实际值(非手写)：

| 步骤 | action | observation | `last_nonprogress_action_text` | `last_nonprogress_observation_text` | `current_observation_text` |
|---|---|---|---|---|---|
| 1(失败) | `go to shelf 1` | `Nothing happens.` | `go to shelf 1` | `Nothing happens.` | `Nothing happens.` |
| 2(成功) | `take mug 1 from shelf 1` | `You pick up the mug 1 from the shelf 1.` | **`go to shelf 1`(未被覆盖)** | **`Nothing happens.`(未被覆盖)** | `You pick up the mug 1 from the shelf 1.`(**已更新**) |
| 3(再次失败) | `go to shelf 2` | `Nothing happens.` | `go to shelf 2`(**已更新到新失败**) | `Nothing happens.`(**已更新**) | `Nothing happens.` |

第2步精确展示了修复效果:成功动作没有污染"no observed progress"记录,但"Current observation"如实反映了这次成功。第3步展示了记录会正常滚动到新的失败,不会永久卡死在第一次失败上。

## 七、新增测试及结果

`ae/controllers/tests/test_replan_prompt_v2.py`(重写,43项检查,覆盖你列出的13类要求里除#11/#12/#13外的全部10类；#11由既有回归套件覆盖,#12/#13见下节)：

| 要求 | 测试 |
|---|---|
| 1. 无进展动作+observation成对记录 | `test_nonprogress_pair_recorded_together`(2项) |
| 2. 进展不覆盖之前的无进展记录 | `test_progress_does_not_overwrite_then_later_nonprogress_updates`(2c/2d) |
| 3. 进展动作不会写成"no observed progress" | 同上测试内(check "3.") |
| 4. 后续无进展会更新记录 | 同上测试内(4a/4b) |
| 5. action/反馈始终同一个step | `test_nonprogress_pair_recorded_together`(原子性) |
| 6. Current observation始终最新 | `test_current_observation_always_latest_regardless_of_progress`(3项) |
| 7. REPLAN三步窗口内进展不被误标 | `test_replan_window_progress_not_mislabeled_on_next_directive`(5项,含窗口内t+1进展后t+2 directive核查) |
| 8. 缺失字段安全降级 | `test_missing_optional_state_degrades_safely_no_none_no_blank_no_fabrication`(7项) |
| 9. reset()清空三个字段 | `test_reset_clears_all_three_fields`(4项) |
| 10. REFLECT/VERIFY逐字不变 | `test_reflect_and_verify_directives_byte_identical_to_v1`(3项) |

全部用真实`StatefulController.step()`序列+真实`active_directive()`/`render_directive()`调用驱动,没有手写字符串断言、没有mock任何核心逻辑;另有一项复刻`agents.py`真实拼接逻辑的端到端测试。

```
43 passed, 0 failed  (test_replan_prompt_v2.py)
```

## 八、完整回归测试结果

全部8个既有controller测试文件,同一隔离副本内运行：

```
test_ae_controller.py                          51 passed, 0 failed
test_ae_full_production_termination_gate.py    46 passed, 0 failed
test_canonical_action.py                       26 passed, 0 failed
test_intervention_outcome.py                   23 passed, 0 failed
test_pir_ablation_modes.py                     41 passed, 0 failed
test_post_intervention_exact_repeat.py         34 passed, 0 failed
test_replan_sustained_evidence.py              20 passed, 0 failed
test_signal_semantics.py                       18 passed, 0 failed
test_replan_prompt_v2.py(本次)                  43 passed, 0 failed
--------------------------------------------------------------------
合计                                            302 passed, 0 failed
```

与本次数据传递相关的`alfworld_runs_ae/tests/`：

```
test_alfworld_action_normalize.py    42 passed, 0 failed
test_output_parser.py                38 passed, 0 failed
test_prompt_budget.py                18 passed, 0 failed
test_put_move_compat_integration.py  21 passed, 0 failed
test_environment_anchor.py            8 passed, 0 failed
--------------------------------------------------------------------
合计                                 127 passed, 0 failed
```

**总计429项检查,0失败。** sustained-only路由(`test_replan_sustained_evidence.py`)、PIR全部消融模式
(`test_pir_ablation_modes.py`)、patch duration/warmup/grace/cooldown/budget/终止策略(`test_ae_full_production_termination_gate.py`/`test_intervention_outcome.py`)均原样通过,无需修改任何预期值。

## 九、只读核查:进展后剩余REPLAN窗口措辞(未修改代码)

**问题真实存在。** 用真实134题sustained-only日志(15次REPLAN武装的`steps_since_meaningful_change`
逐offset取值)核查:`active_directive()`展示给下一次LLM调用的`steps_since_meaningful_change`是**实时读取**
的(这次B+修复没有改动这个字段的更新逻辑,你也明确要求不改`patch_duration_steps`,所以这个问题独立于
本次改动、修复前后都存在)。

**发生次数**:全部15次REPLAN武装的3步窗口里,**5次武装(abs_idx=14、33、49、83、106)、共7个directive实例**会展示字面`"No meaningful progress has been made for 0 steps."`,与同一份directive里静态存在的
`"Discard the current ineffective strategy."`直接矛盾(0步无进展 vs 要求放弃"无效"策略,逻辑不自洽)：

| abs_idx | env_name | q_index | 武装step_idx | 矛盾出现在窗口offset | 该窗口3步完整ssmc取值 |
|---|---|---|---|---|---|
| 14 | pick_heat_then_place_in_recep-Egg...(q=15) | 15 | 3 | offset=2、offset=3(两次) | [0, 0, 0] |
| 33 | look_at_obj_in_light-CD...(q=34) | 34 | 10 | offset=2 | [0, 1, 0] |
| 49 | pick_heat_then_place_in_recep-Egg...(q=50) | 50 | 3 | offset=2、offset=3(两次) | [0, 0, 0] |
| 83 | pick_clean_then_place_in_recep-Cloth...(q=84) | 84 | 11 | offset=2 | [0, 1, 2] |
| 106 | pick_heat_then_place_in_recep-Egg...(q=7) | 7 | 3 | offset=3 | [1, 0, 0] |

abs_idx=14和49这两题最极端:整个3步窗口`local_state_change_proxy`全部是1.0(全程都在推进任务),但因为
directive固定展示满3步(不因中途恢复提前结束,这是已有设计,本次未改),offset=2和offset=3的directive
都会显示"0步无进展"+"放弃当前无效策略"这种自相矛盾的文本。

**最小措辞修正方案(仅提案,不在本轮实施)**：把静态的"Discard the current ineffective strategy."
一行改成条件性措辞,比如按`steps_since_meaningful_change`是否为0动态选择:
- `steps_since_meaningful_change > 0`时保留原句"Discard the current ineffective strategy."
- `steps_since_meaningful_change == 0`时替换为类似"Progress just occurred — continue with a coherent next subgoal instead of discarding what just worked."这样的句子

这个修正只需要在`_render_replan()`里加一个条件分支,不改`patch_duration_steps`、不改路由、不改
`steps_since_meaningful_change`本身的计算/门控逻辑。是否采纳、具体措辞留你决定。

## 十、是否确认除REPLAN prompt文本和必要bookkeeping外,没有其他有效行为变化

**确认。**
- `stateful_controller.py`里唯一涉及决策/信号计算的代码(`_select_intervention()`、`update_state()`、
  patch-expiry评估、`_arm_intervention()`、`_ESCALATION`)本轮**完全未触碰**。
- 新增/改名的3个属性(`last_nonprogress_action_text`/`last_nonprogress_observation_text`/
  `current_observation_text`)只在`active_directive()`里被读取,不出现在任何其它方法里。
- `step()`/`active_directive()`对外签名不变(`display_action_text`/`display_observation_text`/`goal`
  三个kwarg名字和默认值都和上一轮v2实现一致),`agents.py`不需要改动。
- 429项回归测试(覆盖路由、PIR全部消融模式、终止策略、intervention outcome、signal语义、canonical
  action、prompt budget、动作归一化等)全部原样通过,0项需要修改预期值。

## 十一、是否确认没有改动正式仓库、没有运行GPU

**确认。** 全部改动只存在于`/vol/gpudata/jy625-ae-data/scratch_ae3/replan_v2_workdir/`这份隔离副本里;
正式仓库`/homes/jy625/projects/AE3/`下的`ae/controllers/stateful_controller.py`、
`ae/controllers/intervention_renderer.py`、`alfworld_runs_ae/agents.py`均未被本轮任何一次Edit/Write
工具调用触及(可用`sha256sum`核对,正式仓库这几个文件的哈希值与本轮开始前完全一致)。全程未运行任何
GPU任务、未启动Gate、未启动134题正式实验、未清理`reflect_first_encounter_repeat_enabled`孤儿配置项。

---

到此停止。等待你审阅后再决定是否合入正式仓库、是否采纳第九节的措辞修正方案。
