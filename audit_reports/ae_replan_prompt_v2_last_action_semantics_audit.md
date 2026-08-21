# REPLAN prompt-v2:`Last failed action`语义补查(只读,未改代码)

范围:`replan_v2_workdir`里已实现但**尚未合入仓库**的REPLAN prompt-v2代码
(`ae/controllers/stateful_controller.py`/`intervention_renderer.py`)。只读核查,不涉及GPU。

## 一、结论先行

**确认存在你怀疑的问题**:`last_action_text`/`last_observation_text`当前是**每步无条件滚动更新**的
"最近一次动作/观察",不是经过失败判定的"最近一次失败动作"。3步directive窗口内,t+2、t+3展示的
"Last failed action"会滚动更新为**t+1、t+2模型自己刚做的动作**——即便那个动作**取得了真实环境进展**,
也会被t+2/t+3的directive原样标注为"failed"并附带"Do not repeat the failed action"的禁止性指令。
用真实的134题sustained-only日志核实,**15次REPLAN武装里有5次(33%)** 的3步窗口内确实出现了这种情况。

## 二、真实调用时序(精确到代码行)

`stateful_controller.py::step()`最开头(不受任何判定门控)：

```python
self.last_action_text = display_action_text if display_action_text is not None else action
self.last_observation_text = (
    display_observation_text if display_observation_text is not None else observation
)
```

这两行位于`signal_extractor.extract()`调用**之前**,不读取`local_state_change_proxy`或任何其它信号,
**无论这一步是否有意义进展、是否invalid、是否重复,都会执行**。

`active_directive()`每次被调用时,直接读取`self.last_action_text`/`self.last_observation_text`的**当前实时值**:

```python
return render_directive(
    self.active_patch_type, goal=goal,
    steps_since_meaningful_change=self.steps_since_meaningful_change,
    last_action=self.last_action_text, last_observation=self.last_observation_text,
)
```

**完整时序(以一次REPLAN在step N武装、patch_duration_steps=3为例)**：

| 时刻 | 发生的事 | `self.last_action_text`此刻的值 |
|---|---|---|
| step N(武装那一步) | `step()`结束时,`last_action_text`被设为action_N | action_N(触发REPLAN的那个动作) |
| **t+1**(生成action_{N+1}的LLM调用) | `active_directive()`读取到action_N | directive显示"Last failed action: action_N" —— **正确**,这确实是导致REPLAN武装的动作 |
| step N+1执行完 | `controller.step()`用action_{N+1}调用,`last_action_text`**无条件**更新为action_{N+1} | 不管action_{N+1}是否成功,都覆盖了action_N |
| **t+2**(生成action_{N+2}的LLM调用) | `active_directive()`读取到action_{N+1} | directive显示"Last failed action: action_{N+1}" —— **如果action_{N+1}其实成功了(取得进展),这里仍会被标注为failed** |
| step N+2执行完 | `last_action_text`再次无条件更新为action_{N+2} | 同上逻辑 |
| **t+3** | 读取到action_{N+2} | 同样可能把一个成功动作标成failed |

**关键点**:`last_action_text`的更新完全不检查`signals.local_state_change_proxy`(即"这一步是否有意义变化"这个已有信号)——它只是"最近一次经过的动作",不是"最近一次失败的动作"。命名和实际语义不一致。

## 三、真实数据核实(134题sustained-only日志,15次REPLAN武装逐一核查)

对每次REPLAN武装,取窗口内3步(offset 1/2/3)各自的`local_state_change_proxy`(已有信号,1.0=有意义进展,0.0=无变化),检查是否存在"某一步进展了,但下一步的directive仍会把它标为failed"的情况：

| env_name | q_index | 武装step_idx | offset(1,2,3)的local_state_change_proxy |
|---|---|---|---|
| pick_heat_then_place_in_recep-Egg...(q=15) | 15 | 3 | **[1.0, 1.0, 1.0]** |
| look_at_obj_in_light-CD...(q=34) | 34 | 10 | **[1.0, 0.0, 1.0]** |
| pick_heat_then_place_in_recep-Egg...(q=50) | 50 | 3 | **[1.0, 1.0, 1.0]** |
| pick_clean_then_place_in_recep-Cloth...(q=84) | 84 | 11 | **[1.0, 0.0, 0.0]** |
| pick_heat_then_place_in_recep-Egg...(q=7) | 7 | 3 | **[0.0, 1.0, 1.0]** |

**5/15次REPLAN武装的窗口内,offset=1(或=2)的动作取得了真实进展(proxy=1.0),但按当前实现,offset=2(或=3)的directive仍会把这个刚刚成功的动作标注为"Last failed action"并要求"Do not repeat"。** 这不是假设性场景,是真实日志里已经发生过的真实动作序列(v1版本当时展示的是静态文本、不受这个问题影响,但v2一旦合入仓库并跑这些同款场景,会复现这个问题)。

**同样的问题也影响"Environment response"字段**:比如q=15/q=50这两题,offset=1的观察是"You pick up the egg 1..."这类明显成功的反馈文本,如果offset=2的directive把它填进"Environment response"却同时说"Do not repeat the failed action",文本本身就自相矛盾。

## 四、最小修正方案(仅提案,未实施)

两个候选,都只改`last_action_text`/`last_observation_text`的**赋值时机**,不改directive模板文字、不改其它任何行为：

**方案A:武装时刻冻结**。只在`_arm_intervention()`里赋值一次(用当时的action/observation),`step()`顶部的无条件赋值整段删掉。效果:整个3步窗口内"Last failed action"**恒等于触发这次REPLAN的那个动作**,不再滚动。最简单,改动最小,但代价是t+2/t+3看不到窗口内自己刚做的新尝试是否也失败了(如果模型在t+1换了个新动作、结果也失败,t+2的directive不会提及这第二次失败,只会重复提第一次)。

**方案B:按"是否有进展"门控滚动**(需要把赋值从`step()`最开头挪到`signal_extractor.extract()`调用之后,读取`signals.local_state_change_proxy`)：
```python
if signals.local_state_change_proxy < 0.5:
    self.last_action_text = display_action_text if display_action_text is not None else action
    self.last_observation_text = display_observation_text if display_observation_text is not None else observation
```
效果:只要模型持续失败,"Last failed action"会滚动更新到**最近一次真正的失败**(比方案A更贴合窗口内的实时情况);一旦某一步真的取得进展,`last_action_text`就**停止更新**,不会把这个成功动作标成failed——语义上更准确地匹配"failed"这个词本身的含义。代价:多一处判断逻辑,需要把赋值挪到`signal_extractor.extract()`之后(仍在`step()`顶部区域,不影响其后的signal/状态/路由逻辑,因为它们本来就不读这两个属性)。

**倾向建议**:方案B在语义上更准确(“failed”真正只指失败的动作),且改动量仍然很小(一个条件判断+挪动两行代码的位置),我倾向方案B;但方案A更保守、心智负担更低,如果你更看重"绝对不引入任何依赖`local_state_change_proxy`的额外分支"这条原则,方案A也是合理选择。两个方案都不需要新增state字段或config项。

---

到此停止,未修改任何代码,未运行GPU。等你确认采用哪个方案(或提出其它修正)后再实施。
