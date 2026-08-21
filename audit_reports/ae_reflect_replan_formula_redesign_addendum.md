# 补充审计:unresolved_intervention_count详查 + 离线重放措辞修正

对`audit_reports/ae_reflect_replan_formula_redesign.md`的两点补充,仍是只读审计,未修改代码。

---

## 一、`unresolved_intervention_count`详细审计

**全代码库(`stateful_controller.py`)对这个字段只有3处引用,已逐一核查:**

| 行号 | 内容 |
|---|---|
| 116 | `self.unresolved_intervention_count: int = 0`——仅在`reset()`里初始化,每个episode开始时清零一次 |
| 270 | PIR(post_intervention_exact_repeat)块内`+= 1`——**当前配置`post_intervention_repeat_enabled=False`,这条路径永不执行,实际不生效** |
| 379 | 正常patch-duration到期评估块内`+= 1`——**对`self.pending_intervention_type`（可以是VERIFY/REFLECT/REPLAN中任意一个)一视同仁,只要到期时`meaningful_change_since_intervention`为False就+1,不区分是哪种干预类型** |

**全文没有任何地方对这个字段做减法或除episode边界外的清零。**

### 逐问回答

**1. 哪些类型的干预会使其增加**:VERIFY、REFLECT、REPLAN三种都会——只要该次介入在`patch_duration_steps`到期时被判定"没有产生有意义变化",不论是哪种类型,都会让这个计数器+1。**它不是"局部修复(REFLECT)专属的失败计数",是任意类型介入失败的通用计数。**

**2. 何时清零或下降**:只有整个episode结束、下一个episode调用`reset()`时才会归零。**episode内部从不下降,也从不因为后续成功恢复而清零。**

**3. 达到1后是否会在episode剩余阶段一直满足**:**是,永久成立。** 一旦任意一次介入(不论类型)被判定unresolved,`unresolved_intervention_count>=1`这个条件会在该episode剩余的每一步里恒为True,不会因为之后再来几次成功的recovered介入而失效。

### 结论

**`unresolved_intervention_count`不具备类型针对性(VERIFY/REFLECT/REPLAN失败都计入),也不会在恢复后清零(单调不减,一次触发即永久生效)。这正是你在决策规则里预先设定的两个否决条件,均已确认成立。**

### 新公式16次真实REPLAN武装的分类统计(完整134题重放)

用真实controller代码重放,精确追踪每次REPLAN真实武装时,是由`sustained`(steps_since_meaningful_change≥patch_duration_steps)、`plan_history`(unresolved_intervention_count≥1)还是两者共同放行:

| 放行方式 | 次数 |
|---|---|
| sustained_only | 7 |
| plan_history_only | **1** |
| both | 0 |
| (通过既有`_ESCALATION`调度升级路径武装,不经过本次改动的判断,与本次公式改动无关) | 8 |

16次里,只有7次是经由新判断条件(`frustration_high+confidence_low` + 新增证据)从头触发的REPLAN,其中**仅1次纯靠`plan_history`放行**,其余7次全部是`sustained`独立放行,`both`(两者同时满足)0次。另外8次REPLAN来自既有的、本次完全没有改动的`_ESCALATION`调度升级机制(REFLECT/REPLAN unresolved后自动升级),与本次公式改动无关,不应计入这次讨论的"新证据放行"统计。

### 仅由plan_history放行的真实案例

**abs_idx=105,第18步**:`frustration_high(1.00) confidence_low(0.00) ssmc=0 unresolved_count=1`。

**这个案例精确暴露了问题**:`steps_since_meaningful_change=0`——意味着这一步**刚刚发生过有意义的变化**(不是当前正在持续卡住),纯粹因为episode更早之前有过一次别的介入(不确定是VERIFY还是REFLECT)被判定unresolved,留下的`unresolved_count=1`这个陈旧标记,让REPLAN在一个"当前其实刚有进展"的时刻被放行。**这与"计划级持续失败"的本意不符,是一次误放行。**

---

## 二、离线重放结论措辞修正

**原报告第四节"预计触发分布"表述需要修正如下:**

> ~~这是真正的"实际武装"计数~~ → **这是把新公式代入历史信号后,在旧轨迹(旧LLM动作序列)上重放出的"反事实路由行为"计数,不是新公式真实运行时会产生的触发分布。**

**准确的方法论边界**:本次重放全程复用的是**旧运行(REPLAN-always-wins逻辑跑出来的那次134题)记录下来的signals和action**,只改变了`_select_intervention()`内部的判断逻辑本身,env从未重新交互、LLM从未被重新调用。这样得到的"64→73(REFLECT)、33→16(REPLAN)、18→28(VERIFY)"以及"14个episode≥2次REFLECT、11个episode两种类型都出现"等数字,**只能证明"如果把新公式套在这套已经发生过的旧轨迹上,路由决策会如何不同"这一结构性事实**——**不能当作"新公式真实上线后,134题里REFLECT/REPLAN的真实触发次数会是多少"的预测**。

**具体原因**:一旦某一步的路由从REPLAN改判为REFLECT,模型收到的directive文本就变了,下一步模型的真实动作大概率会不同于旧日志里记录的那个动作——而重放用的正是旧日志里固定的动作序列,这在改判发生之后的所有后续step上都不再代表"新公式实际会发生什么",只是延用旧数据做的结构性检查。

**因此**:73/16/28这组数字,以及"REFLECT不再被压制"这个方向性结论,**只能作为"新公式的判断逻辑本身在结构上确实会显著减少REPLAN的触发面、增加REFLECT的触发面"这一结构性事实的证据**,**不能据此断言真实运行中REFLECT和REPLAN不存在新的压制问题**——真实的触发分布、是否存在新的失衡,仍然需要一次真实的GPU/LLM Gate才能验证(本轮明确未做,也未建议现在做)。

---

## 三、是否采用plan_history——按你预设的决策规则给出结论

**你的规则**:"如果它不具备类型针对性或恢复后不会清零,优先考虑仅使用`steps_since_meaningful_change >= patch_duration_steps`作为本轮最小修改,不要为了保留第二个条件而引入新的复杂状态机。"

**两个否决条件均已确认成立**(不具备类型针对性:VERIFY/REFLECT/REPLAN失败均计入;不会在恢复后清零:单调不减,一次触发永久生效),且找到了一个真实的误放行案例(abs_idx=105第18步,ssmc=0却被plan_history单独放行)。

**结论:建议本轮只采用`steps_since_meaningful_change >= patch_duration_steps`一个条件,不采用`unresolved_intervention_count`/plan_history,也不为了修补plan_history的问题(比如改成"只统计REFLECT类型的unresolved"或"允许随恢复而衰减")引入新的独立计数器或状态机——按你的要求,直接采用更简单的单条件版本。**

修改后的最小公式(仅此一处判断,其余完全不变):

```python
if frustration_high and confidence_low:
    if self.steps_since_meaningful_change >= self.config.patch_duration_steps:
        return REPLAN
    # 否则不在这里返回,继续往下走(可能落到REFLECT)
if invalid or (frustration_medium and repeated):
    return REFLECT
if uncertainty_high or surprise_high:
    return VERIFY
return CONTINUE
```

**已提前算好精确数字(仅sustained条件,完整134题重放)**:

| | both条件(73/16/28) | 仅sustained | 变化 |
|---|---|---|---|
| REFLECT | 73 | **74** | +1(abs_idx=105那个误放行案例现在正确落到REFLECT) |
| REPLAN | 16 | **15** | -1 |
| VERIFY | 28 | 28 | 不变 |
| episode含≥2次REFLECT | 14 | 15 | +1 |
| episode含≥2次REPLAN | 3 | 2 | -1 |
| episode两种类型都出现 | 11 | 11 | 不变 |

**如预期,去掉plan_history后数字几乎没有实质性变化(只差1次),同时消除了唯一一个"当前刚有进展却被历史陈旧标记放行REPLAN"的误判案例。仅sustained版本在结构上和保留plan_history的版本效果几乎等价,但逻辑更干净、状态更少——符合你"不引入新复杂状态机"的要求。

---

到此停止,未修改代码,未运行GPU实验。等你确认:是否采用"仅sustained"这个最终版本,以及是否需要先补一次仅sustained条件的精确重放数字再决定是否实施。
