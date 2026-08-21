# REFLECT/REPLAN公式重新设计——只读审计+方案(未改代码)

## 一、当前REFLECT/REPLAN的准确公式、阈值、优先级

```python
# stateful_controller.py::_select_intervention() 第532-565行,严格顺序、短路求值
invalid = signals.invalid_action >= 0.5
repeated = signals.consecutive_exact_action_repeat >= 0.5

if frustration_high and confidence_low:          # 第546行,优先级最高
    return REPLAN
if invalid or (frustration_medium and repeated):  # 第551行
    return REFLECT
if uncertainty_high or surprise_high:             # 第559行
    return VERIFY
return CONTINUE
```

| 量 | 定义 | 阈值 |
|---|---|---|
| `frustration_medium` | `state.frustration >= config.frustration_medium`(瞬时,**无滞回**) | 0.45 |
| `frustration_high` | hysteresis滞回带 | enter=0.70, exit=0.40 |
| `confidence_low` | hysteresis滞回带(方向相反) | enter=0.35, exit=0.55 |
| `frustration`更新公式 | `clip01(0.85*prev + 0.35*invalid_action + 0.30*repeated_action + 0.20*repeated_observation - 0.35*progress_signal)` | decay=0.85 |
| `confidence`更新公式 | `clip01(0.75*prev + 0.35*progress_signal + 0.10*observation_novelty - 0.25*invalid_action - 0.15*repeated_action)` | decay=0.75 |

## 二、repeated_action如何同时影响两类干预

`repeated_action`是`frustration_weights`里的**正**权重项(+0.30)、同时是`confidence_weights`里的**负**权重项(-0.15)——同一个信号在架构上同时把frustration往上推、confidence往下压,两者天然协同变化。

**已用真实134题日志逐步验证**(此前专项审计已完成):`frustration_medium AND consecutive_exact_action_repeat`成立的90个step,**100%同时满足`frustration_high AND confidence_low`**,导致90次里0次真正走到REFLECT分支——全部被优先级更高的REPLAN截胡。逐步复现证实:只要精确重复维持2-3步,`repeated_action`就把frustration推向饱和上限(0.867-1.000区间),同时把confidence推向下限(0-0.124),这是权重公式驱动的结构性结果,不是数据巧合。

## 三、建议的最小公式修改

**只改REPLAN的判定条件,不改REFLECT、不改任何权重/decay/阈值数值、不改VERIFY、不引入新状态机、不新增独立的每类型预算:**

```python
if frustration_high and confidence_low:
    sustained = self.steps_since_meaningful_change >= self.config.patch_duration_steps
    plan_history = self.unresolved_intervention_count >= 1
    if sustained or plan_history:
        return REPLAN
    # 否则不在这里返回,继续往下走(可能落到REFLECT)
if invalid or (frustration_medium and repeated):
    return REFLECT
if uncertainty_high or surprise_high:
    return VERIFY
return CONTINUE
```

**只用两个已存在、未新增的状态量**:

- `steps_since_meaningful_change`:已在`stateful_controller.py`里逐步维护(`local_state_change_proxy<0.5`时递增,有意义变化时归零)——直接复用作"持续无进展"证据,**不新增任何字段**。
- `unresolved_intervention_count`:已在episode级累计维护(每次评估为unresolved时+1,`reset()`清零)——直接复用作"过去已有局部修复失败"的计划级证据,**不新增任何字段**。
- 阈值直接复用`config.patch_duration_steps`(当前3),**不新增任何config字段**,理由:这本来就是代码里"判断一次介入是否生效"的既定观察窗口,"卡住时间≥这个窗口"和"介入没生效"是同一把尺子,概念上一致。

**这如何满足你的6条要求**:

1. `repeated_action`/短期重复/局部动作错误仍然只走REFLECT分支(该分支完全未改)。
2. REPLAN现在必须额外满足"持续无进展(steps_since_meaningful_change≥patch_duration_steps)"或"已有局部修复失败史(unresolved_intervention_count≥1)"之一,不能仅凭同一次repeated_action触发的瞬时frustration_high+confidence_low。
3. 两者理论上仍可能同时满足(见五),此时**不是固定REPLAN优先,而是REPLAN只有在拿到独立于REFLECT信号的额外证据时才会赢**——这就是"根据局部故障/计划故障证据选择"的具体实现:REPLAN赢的唯一原因是它自己有独立证据,不是靠"优先级"这个人为规则赢的。
4. 未要求REFLECT必须先于REPLAN发生——如果一上来就是"已经卡了很久+又精确重复",REPLAN可以直接触发,不强制经过REFLECT。
5. `_select_intervention()`每步都重新计算,`sustained`/`plan_history`两个判据都是动态的,一次介入结束后(recovered或unresolved)状态continues演化,同一episode自然可能出现多次REFLECT、多次REPLAN。
6. 见五、六的数据。

## 四、离线重放的预计触发分布(全真实134题日志,完整episode重放,非仅候选点估算)

用真实controller代码,把新公式临时打入内存(未改动`stateful_controller.py`文件本身),对134题的**完整**step序列逐步重放(不是只看候选点,是把每个episode从头到尾按新公式重新跑一遍决策,信号仍用历史真实记录值精确驱动,保证与旧版逐字段可比):

| | 旧公式(真实记录) | 新公式(完整重放) | 变化 |
|---|---|---|---|
| REFLECT实际武装次数 | 64 | **73** | **+9(+14%)** |
| REPLAN实际武装次数 | 33 | **16** | **-17(-52%)** |
| VERIFY实际武装次数 | 18 | 28 | +10(下游连锁效应,REPLAN少触发后severity轨迹变化,VERIFY更容易被判定为新事件) |

**这是真正的"实际武装"计数,不是候选信号计数**——用的是与之前专项审计相同的严格重放方法(把历史信号精确喂给真实controller,走完整个`step()`的is_event/severity-upgrade/cooldown/warmup门控),不是简单的信号叠加估算。

## 五、REFLECT、REPLAN分别会在哪些真实故障场景触发(具体例子)

**REFLECT真实触发场景**(举两例):
- `invalid_action`:模型生成了不在admissible_commands里的动作(格式错误/幻觉动作名)——REFLECT分支完全未改,继续覆盖这类"局部操作错误"。
- **新增的"改判"场景**(旧版本会走REPLAN,新版本走REFLECT的真实案例):abs_idx=60第24步,旧公式走`REPLAN(frustration_high(0.85) and confidence_low)`,新公式因为`steps_since_meaningful_change`和`unresolved_intervention_count`都不满足"持续/历史"证据,改判走REFLECT(该step恰好`invalid_action`也为真,归类为"局部动作错误"处理)。

**REPLAN真实触发场景**:
- abs_idx=60的更早一步等场景里,`sustained=True`(卡住时间已经≥patch_duration_steps)或`plan_history=True`(之前已经有一次局部修复被判定unresolved)——这些case下REPLAN仍然会触发,且触发时机比旧公式更"名副其实":不是"因为一次精确重复导致瞬时分数飙升",而是真的已经持续卡住或已经试过局部修复但没用。

## 六、是否仍存在某一种干预被另一种长期压制的问题

**REFLECT不再被系统性饿死,证据**:

- 全量武装次数:64→73,**净增加**,不是被压制。
- **14个episode**在新公式下出现**≥2次REFLECT真实武装**(旧公式下REFLECT几乎从未获得过因repeated_action触发的机会,现在能规律性地多次触发)。
- **11个episode**在新公式下**REFLECT和REPLAN都至少各武装一次**——直接实现你要的"两种干预各自有清晰、互不吞噬的触发区域"。
- REPLAN仍保留**3个episode出现≥2次真实武装**——说明REPLAN没有被反向压制,遇到真正持续卡住或有失败史的情况依然会连续触发。

**没有发现新的压制问题。** REPLAN从33降到16,降幅超过一半,但这是"REPLAN不再被单次repeated_action误判触发"的直接结果,不是REPLAN本身的合法触发场景被压制——16次剩下的都满足`sustained`或`plan_history`这两个独立证据之一,是有理有据的触发,不是残留的误判。

---

**未修改代码,未运行GPU实验。** 到此停止,等你确认是否按第三节方案实施(仅改`_select_intervention()`一处判断逻辑,复用`steps_since_meaningful_change`和`unresolved_intervention_count`两个既有状态量,不新增config字段,不改权重/阈值/decay,不改REFLECT/VERIFY分支)。
