# "分类型诊断 + 逐级升级"路由设计审计(只读,无代码修改,无GPU实验)

## ⚠️ 先做一次数据完整性纠正(影响上一份报告的部分数字)

本轮分析过程中发现:**134题里`env_name`不是唯一标识**——只有47个不同env_name,大多数重复出现2-3次(同一task_dir的不同trial共享同名,这在此前的memory记录里已经提示过,这次分析时疏忽未遵守)。凡是用`{r["env_name"]: r for r in all_rows}`这种字典构造做"先收集一批episode名字、再反查结果"的地方,会**静默丢弃除最后一次外的所有同名episode**。

**核实结果:上一份报告(`ae_frustration_routing_audit.md`)第三节"REFLECT优先"反事实的20-episode表受到此bug影响。** 用正确的abs_idx(practice 0-99、exam 100-133)重新核算:

| | 上一份报告(错误) | 本次修正后(正确) |
|---|---|---|
| 涉及episode数 | 20 | **30** |
| 成功 | 5 | 5(不变) |
| 失败 | 15 | **25** |

修正原因:47个env_name里有大量重复,原计算把多个不同trial的episode合并成了一条记录,漏计了10个失败episode。**该报告里"90个step/100%重合/条件重叠表/frustration-confidence数值分布"等基于直接逐行遍历(不经过按名字查表)的统计不受影响,仍然正确**;唯一受影响的是第三节反事实表格的episode计数和成功/失败拆分,已在此更正。`ae_frustration_signal_audit.md`里"novelty_progress_threshold反事实"部分(47/43个episode的成功分布)用了同样有风险的写法,尚未重新核实,标记为待核实,不在本轮范围内处理(你已明确本轮不讨论novelty)。

本轮以下所有数字均已用修正后的方法(按abs_idx)重新计算,不会重复这个错误。

---

## 一、真实职责、触发条件、优先级、状态更新链

```python
# stateful_controller.py::_select_intervention() 第532-565行,严格顺序、短路求值
if frustration_high and confidence_low:
    return REPLAN                                    # 第546行
if invalid_action or (frustration_medium and consecutive_exact_action_repeat):
    return REFLECT                                    # 第551行
if uncertainty_high or surprise_high:
    return VERIFY                                     # 第559行
return CONTINUE
```

状态更新链(信号→state→signature→候选→是否真正武装):

```
SignalExtractor.extract() → update_state() → HysteresisTracker.update()
→ compute_signature() → _select_intervention()(候选)
→ step()里的is_event/warmup/cooldown/budget门控 → 是否真正arm
→ _arm_intervention()武装后进入pending状态
→ 到patch_duration_steps到期时评估meaningful_change_since_intervention
  → True: "recovered",重置last_candidate_severity,允许同severity再次触发
  → False: "unresolved",按_ESCALATION字典({VERIFY→REFLECT, REFLECT→REPLAN, REPLAN→REPLAN})
    调度升级,等cooldown清零后真正武装
```

---

## 二、现有状态是否足以实现候选升级链

| 你要求的状态 | 现有实现 | 是否够用 |
|---|---|---|
| previous/active intervention | `current_intervention_type`/`current_intervention_id`/`pending_intervention` | **够用,已存在** |
| recovery grace | `recovery_grace_remaining`/`consume_grace_step()` | **不够用,语义不对(见三)** |
| 介入结束后的进展判定 | `meaningful_change_since_intervention`(基于`local_state_change_proxy>=0.5`,在patch_duration_steps到期时评估一次) | **够用,已存在,且正是你要的"REFLECT后是否恢复"判据本体** |
| repeated_action | 连续值,窗口内max | 够用,但当前判定"重复卡住"用的是更严格的`consecutive_exact_action_repeat`,不是`repeated_action` |
| repeated_observation | 连续值,窗口内max | 够用,但已不用于任何决策门控(见上一份信号盘点报告) |
| novelty/progress | `progress_signal`/`information_gain_proxy` | 够用,未直接用于本设计 |
| cooldown | `cooldown_remaining` | 够用,已存在 |
| max_interventions | `intervention_count` vs `config.max_interventions` | 够用,已存在 |

**关键发现:你设想的"REFLECT后观察是否恢复,未恢复则升级REPLAN"这个机制,在代码里已经存在,而且是控制器最早期就有的通用机制(`_ESCALATION`字典 + patch-duration到期评估),不是需要新造的东西——只是当前对"首次重复卡住"这一具体情形,从未真正进入这条既有升级链,因为REPLAN在武装REFLECT之前就抢先了。**

---

## 三、`recovery_grace_steps=3`是否真的承担"观察恢复"的作用

**不是,当前实现只用于禁止旧版exhausted_repeated提前终止,不参与controller自己的恢复判定。**

`recovery_grace_remaining`只在`agents.py`里被读取,用来在武装一次intervention后的若干步内,抑制**agent循环自己的旧版"精确重复→立即终止"检查**——给新武装的directive一个不被legacy机制误杀的窗口。`stateful_controller.py`内部从来不读取`recovery_grace_remaining`,它完全不参与"这次REFLECT是否真的让模型走出循环"这个判断。

**真正承担"观察恢复"职责的是`patch_duration_steps`(当前3)配合`meaningful_change_since_intervention`**——这是一个完全独立于recovery_grace的、controller自己内部的评估窗口,在到期时判定recovered/unresolved并按`_ESCALATION`调度升级。这两个机制此前在设计上就是分开的(grace防止外部误杀,patch_duration+_ESCALATION负责内部升级),你的候选设计天然应该复用后者,而不是grace。

---

## 四、"REFLECT后恢复"/"未恢复"的最小可执行判据(复用现有信号)

**不需要新增评分,直接复用已经存在的机制**:

- **恢复**:在REFLECT武装后的`patch_duration_steps`(当前3步)窗口内,`meaningful_change_since_intervention`变为True(即窗口内任一步`local_state_change_proxy>=0.5`)。这与当前代码对**所有**intervention类型(VERIFY/REFLECT/REPLAN)使用的判据完全相同,无需为REFLECT单独定义新逻辑。
- **未恢复**:到`intervention_expiry_step`时`meaningful_change_since_intervention`仍为False → 已有代码自动标记`unresolved`并通过`_ESCALATION[REFLECT]=REPLAN`调度升级到REPLAN。**这一步现有代码已经完整实现,不需要新造。**
- 你候选设计里提到的"REFLECT后仍重复相同动作"可以作为**额外的、更快的**未恢复信号(不必等满3步),复用已有的`consecutive_exact_action_repeat`——但这就是此前被废弃的PIR(post_intervention_exact_repeat)机制的核心逻辑。**必须向你指出这一点**:你现在设计的"REFLECT后若仍重复则加速升级REPLAN"与已经做过消融并因证据不足以证明净收益而搁置的PIR规则,在机制本质上是同一件事,只是触发范围从"任意pending intervention下的精确重复"收窄成"仅在REFLECT pending下"。如果你想要这部分"加速升级",需要明确知晓这是在复用PIR的核心思想,而不是全新机制。

---

## 五、基于现有134题日志的离线统计(已用修正后的abs_idx方法核算)

**90个满足`frustration_medium AND consecutive_exact_action_repeat`的step中,46个在触发时已有其他intervention处于pending状态(23次prev是replan,18次prev是reflect,5次prev是verify),44个是真正的"首次遭遇"(没有任何intervention pending)。**

对44个"首次遭遇"case(你候选设计里"首次确认repeated_action"最直接对应的子集):

| | 数量 |
|---|---|
| 涉及的不同episode(修正后) | **23** |
| 也同时满足invalid_action(即使不算repeated_action,靠invalid_action本身也该是REFLECT,只是同样被REPLAN抢先) | 41 / 44 |
| 纯粹由repeated_action驱动(invalid_action未同时成立) | 3 / 44 |
| 触发后拥有完整3步观察窗口(未被episode提前终止打断) | 12 / 44 |
| ——其中3步内出现进展证据(`local_state_change_proxy>=0.5`) | **5** |
| ——其中3步内确认无进展 | **7** |
| 只有1-2步观察窗口(episode提前结束,数据不足以判断) | 12 / 44 |
| 完全没有观察窗口(触发step就是episode倒数第1-2步) | 20 / 44 |

**23个episode里,最终成功2个,失败21个**(修正前误报为8成功/9失败,已在开头更正)。

**按任务类型**:examine 8、heat 8、clean 3、puttwo 3、put 1——**examine和heat合计16/23(69.6%)**,再次确认这两个已知最弱的类型是"首次重复卡住"现象最集中的地方。

**7个"确认无进展"case里,4个在3步窗口内还继续出现精确重复**(`still_repeats_within_3=True`),对应你候选设计里"REFLECT后仍重复→升级"的直接触发条件。

**当前REPLAN在这些episode里的实际resolve记录**(全部90-condition的30个episode范围内,已修正):`recovered`出现12次,`unresolved`出现9次——**REPLAN并非总是有效,但也不是总是无效**,两种结果都存在。

---

## 六、事实证据 / 可能变化 / 无法离线确定的结果(严格分类)

- **可直接观察的事实**:44个首次遭遇case里只有12个有完整3步观察窗口;这12个里5个有进展证据、7个没有;23个episode里最终2个成功、21个失败;41/44同时也满足invalid_action。
- **候选路由下可能发生的决策变化**:如果"首次遭遇repeated_action+frustration_high+confidence_low"改为路由到REFLECT,则这44个候选点(23个episode)的候选intervention会从REPLAN变为REFLECT;后续是否按patch_duration_steps到期评估、是否升级到REPLAN,取决于该REFLECT武装后模型的**新**行为轨迹。
- **无法离线确定的结果**:REFLECT武装后模型是否真的会因为收到不同的directive文本而产生不同的动作,从而让`meaningful_change_since_intervention`在这44个case里的实际取值发生变化——这必然会与当前记录的("如果走REPLAN会怎样")真实轨迹不同,offline无法预测。5个"当前REPLAN路径下有进展证据"的case不能反推"如果换成REFLECT也会同样恢复"或者"REFLECT会恢复得更快/更慢"。

---

## 七、候选设计是否会意外改变其他路由

- **invalid_action路由**:候选设计明确要求"保持现有REFLECT路由"。经检查,你的候选只针对"repeated_action driven的首次遭遇"这一条件分支增加例外,不改变`invalid`这个OR条件本身的检查方式——**invalid_action单独触发(不伴随repeated_action)时的行为完全不受影响**。但需要注意:41/44的首次遭遇case本身也同时满足invalid_action,这些case目前也被REPLAN抢先(即invalid_action自己的78.2%被抢占问题依然存在、不受本候选设计影响,因为本设计只处理"repeated_action驱动"这个例外,不改变invalid_action自身的路由规则)。
- **无重复动作的high frustration路由(即"没有repeated_action,直接frustration_high+confidence_low")**:候选设计明确"保持直接REPLAN",不受影响——这是REPLAN分支里除"repeated_action驱动"外的所有其他触发原因(如invalid_action长期累积、observation novelty耗尽等),会继续走原有REPLAN路径。
- **VERIFY路由**:候选设计明确本轮不改,VERIFY完全不受影响。
- **cooldown和max_interventions计数**:两者都通过`_arm_intervention()`的共享bookkeeping实现,不区分武装的是哪种intervention类型——**无论武装REFLECT还是REPLAN,intervention_count递增、cooldown_remaining重置的逻辑完全相同,候选设计不会改变这两个计数机制本身**。

---

## 八、实现这一机制所需的最小逻辑变化(不修改代码,仅描述)

在`_select_intervention()`第546行的REPLAN判断之前,插入一个例外条件:

```
若 frustration_high AND confidence_low AND repeated(consecutive_exact_action_repeat)
   AND 当前没有任何intervention处于pending状态(即"首次遭遇")
则 → REFLECT(而不是REPLAN)

否则(包括:没有repeated但frustration_high+confidence_low成立;
      或repeated但已有intervention pending,即非首次)
→ 保持现有REPLAN判断不变
```

REFLECT武装后走**已经存在**的patch_duration_steps到期评估+`_ESCALATION[REFLECT]=REPLAN`,无需新增升级代码。

这是一个**单一、局部的路由例外**,不交换REFLECT/REPLAN整体优先级(invalid_action驱动的、不含repeated的REPLAN路径完全不受影响),不改变任何阈值数值。

---

## 九、现有日志/状态是否足以可靠判断"REFLECT后是否恢复"——明确指出不足之处

**不完全足够,存在两个真实缺口**:

1. **观察窗口经常被episode提前截断**:44个候选case里只有12个(27%)有完整的3步观察窗口,20个(45%)完全没有观察窗口(触发点就在episode结束前1-2步)。这不是"REFLECT机制"本身的缺陷,而是当前样本里"首次重复卡住"经常发生在episode已经接近max_steps或即将耗尽预算的晚期阶段——**这意味着即使换成REFLECT优先路由,相当一部分case可能根本来不及观察到任何"是否恢复"的证据,episode就已经因为其他原因(如max_steps)结束了**。这本身是一个值得关注的现象,但本报告不做进一步推测。
2. **这一切都是在"REPLAN路径"下产生的历史数据,不是"REFLECT路径"下产生的数据**:我们只能观察到"当前REPLAN之后发生了什么",无法观察"如果当初走的是REFLECT会发生什么"——这是offline反事实分析的根本局限,不是数据缺失,而是因果推断本身的限制,必须通过真实重跑才能填补。

---

## 十、最终判断(二选一)

**判断:现有状态和职责定义已经足以表达你设计的升级链(patch_duration_steps+meaningful_change_since_intervention+_ESCALATION三者组合已经是完整的"武装→观察→未恢复则升级"机制,不需要新造),缺的只是一条局部路由例外(见八)。证据支持进入一次小规模GPU Gate,但必须明确以下限制条件,不能把这次审计的样本量当作对最终效果的预判:**

- 样本规模small且不均衡:全134题里只有23个episode真正命中"首次遭遇"条件,其中仅12个有完整观察窗口,2个最终成功、21个最终失败——**这个失败为主的分布本身不能作为"REFLECT优先会更差"或"REPLAN更合适"的证据**,因为这些都是REPLAN路径下产生的结果,不是对照。
- 41/44的首次遭遇case同时也满足invalid_action,说明这个新路由例外的"净增量"覆盖面本身就相对有限(纯粹靠repeated_action、不靠invalid_action撑住REFLECT资格的只有3个case)。
- examine和heat合计占了69.6%的受影响episode,与本项目一贯观察到的"examine/heat是最弱类型"一致,值得在设计Gate时重点关注这两类的结果,而不只看总分。

到此停止。未修改代码,未运行GPU实验,未同时调整优先级和阈值,未启动134题,未根据旧轨迹推断新路由下的成功率。等待你确认下一步(是否批准八节描述的最小路由例外进入GPU Gate,或者需要先补充其他信息)。
