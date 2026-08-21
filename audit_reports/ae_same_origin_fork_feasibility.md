# "同一起点分叉实验"可行性审计(只读,无代码修改,无大规模实验)

本报告只做可行性验证,包含少量本地、无GPU、无LLM调用的确定性检验(环境确定性验证、controller精确重放),用于回答问题本身,不构成"大规模实验"。

---

## 一、ALFWorld能否在触发前保存/恢复完整环境状态 —— 不能,已代码级+实测双重确认

**代码事实**:项目实际使用的环境包装链是`AlfredTWEnv`(顶层)→ `AlfredDemangler`/`AlfredInfos`/`AlfredExpert`(均继承自`textworld.core.Wrapper`)→ 底层`PddlEnv`。

- `textworld.core.Wrapper.copy()`:`raise NotImplementedError()`(硬编码,未被任何ALFWorld子类重写)。
- `textworld.envs.pddl.pddl.PddlEnv`:未实现`copy()`/`save_state()`/`get_state()`等任何快照方法。
- 唯一在textworld库里真正实现了`copy()`的是`textworld.envs.tw.TextWorldEnv`(非PDDL版本,ALFWorld不用这个类)。

**实测确认**(本地脚本,无GPU、无LLM,读取一个真实gamefile跑`env.reset()`):
- `env.copy()` → `AttributeError`(实际封装对象`TextworldBatchGymEnv`根本没有这个方法)。
- `copy.deepcopy(env)` → `TypeError: cannot pickle 'generator' object`(环境内部持有不可深拷贝的generator对象,直接deepcopy失败)。

**结论:环境快照/恢复在当前ALFWorld/TextWorld版本下不可行,没有可靠的绕过方法。**

## 二、除环境外还需要保存哪些状态

| 状态 | 是否需要 | 说明 |
|---|---|---|
| StatefulController全部字段(state/hysteresis/pending_intervention/cooldown_remaining/intervention_count/…) | 需要 | 全是float/int/bool/str/list等原生类型,**天然可深拷贝或从`ae_step_log`逐字段精确重建**,不是问题所在 |
| Agent侧`recent_actions`/`recent_observations`滑动窗口 | 需要 | 由`alfworld_runs_ae/agents.py`主循环维护,不在controller内 |
| 完整LLM prompt历史(ReAct轨迹文本) | 需要 | 分叉后两个分支都要从同一段历史文本继续,否则prompt本身就不同,不是"同起点" |
| `call_counter()`全局计数器 | 需要 | 需要在分叉点记录基准值,后续按分支分别计算增量,不能用同一个全局计数器累加两个分支 |
| Python侧随机数状态 | **不需要** | 已确认controller/signal extraction全程无`random.*`调用,`run_alfworld.py`里的`random.seed()`只用于任务顺序,不影响单题内部决策 |

## 三、能否保证两分支分叉前完全一致,唯一差异是注入的干预类型 —— 可以,但必须用"重放"而非"复制"实现

由于环境不能复制,唯一可行路径是:**从episode开头,用历史日志里记录的真实动作序列逐步`env.step()`重放(不调用LLM),直到分叉点为止**。

**实测确认环境确定性**(本地脚本,无GPU、无LLM):对同一gamefile,独立创建两个env实例,喂入完全相同的4步动作序列,**两次得到的(observation, admissible_commands, reward, done)逐字节完全一致**。这证明"重放到同一状态"这条路径是可靠的——ALFWorld的PDDL状态转移是纯确定性函数,不依赖环境自身的随机性。

Controller状态同理:整个决策链(信号提取→状态更新→signature→`_select_intervention`)对给定的(action, observation, admissible)输入是纯函数,无内部随机性,重放到分叉点也能精确复现。

**唯一需要注意**:分叉点之前的动作序列必须来自**同一个config**下产生的历史日志(本次用`cooldown5`主候选:`warmup=3 grace=3 patch=3 cooldown=5 max=3`),不能借用其他cooldown/patch值跑出来的旧日志——那些日志的controller状态轨迹在别的参数下会不一样,不能直接套用。

## 四、如果不能snapshot,用seed+重放能否恢复到同一状态并验证一致 —— 已验证可行(见三)

不仅理论可行,已经用真实gamefile实测验证:两次独立重放同一动作序列,`observation`/`admissible_commands`逐字节相同。

## 五、temperature=0下的模型非确定性如何控制

本项目本轮之前的dev60三轮复验已经反复证明:即便temperature=0,vLLM在不同推理会话间存在批处理引入的非bit-identical效应,同一配置三次独立运行的成功率可以有明显波动(如grace5三轮44/35/42)。**本实验分叉后的两个分支各自都要独立调用真实LLM继续跑完episode,因此同样存在这个问题,不能只跑一次就下结论。**

**建议**:每个分支从同一分叉点独立重复**5次**(不是1次)。理由:
- 分叉后的续跑比完整episode短(从分叉step到episode结束,不是从头开始),单次成本低,可以负担比"完整134题3轮复验"更多的重复次数;
- 5次足以观察是否存在类似grace5那样的"单次领先、重复后打平或反转"的不稳定模式,同时不会让GPU成本膨胀太多(见七)。

## 六、现有134题日志中,哪些触发点可以被可靠重建

用**精确条件**(不是候选信号,是"旧逻辑真正武装REPLAN 且 新逻辑真正武装REFLECT,两者基于完全相同的历史前缀重放"):对`cooldown5`主候选134题日志的全部23个候选episode(44个候选step)逐一用真实controller代码重放两次(一次装旧逻辑配置,一次装新逻辑配置,均用历史信号精确驱动),结果:

**只有2个可靠重建的分叉点:**

| abs_idx | step | 旧逻辑实际武装 | 新逻辑实际武装 |
|---|---|---|---|
| **37** | **22** | REPLAN(`frustration_high(1.00) and confidence_low`) | REFLECT(`first_encounter_repeated_action_before_replan`) |
| **82** | **11** | REPLAN(`frustration_high(1.00) and confidence_low`) | REFLECT(`first_encounter_repeated_action_before_replan`) |

其余42个候选step(44减2)全部在"武装"这一步就不满足"旧武装REPLAN、新武装REFLECT"的双重要求——绝大多数原因是"condition persists, no new event"(severity在更早的step已经因为`frustration_medium+repeated`这个既有分支被推到REFLECT级别,到达`frustration_high`时不算新事件,新旧两套逻辑在这些点上其实**都不会武装**任何新东西,不构成有效分叉),其余是cooldown/warmup/max_interventions抑制。

**已用exact new-code条件(`frustration_high AND confidence_low AND repeated AND not-pending`)重新扫描过全部2354个真实step,确认没有遗漏任何因`frustration_medium`滞回边界效应(hysteresis导致medium和high不同步的10个真实样本)而被漏掉的候选——44个候选、2个可靠分叉点是完整、精确的结果,不是抽样。**

## 七、可行性结论 + 最小实验实现方案

**同状态恢复:技术可行,但必须通过"重放"实现,不能用任何形式的环境快照/复制(不存在)。**

**最小实验实现方案**(仅描述,不实施,不修改生产代码):

1. 写一个独立的、不改动`_select_intervention()`的小型测试脚本(类似本轮`test_reflect_first_encounter_repeat.py`的精神,但驱动真实env+真实LLM,不是纯controller单测)。
2. 对每个确认的分叉点(abs_idx, step):
   - 用`make_single_task_env`+`AnyOpenAILLM`重新创建env和LLM client;
   - 从episode开头,把历史日志里记录的真实动作**逐步喂给真实env**(`env.step(action)`,不调用LLM),同步用真实controller驱动到分叉点前一步——这一步完全复用真实的`ae_full_baseline.run_episode`内部逻辑,只是动作来源换成"读历史日志"而不是"问LLM";
   - 到分叉点这一步:**不经过`_select_intervention()`**,直接调用`controller._arm_intervention(InterventionType.REPLAN 或 REFLECT, escalated_from=None)`手动指定分支,跳过路由判断本身(这是刻意设计——本实验目的是比较"两种directive文本对同一状态的即时效果",不是测试路由逻辑,路由逻辑本身已经在23题Gate里验证过);
   - 从这一步开始转为真实LLM调用,让agent正常跑完episode。
3. 每个(分叉点, 分支)组合重复5次,用不同的推理会话(重新建LLM client实例)。

**这个方案的关键性质**:两个分支在分叉点之前的env状态、controller状态、prompt历史**逐字节完全相同**(因为都来自同一段确定性重放,从未调用过LLM),分叉点之后的唯一差异就是收到的directive文本(REFLECT模板 vs REPLAN模板)——这是你要求的"唯一差异是注入的干预类型"的精确实现。

## 八、可靠触发点数量及abs_idx/step

**2个:abs_idx=37(step=22)、abs_idx=82(step=11)。**

## 九、每分支建议重复次数及理由

**每个分叉点、每个分支5次,共2(分叉点)×2(分支)×5(重复)=20次续跑。** 理由见五:单次结果不可靠(已有先例证明),5次是在"能观察到波动模式"和"GPU成本可控"之间的折中,与本项目已确立的重复验证惯例(dev60用3轮)在同一量级,略多是因为单次续跑比完整episode便宜。

## 十、预计GPU成本

- 分叉点重放阶段:纯env step,无LLM调用,成本可忽略(每分叉点最多22步,本地/GPU均可,不消耗vLLM推理资源)。
- 真实续跑阶段:20次续跑,每次续跑长度上限是`MAX_STEPS(50) - 分叉step`,按历史该题的平均calls估算(idx=37历史约35calls,idx=82历史约15-20calls),**保守估计总LLM calls在400-700次之间**——远小于一次134题全量跑(约2300+calls),用同一张A40、同一vLLM服务即可,预计总耗时数十分钟量级(基于本轮23题Gate约几分钟的经验外推)。

## 十一、如何判定REFLECT相对REPLAN有效

**分层判定,不合并成一个数字:**

1. **即时恢复层**(复用已有的`meaningful_change_since_intervention`判据,不新造评分):分支REFLECT在patch_duration_steps内是否出现`local_state_change_proxy>=0.5`,对比分支REPLAN是否同样出现——这是"即时恢复能力"的直接、可复用判据。
2. **episode最终结果层**:分支各自续跑到episode结束的success/failure、总calls、总步数、intervention_count——**这一层能看到完整episode结果,但样本只覆盖这2个具体的卡住场景,不能当作REFLECT/REPLAN在全部134题分布上的效果预测**,必须在报告中明确写清这个边界。
3. **稳定性层**:同一分支5次重复之间的方差——如果单个分支5次结果本身就分散(如既往grace5的案例),则该分叉点的比较结论不可信,需要如实报告"数据不足以判断",不能只挑对自己有利的一次。

## 十二、边界说明(必须明确,直接回应你的问题7)

**这个实验测到的核心是"REFLECT与REPLAN两种directive文本,在完全相同的卡住状态下,对模型后续行为的即时/短期影响"。** 由于续跑一路跑到episode结束,它**也能观察到**完整episode的成功/失败,但由于:
- 样本只有2个具体的卡住场景(都恰好来自examine/clean类型的特定物品-位置组合),
- 每个场景背后是这134题里极少数、结构上比较特殊的情形(都经过severity-upgrade edge-detection的多重条件才被筛选出来),

**这个实验的结果不能外推为"REFLECT整体上比REPLAN更适合处理重复卡住"这类全局结论,只能回答"在这2个具体已知的历史卡住状态下,REFLECT和REPLAN分别导致了什么"。** 如果需要更有代表性的证据,仍然需要依赖之前23题Gate那种"真实分布采样"路径,而不是本实验这种"精确复现单点"路径——两者目的不同,不能互相替代。

---

到此停止。未修改代码,未运行大规模实验,未调整frustration/confidence公式。等待你确认是否按第七节方案实施(仅2个分叉点×2分支×5次重复,约400-700次LLM calls)。
