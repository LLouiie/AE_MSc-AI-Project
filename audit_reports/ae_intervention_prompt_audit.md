# REFLECT/REPLAN/VERIFY 三套干预prompt只读审计

只读审计,未修改任何代码/配置/prompt,未运行GPU。全部真实案例取自当前sustained-only版本的
134题运行日志(`ae/runners/runs/ae_full_sustained_v1_a40_{practice,exam}/episode_log.jsonl`)。

---

## 一、三套prompt完整原文及文件位置

文件:`ae/controllers/intervention_renderer.py`

```python
_DIRECTIVES = {
    InterventionType.VERIFY: (
        "Re-check the current goal and latest observation.\n"
        "Identify what information is still missing.\n"
        "Prefer one information-gathering action before changing the whole strategy."
    ),
    InterventionType.REFLECT: (
        "Diagnose why the recent actions did not change the environment.\n"
        "Do not repeat the same failed action.\n"
        "Choose a corrected action from the current environment state."
    ),
    InterventionType.REPLAN: (
        "Restate the task objective.\n"
        "Discard the current ineffective strategy.\n"
        "Construct a short plan of 2-4 subgoals from the current state.\n"
        "Execute only the next subgoal now."
    ),
}
```

实际展示时外面包一层固定头:`f"[ACTIVE CONTROL DIRECTIVE]\n{body}"`(`render_directive()`)。

三套prompt本身都是**静态模板**,不含任何按次填充的变量(不引用具体是哪个动作重复、哪个observation、当前第几步)。

---

## 二、注入条件、注入位置、持续步数

代码位置:`alfworld_runs_ae/agents.py` 第311-335行(`ALFWorldAgent.run()`主循环)。

- **注入条件**:`self.controller.active_directive()`只要`active_patch_type is not None`就返回对应文本,否则返回`""`。`active_patch_type`由`stateful_controller.py::_arm_intervention()`在一次intervention武装时设置。
- **注入位置**:拼在**当前这一步生成用的prompt末尾**——`prompt_for_llm += f"\n\n{directive}\n"`,在完整trajectory(任务描述+全部历史)之后、模型即将生成本步Thought/Action之前。**directive从不写入持久化的`history`/`trajectory`**——每一步都是临时拼接、用完即弃,下一步prompt从`history`重新拼出时不含它(这点`intervention_renderer.py`模块docstring也明确写了)。
- **持续步数**:由`active_patch_remaining`控制,武装时置为`config.patch_duration_steps`(本配置=3),此后每步(只要没有新intervention重新武装)递减1,减到0时`active_patch_type`清空、directive消失。**如果同一意图在窗口内被升级(如REFLECT unresolved后escalate到REPLAN),`_arm_intervention()`会把计数器重置为完整的3步**,即新directive类型接管一个全新的3步窗口,不是接着旧窗口余下的步数走。
- 结论:一次武装的directive,原样不变地展示给接下来最多3次LLM调用(除非提前被更高severity的新武装覆盖),不会根据这3步内model的实际反应做任何调整。

---

## 三、每套实际收到哪些变量

一次LLM调用实际收到的完整prompt = `_build_base_prompt()` + `initial_ob` + 全部`history`(Thought/Action/Observation逐轮回放)+ (如有)directive,具体到每类变量:

| 变量 | 是否出现在prompt里 | 说明 |
|---|---|---|
| 原始任务(goal) | **是,但只出现一次** | 在`_build_base_prompt()`里`"Here is the task:\n{goal}\n"`,位于整个prompt的**前部**(两个few-shot示例之后),不会随着对话轮数增多而重复出现。对一个跑到20-30步的episode,directive出现时goal文本可能已经在几千token之前 |
| 当前观察(最新observation) | 是 | `history`最后一轮的`Observation:`行(或还没有任何action时的`initial_ob`) |
| 近期动作与反馈 | 是,且是**全部历史**,不是"近期"窗口 | `_build_trajectory()`把`history`列表**逐条完整回放**(Thought/Action/Observation三元组),只有在总token数超预算时才会从最老的开始丢(先丢reflections,再丢最老的history步骤,`goal`+`initial_ob`+最新几步保证不丢)——但ae_full基线`reflections`参数通常为空(单trial运行,无跨episode记忆库),所以正常情况下这条"逐步丢弃"路径极少触发,大多数episode里完整历史都在 |
| 当前持有物 | **否,没有独立字段** | ALFWorld的observation文本里没有持续的"You are carrying: X"状态行,持有物信息只隐含在过去某一步"You pick up X..."这句observation里,模型要靠读完整历史自己推断,prompt里不存在专门列出"当前手持物品"的结构化变量 |
| admissible actions(可执行动作列表) | **否,完全不出现在prompt里** | `admissible_before`/`admissible_after`只在代码里用于两处:①`action_is_admissible`合法性统计(纯日志用途),②`output_parser.parse_agent_output(raw_generation, admissible_before)`做动作解析时的模糊匹配兜底。**模型从未在任何一次LLM调用里被明确告知"你现在能做哪些动作"**——这是ReAct baseline本身的既有设计(react/reflexion/ae_full三个baseline共用同一套prompt构造逻辑,不是AE新增的缺陷),不是AE干预机制专属问题,但意味着VERIFY/REFLECT/REPLAN也都拿不到这个信息来帮模型做"选一个合法动作"的判断 |
| directive本身 | 视intervention状态而定 | 见上节 |

---

## 四、REPLAN是否真实重新注入了原始任务文本

**没有。** REPLAN的directive原文是`"Restate the task objective."`——这是一句**要求模型自己在输出里复述任务**的指令,不是把原始`goal`字符串再贴一遍到prompt里。原始任务文本全程只在`_build_base_prompt()`里出现一次(见三),REPLAN(以及REFLECT/VERIFY)都不会在展示directive时把它重新拼进prompt。

从134题真实日志抽取全部15次REPLAN武装,查看武装后紧接着的模型输出:**没有一次的Thought文本里出现类似"Plan: 1)...2)...3)..."的显式2-4条子目标列表**,大多数就是普通的一句续写式Thought。仅有2次(见第六节案例5、6)模型的输出**开头明显试图"restate the task"**(如"Okay, let's restate the task: examine the alarmclock with the desklamp..."),但这两次都导致**输出没有遵守"Thought: 一行 / Action: 一行"的强制两行格式**,被解析成了一个无法执行的畸形action(environment返回"Nothing happens.")。

**根本冲突**:`_FORMAT_INSTRUCTION`(在`_build_base_prompt()`里,directive之前就已经写入prompt)强制要求"Respond with exactly two lines: Thought: <one concise reasoning step> / Action: <one environment command>",且LLM调用固定带`stop=["\nObservation:", "\nThought:"]`——这个格式约束对REFLECT/VERIFY的单句诊断类要求是兼容的,但**REPLAN要求的"构造2-4条子目标计划"在结构上根本装不进这一行Thought里**,模型要么忽略这个要求只写一句话(15次里13次是这样),要么试图真的复述/列计划就会撑破两行格式导致输出解析失败(15次里2次是这样,且都发生在"Recovered"和"unresolved"各一次里)。

---

## 五、三套prompt职责是否重叠,模型是否把三者都执行成普通续写

**是,重叠明显,且实际输出确实高度趋同于"普通续写"。**

对比第六节案例:VERIFY指令要求"prefer one **information-gathering** action"(优先做信息收集类动作),但案例1里模型武装VERIFY后紧接着做的是`take pan 1 from stoveburner 2`——这是**直接的任务推进动作**,不是信息收集动作,与VERIFY指令字面要求不符,但也不算"错",只是模型没有区分"这是VERIFY,应该先探查"和"这是普通续写,直接干"。

REFLECT要求"diagnose why + do not repeat + choose corrected action",REPLAN要求"restate + discard + plan subgoals + execute next subgoal"——但两者真实输出在**结构上完全一样**:一句Thought(带一点诊断性措辞,如"...did not result in any observation, which might mean...")+ 一个Action。抽样对比找不出任何可以仅凭输出文本区分"这是REFLECT触发的"还是"这是REPLAN触发的"的结构性特征——**唯一的区别是输入的directive文本不同,模型对这个区别的反应几乎没有可观测差异**。这直接印证了你的假设:三套prompt在实际执行层面被模型统一处理成了"在下一句续写里带一点自我诊断措辞",而不是三种有区分度的干预策略。

---

## 六、真实案例(directive原文 + 模型紧接着的输出)

### VERIFY

**案例1**(`pick_clean_then_place_in_recep-Pan-...` q_index=7,成功)
- 触发:`go to stoveburner 2` → `You arrive at stoveburner 2. On the stoveburner 2, you see a pan 1.`,reason=`uncertainty_high(0.70)`
- directive:`Re-check the current goal and latest observation. Identify what information is still missing. Prefer one information-gathering action before changing the whole strategy.`
- 模型输出:Thought=`"Now I find a pan (1). Next, I need to take it."` / Action=`take pan 1 from stoveburner 2`
- **判断**:未按"信息收集"要求行动,直接推进任务——但结果是好的(成功),说明VERIFY在这里更像是"打断一下、模型继续做它本来就该做的事",而非真正触发了"先核查"的行为。

**案例2**(`pick_and_place_simple-Vase-...` q_index=12,成功)
- 触发:`go to shelf 2` → `On the shelf 2, you see nothing.`
- 模型输出:Thought=`"I have checked many locations but haven't found a vase yet. I should check the remaining shelves (3-12) and sidetable 1 next."` / Action=`go to shelf 3`
- **判断**:这次确实是"信息收集"性质(继续搜索),与VERIFY指令方向一致,但这本来就是任务没找到东西时的默认下一步,不能确认是directive导致的,而非模型自己的常规策略。

### REFLECT

**案例3**(`look_at_obj_in_light-Bowl-...` q_index=1,成功)
- 触发:`go to drawer 3` → `Nothing happens.`,reason=`invalid_action`
- directive:`Diagnose why the recent actions did not change the environment. Do not repeat the same failed action. Choose a corrected action from the current environment state.`
- 模型输出:Thought=`"The action 'go to drawer 3' did not result in any observation, which might mean the drawer is closed or the command was not properly executed. I should check if the drawer is closed and open it if necessary."` / Action=`open drawer 3`
- **判断**:诊断+换动作,基本符合REFLECT指令的字面要求,是三类里遵循度最高的一次。

**案例4**(`pick_heat_then_place_in_recep-Mug-...` q_index=9,失败)
- 触发:`heat mug 1 with microwave 1` → `Nothing happens.`,reason=`invalid_action`
- 模型输出:Thought=`"The mug 1 is already in the microwave 1, but heating it didn't work. Maybe I need to close the microwave 1 first before heating."` / Action=`close microwave 1`
- **判断**:诊断合理、换了动作,行为上遵循REFLECT指令,但这题最终仍失败(该题在本次不属于REPLAN案例,略过其后续)。

### REPLAN

**案例5**(`look_at_obj_in_light-AlarmClock-...` q_index=39,**unresolved**,失败)
- 触发:reflect intervention unresolved,升级为replan
- directive:`Restate the task objective. Discard the current ineffective strategy. Construct a short plan of 2-4 subgoals from the current state. Execute only the next subgoal now.`
- 模型输出(t+1):Thought=`"...I should first take the alarmclock."` / Action=`take alarmclock 1 from desk 2` → `Nothing happens.`
- 模型输出(t+2):**Thought=None**,Action=`"Okay, let's restate the task: examine the alarmclock with the desklamp. The user has tried several steps but hasn't succeeded. Let me figure out what's going on. First, the alarmclock is on desk 2, a..."` → `Nothing happens.`
- **判断**:t+2是模型**唯一一次真的试图执行"restate the task"这个具体要求**,但直接把整段复述文字写进了本该是纯Action的字段(没有`Thought:`前缀),违反两行格式,被环境判定为无效动作。这一intervention最终unresolved。

**案例6**(`pick_heat_then_place_in_recep-Egg-...` q_index=7,**recovered**,失败题但该次replan本身recovered)
- 模型输出(t+1):**Thought=None**,Action=`"Okay, let's see. The task is to heat some egg and put it in the garbage can. So first, I need to find an egg. The user tried checking the fridge but it was closed, then tried opening it again, but may..."` → `Nothing happens.`
- 模型输出(t+2):Thought=`"I need to find an egg. Since the fridge didn't yield results, I'll check other likely locations like countertops, cabinets, or shelves next."` / Action=`go to countertop 2` → 正常观察
- **判断**:t+1同样违反两行格式(浪费一步),但t+2模型自行恢复正常格式并推进,intervention本身被判定recovered——**但这次recovered更像是"模型自我纠正回正常续写"的结果,不是"REPLAN的restate+plan要求被成功执行"的结果**;把长篇复述塞进Action字段本身就是一次无效步骤。

**案例7**(`pick_clean_then_place_in_recep-Knife-...` q_index=38,**unresolved**,失败,该题连续两次REPLAN都unresolved)
- 模型输出(t+1):Thought=`"I have not found a knife yet. I need to check other locations like diningtable 2 and diningtable 3."` / Action=`go to diningtable 2` → `Nothing happens.`
- 模型输出(t+2):**几乎逐字重复同一个Thought**,Action=`go to diningtable 2`(**与t+1完全相同的动作**)→ `Nothing happens.`
- **判断**:这是REFLECT"do not repeat the same failed action"和REPLAN"discard the current ineffective strategy"两条指令**同时被违反**的直接证据——同一intervention窗口内,模型对同一失败动作原样重试了一次。

---

## 七、REPLAN的5次recovered与10次unresolved专项核查

15次REPLAN武装逐一核查后的分类:

| 结果 | 次数 | 窗口内(t+1,t+2)是否出现完全相同的重复动作 |
|---|---|---|
| recovered | 5 | 0/5(均无窗口内逐字重复;2次里t+1格式错误但t+2自行恢复推进,3次是model自然地做出了不同的新动作) |
| unresolved | 10 | **7/10** 窗口内t+1与t+2是完全相同或近乎逐字相同的Thought+Action(见案例7、以及`Pillow`/`Mug`/`Pencil`×2次/`CD`/`Potato`等) |

**核心发现**:unresolved的10次里,大多数(7/10)不是"模型换了个新策略但没做对",而是**模型在同一个3步directive窗口内对已经失败过一次的动作原样重试**——即REFLECT/REPLAN两条指令里都明确要求的"不要重复失败动作"被直接违反。

**逐一核对你提出的三个具体缺失点:失败是否与prompt缺少题意、当前状态、明确的新计划要求有关**：

1. **题意(任务目标)是否缺失**:**不是主因**。10次unresolved案例里,模型的Thought文本几乎每次都能正确复述任务("The task is to heat a potato and put it in the garbagecan"、"I need to examine the alarmclock with the desklamp"等)——goal文本虽然物理上离当前位置很远,但模型显然记得住,没有表现出"忘记题目"的迹象。
2. **当前状态(最新observation)是否缺失**:**技术上不缺失,但信息量不足**。模型看到的最新observation几乎全是`"Nothing happens."`——这是一个**不携带任何诊断信息**的null观察,没有告诉模型"为什么"失败(是位置不对、物体不存在、还是动作语法不对)。当前3套directive都没有针对这种情况补充具体线索(比如明确指出"你刚才的动作`take potato 1 from microwave 1`已经连续失败,原因可能是XX"),模型只能靠自己在一句Thought里猜测原因,猜错就直接重试原动作。
3. **明确的新计划要求是否缺失**:**REPLAN有这个要求,但结构上无法真正执行**(见第四节)。"Construct a short plan of 2-4 subgoals"这句话本身是明确的,但两行输出格式的硬约束让模型没有地方写这个计划——模型要么忽略这条要求(多数情况),要么尝试遵守就破坏输出格式(2次实测)。

**结论**:unresolved的主因更像是**(a) directive是通用静态模板,不携带"具体是哪个动作/哪次失败"的针对性信息,导致模型无法针对性避开刚失败的那个动作**,以及**(b) REPLAN的"多子目标计划"要求与全局强制的单行Thought+单行Action输出格式结构性冲突**——而不是"题意"或"当前状态"这两个变量本身在prompt里缺失。

---

## 八、审计结论与修改建议(仅结论,不实施)

### 结论

1. 三套directive文本本身语义清晰、职责设计上是分层的(VERIFY信息核查 < REFLECT局部纠错 < REPLAN计划级重启),**但都是不含具体案例信息的静态模板**,注入位置固定(prompt末尾)、持续时间固定(patch_duration_steps=3步),期间不随模型的中间反应做任何调整。
2. REPLAN从未真正重新注入原始任务文本,只是要求模型自己复述——而全局强制的两行输出格式(`_FORMAT_INSTRUCTION`+`ACT_STOP`)让"复述任务+列2-4条子目标"这个具体要求在结构上很难被真正执行;实测15次里仅2次模型尝试直接执行这条要求,且两次都因此产出了格式错误、被环境判为无效的动作。
3. 三套prompt的实际输出高度同质化——REFLECT和REPLAN触发后的模型响应在结构上无法区分,VERIFY触发后模型也经常直接做任务推进动作而非信息收集动作,支持"模型把三者都当普通续写处理,只是顺带在Thought里加一句自我诊断措辞"这一判断。
4. REPLAN的10次unresolved里7次的直接可观测原因是:**同一3步directive窗口内,模型对同一个刚失败的动作原样重试了一次**——这直接违反了REFLECT/REPLAN指令原文都写明的"不要重复失败动作"。失败主因不是prompt里缺少题意或当前状态(两者都在,模型也记得),而是directive不携带"具体是哪个动作刚失败"这类针对性信息,且REPLAN的子目标计划要求与输出格式冲突。

### 修改建议(仅列建议,不实施)

- **directive可以携带针对性信息,而不是纯静态模板**:比如在REFLECT/REPLAN的directive文本里,程序化地嵌入"你最近重复的动作是:`{action}`"这样的具体信息(数据已经在`signals.consecutive_exact_action_repeat`判定时确定,只是没有传给directive渲染函数),让模型能明确避开那一个动作,而不是靠自己从历史里推断。
- **重新评估REPLAN的输出格式要求是否可执行**:要么放宽"构造2-4条子目标计划"这一条(改为更贴合单行Thought容量的表述,如"在一句话内明确下一个子目标,而非直接重复上一策略"),要么单独为REPLAN触发的这一步放宽`ACT_STOP`/两行格式限制,允许模型用稍长的输出真正列出计划(但这会改变现有输出解析/token预算逻辑,影响面较大,需要单独评估)。
- **可以考虑给VERIFY的"信息收集动作"要求配合一个更明确的判据**,或干脆承认当前VERIFY在实际执行中就是"轻量提醒",不强求模型必须做纯信息收集动作。
- 以上均为**建议**,均未实施,是否采纳、按什么范围/顺序验证由你决定。
