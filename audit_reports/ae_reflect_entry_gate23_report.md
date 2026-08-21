# reflect-first-encounter-entry 23题成对Gate报告

## 运行条件确认

同一A40(gpuvm35, SLURM job 270382)、同一Qwen/Qwen3-8B(vLLM 0.25.1, max_model_len=40960)、temperature=0、相同推理参数;两个run都是**本轮新跑**(未复用旧134题日志)。实际加载配置核对:

```
Control:   warmup=3 grace=3 patch=3 max_interventions=3 cooldown=5 pir_enabled=False reflect_first_encounter_repeat_enabled=False
Candidate: warmup=3 grace=3 patch=3 max_interventions=3 cooldown=5 pir_enabled=False reflect_first_encounter_repeat_enabled=True
```

两份yaml逐行diff确认唯一差异就是这一个字段。23个abs_idx完全一致(与manifest固定顺序`[0,8,14,17,32,37,38,47,49,55,60,62,63,67,68,81,82,98,100,106,120,125,133]`一一对应),task-identity交叉核对0错配,均23/23完成,0个incomplete。

## 一、总体success

| | success |
|---|---|
| Control | **5/23 (21.7%)** |
| Candidate | **7/23 (30.4%)** |

## 二、逐题配对

both_win=4,both_lose=15,control赢/candidate输=1(idx=60),candidate赢/control输=3(idx=62,82,100)。

## 三、实际发生路由变化的episode数

**全部23题里,新入口真正"武装"(intervention实际变成REFLECT且reason含`first_encounter_repeated_action_before_replan`)只发生在3个episode:idx=37(第22步)、idx=49(第4步)、idx=82(第11步)。其余20题,新入口的目标条件从未真正触发过一次arm(可能是从未满足候选条件,也可能满足了但被现有cooldown/severity-upgrade等下游门控抑制,与之前离线复核的44 vs 2这一发现一致——真实Gate里活跃次数比44略多但和"真正武装"的2高度接近,3次)。**

## 四、首次分叉前轨迹是否一致(逐题核查,严格区分"模型先分叉"和"controller先分叉")

| idx | 首次分叉step | 分叉前动作是否完全一致 | 分类 |
|---|---|---|---|
| 0 | 21 | 否(step21动作已不同) | 模型先分叉(与新入口无关) |
| 8 | 9 | 否 | 模型先分叉 |
| 14 | 无分叉(全程一致) | — | 无分叉 |
| 17 | 无分叉 | — | 无分叉 |
| 32 | 无分叉 | — | 无分叉 |
| **37** | **22** | **是(前21步逐字一致)** | **唯一"controller先分叉"的干净案例——分叉点正是新入口触发点(ctrl走REPLAN,cand走REFLECT via新入口)** |
| 38 | 7 | 否 | 模型先分叉 |
| 47 | 无分叉 | — | 无分叉 |
| 49 | 1 | 否(第1步动作已不同) | 模型先分叉(新入口在candidate自己已分叉的路径上于第4步独立触发,不能归因于两版对照) |
| 55 | 无分叉 | — | 无分叉 |
| 60 | 7 | 否 | 模型先分叉 |
| 62 | 7 | 否 | 模型先分叉 |
| 63 | 无分叉 | — | 无分叉 |
| 67 | 14 | 否 | 模型先分叉 |
| 68 | 无分叉 | — | 无分叉 |
| 81 | 20 | 否 | 模型先分叉 |
| 82 | 6 | 否(第6步动作已不同,早于新入口第11步的触发点) | 模型先分叉在前,新入口在candidate自己已分叉的路径上独立触发,不能归因 |
| 98 | 5 | 否 | 模型先分叉 |
| 100 | 8 | 否 | 模型先分叉 |
| 106 | 6 | 否 | 模型先分叉 |
| 120 | 19 | 否 | 模型先分叉 |
| 125 | 17 | 否 | 模型先分叉 |
| 133 | 11 | 否 | 模型先分叉 |

**关键结论:23题里,只有idx=37是干净的、前缀完全相同、分叉点正是新入口生效点的因果案例;idx=49和idx=82虽然真正触发了新入口,但触发之前轨迹早已因模型/会话层面的普通非确定性分叉,不能把这两题的最终结果归因于新机制。7个episode(14,17,32,47,55,63,68)全程零分叉,新机制在这些题里完全没有介入过。**

**这直接回应你反复强调的原则——不能把23题的差异笼统归因于新入口:4个翻转题(62,82,100,60)里,只有82沾边新入口,且沾边方式是"在已经分叉的路径上独立触发",不构成干净因果证据;62、100、60这三个翻转与新机制完全无关。**

## 五、REPLAN→REFLECT的实际变化次数 / REFLECT后meaningful change次数 / REFLECT到期升级REPLAN次数

- 新入口真正触发(即REPLAN候选被改写为REFLECT):**3次**(idx 37/49/82,如上)。
- 其中idx=37:候选REFLECT武装后,该intervention最终以`recovered`结束(интervention_count从control的3降到candidate的2,calls从37降到32,双方都success)——**这是本次Gate里唯一一个"干净、可追溯、机制生效"的正向证据:同样成功,candidate少用一次介入、少花5次calls**。
- idx=49:新入口触发后该intervention后续状态未在本次范围内详细追踪出"recovered"记录(episode最终exhausted_repeated失败)——但由于该题在触发点之前就已经因模型分叉走上独立路径,不能反推"是新入口导致了失败"。
- idx=82:新入口触发后episode最终success,但同样因为触发点之前已分叉,不能干净地归因于新入口。

## 六、干预总数及类型

| | VERIFY | REFLECT | REPLAN | 合计 |
|---|---|---|---|---|
| Control | 7 | 25 | 19 | 51 |
| Candidate | 6 | 25 | 13 | 44 |

REFLECT总数持平(25=25),REPLAN总数candidate明显更少(19→13)——与"部分REPLAN被新入口截胡"的设计意图方向一致,但由于上面第四节已证明23题里真正因新入口改变路由的只有3次,REPLAN总数下降的其余部分,更可能来自模型/会话分叉后走出了不同长度/不同性质的轨迹(每个episode本身的介入次数分布不同),不能整体算作新机制的净效果。

## 七、steps、LLM calls、成本指标

| | 总calls | 均值 | 总steps | 均值 |
|---|---|---|---|---|
| Control | 651 | 28.30 | 637 | 27.70 |
| Candidate | 582 | 25.30 | 569 | 24.74 |

总量上candidate更省(-69 calls,-68 steps),但同样受"7题从未分叉+15题模型先分叉"这一构成主导,不能整体算作机制本身的成本收益,只有idx=37那笔"少1次介入、少5次calls"是可干净归因的。

## 八、termination reason

| | Control | Candidate |
|---|---|---|
| success | 5 | 7 |
| exhausted_repeated | 14 | 13 |
| env_done_without_success | 4 | 2 |
| max_steps | 0 | 1 |

## 九、crash / parser failure / 无效动作 / 异常循环

| | parse失败次数 | invalid_action次数 | incomplete episode |
|---|---|---|---|
| Control | 3 | 219 | 0 |
| Candidate | 1 | 169 | 0 |

无崩溃、无incomplete、parse失败率两边都很低,candidate侧甚至更少——**没有发现任何异常或干预失控迹象**。

## 十、按任务类型拆分

| 类型 | n | Control成功 | Candidate成功 |
|---|---|---|---|
| clean | 3 | 1 | 1 |
| examine | 8 | 1 | **3** |
| heat | 8 | 2 | 2 |
| put | 1 | 0 | 0 |
| puttwo | 3 | 1 | 1 |

examine类是唯一有净改善的类型(1→3),但该类8题里,前面第四节已确认唯一"干净"的新入口案例(idx=37)属于clean类型不是examine——examine的3题改善(62,82,100这三个翻转)全部经第四节判定为"模型先分叉,与新入口无关"。**不能把examine类的表面改善说成新入口对examine类特别有效。**

## 十一、分层报告:A组(invalid_action+repeated_action)vs B组(纯repeated_action)

23个abs_idx来自审计阶段"候选条件成立"(不是"真正武装")的分类;而本次Gate里**真正武装新入口的只有3次(idx 37/49/82)**,已经比A/B两组细分更精确——直接看这3次各自属于哪一组:

- idx=37触发时:该step的signals显示`invalid_action`同时也为True(A组模式,与invalid_action重叠)。
- idx=49触发时:同样`invalid_action`同时为True(A组模式)。
- idx=82触发时:同样`invalid_action`同时为True(A组模式)。

**本次Gate里,3次真正触发全部属于A组(与invalid_action重叠),0次是B组(纯repeated_action驱动)。样本太小,不能得出"B组无效"的结论,只能如实报告本次3次触发的构成,不把两组混在一起宣称"重复动作REFLECT有效"。**

---

## 判定

**机制层面:确认按预期工作。** 新入口在本次真实GPU运行里确实触发了3次(证明它不是纯理论上可达,是真的会在真实轨迹里生效),其中idx=37提供了一个完全干净、前缀一致、可归因的正向证据(同样成功,少1次介入、少5次calls)。全程无崩溃、无异常循环、无incomplete、parse失败率不升反降。

**策略效果层面:证据不足以判断,不是"净收益"也不是"净下降",是典型的"胜负接近且证据里掺杂大量与机制无关的噪声"情形。** 总分+2(5→7)、总calls-69,数字方向都正向,但逐题分叉分析显示:23题里20题的最终结果与新入口完全无关(7题零分叉、13题模型先分叉且新入口从未触发或触发得太晚已无法归因),只有1题(idx=37)是干净的机制证据,且是"两边都成功"的效率改善,不是"扭转输赢"的证据。4个翻转题里3个(62,82,100)与新入口无关纯属模型/会话噪声,1个(control赢的idx=60)也与新入口无关。

**按你给的判定规则,这属于"胜负接近或证据矛盾"这一类——已完成逐题分叉轨迹分析(见第四节),不直接启动134题或更大规模验证。**

到此停止,等待你确认下一步(是否需要为idx=37这类"干净触发"案例扩大样本做针对性验证,或者维持现状不再推进这个方向)。
