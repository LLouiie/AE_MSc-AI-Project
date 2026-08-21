# 134题正式跑 — sustained-only路由公式(REPLAN仅在steps_since_meaningful_change>=patch_duration_steps时武装)

Run 2026-08-06,基于`ae_full_cooldown5_candidate`(102/134)那次运行的完全相同代码底座+配置,仅将
`stateful_controller.py::_select_intervention()`里`frustration_high and confidence_low`分支的判定逻辑
改为sustained-only版本(见下方"单变量一致性核实"),其余全部冻结。

## 运行前核对

- git分支:`feature/intra-episode-ae`,working tree未commit(项目全程惯例)
- 实际加载配置:`configs/controllers/ae_tune_dev_cooldown5.yaml`——`warmup=3 grace=3 patch=3 max_interventions=3 cooldown=5 pir_enabled=False`,与上一轮134题**完全相同的yaml文件**(SHA256核对一致,未改动)
- GPU:NVIDIA A40(gpuvm35,SLURM job 270382),与上一轮同一张卡、同一vLLM服务
- 模型:Qwen/Qwen3-8B,vLLM 0.25.1,temperature=0,与上一轮相同
- 134题清单:同一份`data/alfworld/alfworld_tasks_suffix.json`,seed=42,practice(100)+exam(34)顺序固定——逐题核对**env_name顺序134题0处不一致**,确认数据顺序未变
- 运行目录:`ae/runners/runs/ae_full_sustained_v1_a40_practice`(100题)、`ae/runners/runs/ae_full_sustained_v1_a40_exam`(34题)
- 日志异常检查(calls/steps≤0、intervention_count>3、success与termination_reason矛盾):134题逐一核查,**0处异常**

## 零、单变量一致性核实(核心要求,先于结果呈现)

用运行前一天冻结时保存的完整`git diff`补丁(`/vol/gpudata/jy625-ae-data/logs/freeze_nopir_v2_5_tracked.patch`,即上一轮134题运行所用的确切代码状态),精确重建出`stateful_controller.py`/`ae_full.py`/`run_alfworld.py`/`environment.py`当时的字节级内容,逐文件与当前版本做diff:

| 文件 | 结果 |
|---|---|
| `ae/controllers/stateful_controller.py` | **唯一差异**是`_select_intervention()`里这一处REPLAN判定(完整unified diff见下),其余逐字节相同 |
| `ae/baselines/ae_full.py` | 逐字节相同,0差异 |
| `ae/runners/run_alfworld.py` | 逐字节相同,0差异 |
| `alfworld_runs_ae/environment.py` | 逐字节相同,0差异 |
| `ae/controllers/signals.py` / `canonical_action.py` / `alfworld_runs_ae/agents.py` | SHA256与冻结记录一致,未变 |
| `ae/controllers/config.py` | 有一处差异,但只是`reflect_first_encounter_repeat_enabled`孤儿字段的声明+yaml加载代码(本轮实施sustained-only时特意保留未清理),本次用的yaml从未设置这个字段,加载后取默认值False且不被任何路由逻辑读取——**对本次实际加载的配置值没有任何影响** |

```diff
         if signature.frustration_high and signature.confidence_low:
-            return (
-                InterventionType.REPLAN,
-                f"frustration_high({state.frustration:.2f}) and confidence_low({state.confidence:.2f})",
-            )
+            if self.steps_since_meaningful_change >= self.config.patch_duration_steps:
+                return (
+                    InterventionType.REPLAN,
+                    f"frustration_high({state.frustration:.2f}) and confidence_low({state.confidence:.2f}) "
+                    f"sustained(steps_since_meaningful_change={self.steps_since_meaningful_change}"
+                    f">=patch_duration_steps={self.config.patch_duration_steps})",
+                )
+            # 未sustained时不在此返回,继续向下判断(可能落到REFLECT)
```

**代码层面可以证明:这次运行与上一轮134题之间,只有这一处路由公式不同。**

**但必须明确指出边界**:代码单变量不等于结果单变量。本项目已反复验证(如grace5三轮44/35/42的波动)temperature=0下vLLM推理会话间存在非bit-identical效应。本次逐题翻转表(见二)里,**abs_idx=1和abs_idx=12两题在旧版本里全程0次intervention武装(`old_count=0`)却在新版本里出现intervention并翻转为失败**——这直接证明这两题的分叉发生在routing公式生效之前,是纯粹的模型行为非确定性,与本次代码改动无关。**因此下面的成功率差异是"版本对比"意义上的描述性观察,不作因果归因**,与上一轮报告对cooldown=5那次差异的处理方式一致。

## 一、overall / practice / exam

| | 旧版本(cooldown5, REPLAN无条件) | 新版本(sustained-only) | 差值 |
|---|---|---|---|
| overall | 102/134 (76.1%) | **99/134 (73.9%)** | **-3** |
| practice | 78/100 (78.0%) | 75/100 (75.0%) | -3 |
| exam | 24/34 (70.6%) | 24/34 (70.6%) | 0 |

## 二、六类任务成功率

| 类型 | 旧版本 | 新版本 | 差值 |
|---|---|---|---|
| clean | 26/31 (83.9%) | 25/31 (80.6%) | -1 |
| cool | 20/21 (95.2%) | 19/21 (90.5%) | -1 |
| examine | 8/18 (44.4%) | 7/18 (38.9%) | -1 |
| heat | 14/23 (60.9%) | 15/23 (65.2%) | +1 |
| put | 23/24 (95.8%) | 23/24 (95.8%) | 0 |
| puttwo | 11/17 (64.7%) | 10/17 (58.8%) | -1 |

## 三、逐题翻转表(abs_idx级,基于134题严格一一对应,env_name顺序已核对一致)

| | 数量 |
|---|---|
| 两版本都成功 | 93 |
| 两版本都失败 | 26 |
| 失败→成功(新版本救回) | 6 |
| 成功→失败(新版本翻车) | 9 |
| **净变化** | **-3** |

**失败→成功(6题)**:
| abs_idx | 类型 | 旧终止原因 | 新终止原因 | 新版本intervention构成 |
|---|---|---|---|---|
| 0 | examine | exhausted_repeated | success | reflect×1 |
| 47 | puttwo | exhausted_repeated | success | verify×1 |
| 62 | examine | exhausted_repeated | success | reflect×1 |
| 68 | heat | exhausted_repeated | success | reflect×2 |
| 81 | put | exhausted_repeated | success | verify×1, reflect×1 |
| 125 | heat | exhausted_repeated | success | reflect×1 |

**成功→失败(9题)**:
| abs_idx | 类型 | 旧终止原因 | 新终止原因 | 新版本intervention构成 | 备注 |
|---|---|---|---|---|---|
| 1 | puttwo | success | exhausted_repeated | reflect×1 | **旧版本old_count=0**,分叉在routing生效前,非routing公式导致 |
| 12 | cool | success | exhausted_repeated | reflect×2 | **旧版本old_count=0**,同上,非routing公式导致 |
| 33 | examine | success | exhausted_repeated | verify×0 reflect×2 replan×1 | |
| 37 | clean | success | exhausted_repeated | verify×1 reflect×0 replan×2 | 此前23题Gate里唯一"干净因果"案例(旧first-encounter机制),这次sustained-only下未复现改判(仍REPLAN),且结果翻车 |
| 51 | put | success | exhausted_repeated | reflect×1 | |
| 64 | examine | success | exhausted_repeated | reflect×1 | |
| 69 | examine | success | exhausted_repeated | reflect×1 | |
| 71 | heat | success | exhausted_repeated | reflect×2 | |
| 114 | puttwo | success | exhausted_repeated | reflect×1 | |

完整134行明细见`audit_reports/ae_full_sustained_v1_134_per_task.json`。

## 四、LLM calls / environment steps

| | 旧版本 | 新版本 |
|---|---|---|
| calls总数 | 2377 | 2466 |
| calls均值 | 17.74 | 18.40 |
| steps总数 | 2354 | 2444 |
| calls/success | 23.30 | 24.91 |

## 五、VERIFY/REFLECT/REPLAN真实触发(武装)与恢复情况

### 真实武装次数

| 类型 | 旧版本(REPLAN无条件) | 新版本(sustained-only) | 差值 | 此前反事实重放预测(供参照,非本次结果) |
|---|---|---|---|---|
| VERIFY | 18 | **31** | +13 | 28 |
| REFLECT | 64 | **76** | +12 | 74 |
| REPLAN | 33 | **15** | **-18** | 15 |

**真实运行方向与此前基于旧轨迹的反事实重放高度一致**(REPLAN大幅减少、REFLECT/VERIFY相应增多,REPLAN的15与预测的15完全吻合),但这仍然是**这一次独立LLM会话的真实结果**,不是对反事实重放数字的验证——反事实重放本身此前已被要求只能表述为"旧轨迹上的结构性检查",这里给出的是**真实运行的实际触发分布**,两者性质不同。

### 恢复情况(recovered / unresolved,按intervention出现的outcome评估事件精确计数)

| 类型 | 旧版本 recovered/unresolved (rate) | 新版本 recovered/unresolved (rate) |
|---|---|---|
| VERIFY | 18/0 (100.0%) | 31/0 (100.0%) |
| REFLECT | 49/14 (77.8%) | 58/17 (77.3%) |
| REPLAN | 21/10 (67.7%) | **5/10 (33.3%)** |
| **合计** | 88/24 (78.6%) | 94/27 (77.7%) |

总体恢复率基本持平(78.6%→77.7%),VERIFY/REFLECT的恢复率也基本持平(说明sustained-only改动确实没有影响这两个分支的行为,符合"只改一处判断"的设计意图)。

**但REPLAN的恢复率从67.7%骤降到33.3%**——样本量小(15次),需谨慎解读,但一个合理的结构性解释是:sustained-only版本下,能真正武装REPLAN的15次都是"确实持续无进展≥patch_duration_steps"的案例,即**筛掉了容易被局部修复(REFLECT本可处理)的那些"假REPLAN",剩下的都是更顽固、更难恢复的真正计划级卡死**——REPLAN触发次数少了,但触发时机"更难"了,这与恢复率下降的方向是自洽的,但**这是描述性观察,不是本次验证过的因果结论**,需要更大样本或专门设计的对比才能确认。

## 六、intervention_count分布 / termination_reasons

| intervention_count | 旧版本 | 新版本 |
|---|---|---|
| 0 | 69 | 67 |
| 1 | 31 | 32 |
| 2 | 18 | 15 |
| 3(打满预算) | 16 | 20 |

termination_reasons:旧`{success: 102, exhausted_repeated: 23, env_done_without_success: 6, max_steps: 3}`,新`{success: 99, exhausted_repeated: 22, env_done_without_success: 8, max_steps: 5}`。

## 七、结论

1. **代码单变量已证明**:与上一轮134题之间,`stateful_controller.py`只有这一处REPLAN判定不同,其余相关文件(ae_full.py/run_alfworld.py/environment.py/signals.py/canonical_action.py/agents.py/配置yaml)逐字节或SHA256确认无差异。
2. **结果差异不作因果归因**:总成功率99/134 vs 102/134(-3),逐题翻转表里至少2个"成功→失败"的翻转(abs_idx=1、12)在旧版本里从未武装过任何intervention,证明分叉发生在routing公式生效之前,属于run间LLM/vLLM非确定性,不是本次代码改动导致——这与本项目一贯经验(grace5三轮44/35/42波动)一致。**因此本次99/134只能作为"这一次sustained-only版本真实运行"的描述性记录,不能断言sustained-only比原REPLAN无条件版本更差或更好。**
3. **真实触发分布方向符合预期**:REPLAN从33降到15,REFLECT从64升到76,VERIFY从18升到31——这是sustained-only公式在真实运行下确实按设计生效的直接证据(REFLECT不再被同一次repeated_action瞬时触发的REPLAN截胡)。
4. **REPLAN恢复率下降(67.7%→33.3%)值得后续关注**,但样本量小(10 vs 30),且这次只有一轮运行,不构成可下结论的证据。

若要判断sustained-only版本是否真的优于/劣于原版本,需要按本项目已确立的重复验证惯例(多轮独立运行,如dev60三轮复验)才能把"代码差异"和"run间噪声"分开——本次只是单轮真实结果记录,到此停止。
