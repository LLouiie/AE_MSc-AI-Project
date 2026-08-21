# 134题正式跑 — cooldown5候选配置(warmup=3, grace=3, patch=3, cooldown=5, max_interventions=3)

Run 2026-08-06,基于`ae_full_nopir_v2_5_frozen`代码底座(见`audit_reports/ae_full_nopir_v2_5_frozen.md`),仅将`cooldown_steps`从3改为5,其余参数/代码/环境全部冻结。

## 运行前核对(全部通过)

- git HEAD: `1d4bb66b22eeb26bad0cc9741f0997cd8a663536`(与冻结记录一致)
- 代码SHA256:`stateful_controller.py`/`config.py`/`signals.py`/`canonical_action.py`/`agents.py`全部与冻结记录逐字节一致
- 实际加载配置:`warmup=3 grace=3 patch=3 max_interventions=3 cooldown=5 pir_enabled=False`
- GPU:NVIDIA A40(gpuvm35, SLURM job 270382),与冻结记录一致
- 模型:Qwen/Qwen3-8B,vLLM 0.25.1,max_model_len=40960
- PyTorch 2.11.0+cu130 / CUDA 13.0,与冻结记录一致
- 134题清单:`data/alfworld/alfworld_tasks_suffix.json`,134条,全部valid_unseen,顺序固定
- 运行目录:`ae/runners/runs/ae_full_cooldown5_candidate_a40_practice`(100题)、
  `ae/runners/runs/ae_full_cooldown5_candidate_a40_exam`(34题),`config.json`快照已由`snapshot_config()`自动保存在各自目录

## 一、overall / practice / exam

| | 结果 |
|---|---|
| overall | **102/134 (76.1%)** |
| practice | 78/100 (78.0%) |
| exam | 24/34 (70.6%) |

## 二、六类任务

| 类型 | 成功 | 总数 | 成功率 |
|---|---|---|---|
| clean | 26 | 31 | 83.9% |
| cool | 20 | 21 | 95.2% |
| examine | 8 | 18 | **44.4%** |
| heat | 14 | 23 | 60.9% |
| put | 23 | 24 | 95.8% |
| puttwo | 11 | 17 | 64.7% |

examine仍是最弱类型(与历史所有版本一致),heat次弱。

## 三、逐题结果、终止原因、intervention次数

完整134行明细见`audit_reports/ae_full_cooldown5_candidate_134_per_task.json`(字段:abs_idx/split/q_index/env_name/task_type/success/termination_reason/intervention_count/llm_calls/environment_steps)。

## 四、LLM calls / environment steps

| | 总数 | 均值 | 中位数 |
|---|---|---|---|
| calls | 2377 | 17.74 | 13.0 |
| steps | 2354 | 17.57 | 13.0 |

calls/success = 2377/102 = **23.30**

## 五、VERIFY/REFLECT/REPLAN数量与恢复率

| 类型 | 次数 |
|---|---|
| VERIFY | 18 |
| REFLECT | 64 |
| REPLAN | 33 |

recovered=88,unresolved=24,**恢复率(recovered/(recovered+unresolved)) = 78.6%**(统一有效分母,方法与dev60阶段一致)。

## 六、intervention_count分布 / 打满预算episode

| intervention_count | episode数 |
|---|---|
| 0 | 69 |
| 1 | 31 |
| 2 | 18 |
| 3(打满预算) | **16** |

termination_reasons:`{'success': 102, 'exhausted_repeated': 23, 'env_done_without_success': 6, 'max_steps': 3}`

warmup_suppressed_count总计:3

## 七、intervention抢占 / pending-at-end / 日志异常

用与dev60阶段完全相同的精确到期步算术方法核查全部134题的所有intervention:

- **TRUE_preemption: 0**(P=3,C=5,满足P≤C+1,理论上不应出现,实测也确认为0)
- resolved_visible: 112
- pending_at_end: 3(episode在到期前正常结束,非异常)

**日志异常检查**(incomplete标记、calls/steps≤0、intervention_count>3、success与termination_reason矛盾):对全部134个episode逐一核查,**0处异常**。运行条件(GPU/模型/代码版本/配置)全程与运行前核对结果一致,过程中未发现任何不一致需要停止的情况。

## 八、与历史134题结果的描述性比较(不做因果归因)

| 版本 | overall | practice | exam | 备注 |
|---|---|---|---|---|
| v1(post_fix_legacy) | 100/134(74.6%) | — | — | warmup修复前 |
| v2(warmup修复) | 97/134(72.4%) | — | — | frustration门控修复前 |
| v2.5(nopir,frustration门控修复,PIR禁用) | 100/134(74.6%) | 78/100 | 22/34 | 本次cooldown5的直接底座 |
| v3(post_intervention_repeat规则) | 93/134(69.4%) | — | — | PIR规则,已弃用 |
| **本次:cooldown5(仅cooldown 3→5)** | **102/134(76.1%)** | **78/100** | **24/34** | |

**明确不将这一差异因果归因于cooldown=5**——这是与v2.5非同批次、非配对的运行,存在本项目反复验证过的run间模型/vLLM会话非确定性(如grace5在dev60重复验证里出现一次明显劣于baseline的情况)。可以客观描述的是:数字上,本次结果比同一份dev60选出的底座版本(v2.5)高2题,方向与dev60上观察到的cooldown5净优势(+5/+4/+1,三次配对全部为正)一致,但134题这一次运行本身不构成独立的因果验证,只能作为一次探索性观察记录。

---

134题正式跑完整报告完毕。参数组合探索(第一/二/三阶段)仍在后台继续进行,不受此次134题结果影响,将在全部完成后一并汇总最终报告。
