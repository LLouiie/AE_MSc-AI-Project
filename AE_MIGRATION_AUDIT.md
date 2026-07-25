# AE 项目方向调整 — Stage 0 仓库审计报告

生成时间: 2026-07-25

## 0. 一个前置事实,会影响后续所有阶段的落地位置

物理上有 **两个独立的 git 仓库**,命名和内容对不上,必须先厘清:

| 本地路径 | GitHub remote | 内容 |
|---|---|---|
| `~/projects/reflexion` | `github.com/LLouiie/AE`(noahshinn/reflexion 的 fork,origin 改名) | HotpotQA/FEVER/ALFWorld 的 **AE 自建 agent 栈**(`agents.py`/`llm.py`/`environment.py`,不依赖 langchain),以及本次要调整方向的**旧 Rule Library + affect 排程脚手架**(`consolidation.py`/`schedulers.py`/`run_practice.py`/`run_exam.py`) |
| `~/projects/AE` | `github.com/LLouiie/AE2` | `baselines/ExpeL`(官方 ExpeL AAAI 实现,hydra 驱动)、`baselines/ReAct`(官方 ReAct/wikienv 实现)、`baselines/reflexion`(官方 Reflexion 仓库的另一份 fork,langchain 版,**当前 thesis 里报告的 Reflexion HotpotQA 分数就是这份代码跑出来的**)。顶层 `src/`、`configs/`、`scripts/` 只有 `.gitkeep`,是空脚手架,从未使用。 |

也就是说,你消息里说的"当前仓库"实际横跨两个仓库、三套互相独立的 ReAct/Reflexion 实现。这不是我能替你决定的事,详见文末的确认问题。

---

## 1. 当前仓库结构和执行流程

### `~/projects/reflexion`(GitHub: AE)

```
hotpotqa_runs/          AE 自建栈,当前唯一活跃维护的 HotpotQA 目录
  agents.py                ReactAgent(纯 ReAct) / ReactReflectAgent(Reflexion,继承 ReactAgent)
  llm.py                   AnyOpenAILLM:纯 openai-python 客户端包一层,连本地 vLLM,call_counter 全局计数
  environment.py           DistractorDocstore(本地10篇文档模糊匹配) / QAEnv(旧,langchain gym 包装,已被 run_episode.py 绕开)
  wiki_docstore.py         WikipediaDocstore(真实 MediaWiki API 检索,question-only)
  run_episode.py           **裸单题 ReAct 跑法,无 RulePool、无 reflection、无 scheduler**,docstring 里明写"No inter-episode state"
  run_practice.py          旧方向主入口:ReactReflectAgent 多轮重试 + RulePool + Scheduler.decide() 触发 consolidate()
  run_exam.py              读取 run_practice.py 产出的 rules.jsonl 最后一行,冻结规则后单轮测试
  consolidation.py         RulePool 类(ExpeL 风格 count-based 规则增删) + consolidate() 调 LLM 做 ADD/EDIT/REMOVE/AGREE
  schedulers.py            NeverScheduler / FixedScheduler,decide() 接口里显式写了
                            "# Phase 2 预留: confidence, disagreement, frustration"
                            —— 这是旧方向已经预留、但从未实现的 affect 接口,和你现在想做的
                            frustration/uncertainty 概念是同一条线的延续,只是旧方向打算用在
                            "何时巩固规则"上,新方向要用在"episode 内何时干预"上。
  react.py                 **legacy**,langchain 版 ReactAgent/ReactReflectAgent,已被 agents.py 取代,
                            仍被 test_wiki_docstore.py 等少数测试引用
  test_*.py, prompts.py, fewshots.py, util.py, tests.py, mocks.py  测试/辅助,现状良好
  archive/run_hotpot_qwen32b_sample100/   本 session 之前刚建的存档目录(与本次任务无关,勿动)
  analysis/, notebooks/, results/, runs/, data/   历史产出,不动

alfworld_runs_ae/        AE 适配版,复刻 hotpotqa_runs 同一套模式
  agents.py                 ALFWorldAgent(纯 ReAct,单轮,已经是干净的类,可直接复用)/
                             ALFWorldReflectAgent(Reflexion 多轮 + rules_text 注入)
  environment.py            make_alfworld_env / get_practice_tasks / get_task_type
  run_practice.py            同样耦合 RulePool + Scheduler(import consolidation/schedulers)
  run_exam.py
  (没有 run_episode.py 等价物 —— ALFWorld 目前没有"裸 ReAct 单任务跑法"的入口脚本,
   但类本身 ALFWorldAgent 已经具备,只差一个入口脚本)

fever_runs/               同样模式:agents.py(FEVERReactReflectAgent) + environment.py +
                           run_practice.py/run_exam.py 耦合 RulePool/Scheduler

alfworld_runs/            **原始 Reflexion 论文代码,未做 AE 适配,历史遗留,当前实验不用**
webshop_runs/             同上,未适配,不用
programming_runs/         同上,未适配,不用

client, server            仓库根目录两个 0 字节文件,未跟踪,看起来是误操作产生的空文件,与本任务无关
```

### `~/projects/AE`(GitHub: AE2)

```
baselines/ExpeL/          官方 ExpeL(AAAI 2024)完整实现,hydra 配置驱动
                           agent/expel.py, configs/, envs/, memory/, models/
                           insight_extraction.py + eval.py + transfer_fever.py
                           已跑出本次 5-seed 方差实验(HotpotQA + FEVER),独立于 reflexion 仓库,
                           不依赖 reflexion/hotpotqa_runs 的任何代码
baselines/ReAct/          官方 ReAct(Yao et al.)实现,wikienv.py + wrappers.py,run_hotpot.py/run_fever.py
                           **和 reflexion/hotpotqa_runs/agents.py::ReactAgent 是两份完全独立的 ReAct 实现**
baselines/reflexion/      官方 Reflexion 仓库另一份 fork(langchain 版,pre-2026-07-12 LangChain-removal
                           重构之前的快照),hotpotqa_runs/run_hotpot.py 就是当前 thesis 报告的
                           Reflexion HotpotQA 分数(51% EM, trial 5)的产出代码
data/, notebooks/          小体量评测数据 + 分析 notebook
src/, configs/, scripts/   **只有 .gitkeep,空脚手架,从未使用,可以忽略**
```

### 关键结论:同一个"ReAct"和"Reflexion"概念,现状是三份独立实现

| | ReAct 实现 | Reflexion 实现 |
|---|---|---|
| A | `AE/baselines/ReAct`(官方 wikienv 版) | `AE/baselines/reflexion`(官方 langchain 版,**当前 thesis 用的就是这份**) |
| B | `reflexion/hotpotqa_runs/agents.py::ReactAgent`(AE 自建,run_episode.py 用) | `reflexion/hotpotqa_runs/agents.py::ReactReflectAgent`(AE 自建,run_practice.py 用,仍 import RulePool 但可传空规则) |
| C | `reflexion/hotpotqa_runs/react.py`(legacy,已废弃) | 同上文件内的 ReactReflectAgent(legacy) |

新方向的 controller 需要一个"每步能拿到 hook 的干净 ReAct loop"——B 组的 `ReactAgent.step()` 结构最合适(见第2节),但 A 组才是 thesis 目前实际报告分数的来源。这两者不一致,需要你决定新实验的 A1/A2 baseline 到底接到哪一组代码上,还是干脆重新单独跑一遍来源统一。

---

## 2. 新旧方法的模块映射

| 新方向需要的东西 | 现有代码里最接近的实现 | 备注 |
|---|---|---|
| `StepContext` / 每步 hook 点 | `reflexion/hotpotqa_runs/agents.py::ReactAgent.step()` | 目前 Thought→Action→Observation 是一整块方法,没有对外暴露的 hook;需要重构成"跑一步 + 返回结构化结果"或在外层包一层 |
| `tool_error` 信号 | 已经存在,只是没有结构化输出:`step()` 里 `except Exception as e: ... 'Could not find that page'` | 只需要把这个 except 分支的结果也写进 StepContext,而不是只拼进 scratchpad 文本 |
| `repeated_action` 信号 | **已有现成实现**:`alfworld_runs_ae/agents.py::ALFWorldAgent.run()` 里 `if action == last_action: is_exhausted = True` | 直接复用这个判定逻辑到 signals/repetition.py |
| `observation_novelty` / `evidence_conflict` / `answer_disagreement` | 无现成实现 | 全新代码 |
| `RulePool` / `consolidation.py` / `schedulers.py` | 存在,但明确不再服务新方向 | 移入 legacy,配置关闭 |
| `VERIFY` operator | 无直接对应。最接近的是 Reflexion 的 `reflect()`,但那是"整个 episode 失败后反思",VERIFY 要求是"episode 内针对当前证据做一次定向复核",语义不同,需要新 prompt | 新写 |
| `REPLAN` operator | 无对应,ADaPT 的"executor 卡住时调 planner 分解子任务"是最接近的概念先例,但两个仓库都没有实现过 | 新写,需先读 ADaPT 论文/官方代码 |
| vLLM/actor LLM 调用 | `reflexion/hotpotqa_runs/llm.py::AnyOpenAILLM` | 可直接复用,alfworld_runs_ae 已经在用 sys.path hack 复用它 |
| ALFWorld 环境封装 | `reflexion/alfworld_runs_ae/environment.py` | 可直接复用 |
| Trajectory/checkpoint 日志模式 | `run_episode.py`/`run_practice.py` 的 JSONL append + `done` 集合续跑 + `config.json` 快照(含 git commit) | 模式很干净,新 controller 的日志应该沿用同一套 |

---

## 3. 可以保留、原样复用的代码

- `reflexion/hotpotqa_runs/{agents.py, llm.py, environment.py, wiki_docstore.py, fewshots.py, prompts.py, util.py, run_episode.py}`
- `reflexion/alfworld_runs_ae/{environment.py}`,以及 `agents.py` 里的 `ALFWorldAgent` 部分(纯 ReAct,不含 Reflexion/RulePool 的那部分)
- `AE/baselines/ExpeL`(整体不动,作为旧方向 thesis 章节继续可跑)
- `AE/baselines/reflexion`(整体不动,当前 thesis Reflexion 分数来源)
- `AE/baselines/ReAct`(整体不动)
- HPC job script 模式(vLLM serve + done-marker 分阶段 + Telegram 通知)

## 4. 需要隔离或停用(移入 legacy,配置关闭,不删除)

- `reflexion/hotpotqa_runs/{consolidation.py, schedulers.py, run_practice.py, run_exam.py, test_rulepool.py}`
- `reflexion/alfworld_runs_ae/{run_practice.py, run_exam.py}`(以及 agents.py 里 `ALFWorldReflectAgent` 中依赖 rules_text 的部分,如果新 Reflexion baseline 改用这份实现,需要把 RulePool 依赖摘掉,只留 reflection)
- `reflexion/fever_runs/{run_practice.py, run_exam.py}`(同款耦合,FEVER 不在本阶段 pilot 范围,先原样冻结)
- `AE/baselines/ExpeL` 的 `insight_extraction.py` / practice 阶段(整个 ExpeL 训练+规则提炼管线,作为"旧方向"的一部分标记 legacy,但因为它是官方论文复现,不建议物理挪动,只在文档/README里注明"不再是新实验的一部分")

## 5. 新增模块清单(仅覆盖当前范围:AE / ADaPT / ReflAct / ReflexGrad / ALFWorld pilot)

```
controllers/
    base.py              MetaController Protocol, StepContext/AppraisalState/InterventionDecision dataclass
    ae_appraisal.py       新方法本体

agents/                  ADaPT / ReflAct / ReflexGrad 的 agent 封装(命名待定,取决于选定落地仓库的既有惯例)
    adapt.py
    reflact.py
    reflexgrad.py

interventions/
    verify.py
    replan.py

signals/
    progress.py
    repetition.py          可直接照抄 alfworld_runs_ae 的 is_exhausted 判定
    evidence_conflict.py
    uncertainty.py
    surprise.py
    recovery.py

logging/
    trajectory_logger.py   沿用 run_episode.py 的 JSONL + config.json 快照模式
    intervention_logger.py

configs/
    controllers/
    experiments/

runners/
    run_pilot_alfworld.py  Stage 7 要求的可复现命令入口
```

## 6. Baseline 实现难度和依赖

| Baseline | 现状 | 难度 | 依赖/风险 |
|---|---|---|---|
| ReAct | ALFWorld 侧 `ALFWorldAgent` 已是干净单轮类,直接可用;HotpotQA 侧 B 组 `ReactAgent` 同样干净 | 低,主要是写个不含 RulePool 的入口脚本 | 无 |
| Reflexion | ALFWorld `ALFWorldReflectAgent`、HotpotQA `ReactReflectAgent` 均已存在多轮重试逻辑 | 低,需摘掉 rules_text 耦合(传空字符串即可,不需要删代码) | 需确认是否要与 thesis 现有 Reflexion 分数(来自 AE/baselines/reflexion 官方实现)保持一致口径 |
| ADaPT | 无任何现有代码 | 中,核心是 executor 失败时递归分解子任务再执行 | 必须先读 arXiv:2311.05772 + 官方实现再动手,不能凭本说明猜细节 |
| ReflAct | 无任何现有代码 | 中,持续对比当前 world state 与目标 | 优先级低于 ReflexGrad,论文来自 EMNLP 2025 |
| ReflexGrad | 无任何现有代码 | 高,四个子机制(TODO planning + fast refinement(k=3) + progress gate + slow reflection(m=5)) | 必须先读 arXiv:2511.14584 v4 + 官方代码;是本阶段风险最高的一项 |
| AE(新方法) | 无现有代码,但 signals 层有一半可以复用/借鉴现成信号(tool_error、repeated_action) | 中高,主要风险在信号工程质量而非架构 | 权重/阈值/衰减必须放配置,不能在 test set 调参(你已明确要求) |

## 7. 分阶段改造顺序(按你缩小后的范围)

沿用你给的 Stage 0-7,不做 controlled baseline 相关的 Stage 4:

- **Stage 0**(本文档):审计,不动代码
- **Stage 1**:确认改动已提交 → 给旧 Rule Library 版本打 tag → 新建 `feature/intra-episode-ae` 分支 → 保留 ExpeL/Rule Library 代码不动 → 用配置隔离旧管线
- **Stage 2**:统一 `MetaController`/日志接口,先在 ALFWorld 上跑通"只有 CONTINUE 的空转 controller"(纯粹验证插桩正确,不作为正式 baseline 交付)
- **Stage 3**:实现共用的 `VERIFY`/`REPLAN` operator,确保所有 controller/baseline 复用同一份 prompt
- **Stage 5**(你的编号,略过 Stage 4 受控 baseline):实现 AE controller 本体(多维状态、累积衰减、cooldown、hysteresis、recovery、budget)
- **Stage 6**:按优先级实现 paper baseline —— ReflexGrad(完整版) → ADaPT → ReflAct
- **Stage 7**:ALFWorld pilot 可复现入口脚本(controller 选择/task subset/seed/model/max steps/token budget/output dir/resume 全部可配)

## 8. 预计涉及的文件

新增(全部在待定的目标仓库内):
`controllers/*.py`, `agents/{adapt,reflact,reflexgrad}.py`, `interventions/{verify,replan}.py`, `signals/*.py`, `logging/{trajectory_logger,intervention_logger}.py`, `configs/controllers/*.yaml`, `configs/experiments/*.yaml`, `runners/run_pilot_alfworld.py`

修改(仅限"通过配置隔离旧管线",不改逻辑):
`alfworld_runs_ae/run_practice.py`、`run_exam.py`(如果决定复用这份 ALFWorldReflectAgent 作为新 Reflexion baseline,需要把 `rules_text` 参数改为可选/默认空,不再 import consolidation/schedulers)

不修改:
`consolidation.py`, `schedulers.py`, `test_rulepool.py`, `AE/baselines/ExpeL/*`, `AE/baselines/reflexion/*`, `AE/baselines/ReAct/*`

## 9. 主要风险

1. **两仓库物理归属未定**——`feature/intra-episode-ae` 分支该建在 `~/projects/reflexion` 还是 `~/projects/AE`,两边都各有一部分"能复用的代码",目前无法自行判断,见文末问题。
2. **Reflexion baseline 口径不一致**——thesis 现有分数来自 `AE/baselines/reflexion`(官方 langchain 实现),而新方向最方便挂 controller 的却是 `reflexion/hotpotqa_runs` 里的自建 `ReactReflectAgent`。如果新论文里同时引用两个分数,需要明确注明是两套不同实现,不能混为一谈。
3. **ReflexGrad/ADaPT/ReflAct 尚未读论文**——目前只有你消息里的摘要转述,必须先各自查一遍 arXiv 原文和官方代码再写,否则存在"适配版误称为精确复现"的风险(你已明确禁止)。
4. **ALFWorld 环境本身尚未在本次 session 验证可运行**——`alfworld_runs_ae` 依赖 `alfworld-download` 和 `ALFWORLD_DATA` 环境变量,需要先跑一次最小 smoke test 确认环境依赖仍然健康,再开始 pilot。
5. **GPU 排队/walltime 风险**——本周 HotpotQA/FEVER 多 seed 实验已经因为 RTX6000 排队和 16 小时 walltime 吃了两次亏,ALFWorld pilot 的 job script 需要按这次的教训预留更充足的 walltime。
6. **sys.path hack**——`alfworld_runs_ae/agents.py` 目前用 `sys.path.append('../hotpotqa_runs')` 复用 `llm.py`,新模块如果也这样做,要避免和新增的 `controllers/`、`signals/` 包路径冲突。

## 10. 第一阶段最小可运行版本

ALFWorld 单任务 ReAct 入口脚本(复用现成的 `ALFWorldAgent`,仿照 `hotpotqa_runs/run_episode.py` 的裸跑法,不含 RulePool/Scheduler)+ `MetaController` Protocol + AE controller 骨架(先允许所有权重为 0,等价于永远 CONTINUE,用来验证插桩管线本身正确)。在 10 个任务上跑通:
- controller 能正确拿到每步 StepContext
- 日志格式(JSONL + config.json)符合 Stage 7 要求的可复现命令
- token/调用成本被正确记录

跑通后再逐步把 AE 的真实信号计算和 REPLAN/VERIFY 接上,而不是一次性把所有信号都写好再测试。

---

## 需要你决定,我无法自行判断的问题

1. **`feature/intra-episode-ae` 该建在哪个仓库?**
   - A. `~/projects/reflexion`(GitHub: AE)—— 旧 Rule Library 脚手架、AE 自建 ReAct/Reflexion 实现、ALFWorld 适配都在这里,物理上改动最小
   - B. `~/projects/AE`(GitHub: AE2)—— thesis 目前实际使用/汇报的 ReAct/Reflexion/ExpeL 官方实现都在这里
   - C. 两边都建同名分支,新代码只放一份,另一边通过某种方式引用(会引入跨仓库依赖,不推荐但列出选项)

2. **新的 Reflexion baseline 用哪一份实现打底?**
   - 沿用 `AE/baselines/reflexion`(官方 langchain 版,和 thesis 现有分数口径一致,但重构出 hook 点成本更高)
   - 还是改造 `reflexion/hotpotqa_runs/agents.py::ReactReflectAgent`(自建版,加 hook 容易,但会和 thesis 现有 Reflexion 分数不是同一份代码跑出来的)

3. **`client`/`server` 两个 0 字节文件要不要顺手删掉?**(与本任务无关,只是审计时顺带发现的疑似误操作残留)
