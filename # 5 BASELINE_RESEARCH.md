# 5 Baseline 调研报告(ReAct / Reflexion / ADaPT / ReflAct / ReflexGrad)

生成时间: 2026-07-25 · 代码调研范围: `reflexion-intra-episode`(`feature/intra-episode-ae` 分支,基于 `legacy/rule-library-2026-07-25` tag)

只调研,未改任何代码。

---

## 1. 每篇论文的核心执行流程

### 1.1 ReAct (Yao et al., arXiv:2210.03629, ICLR 2023)
交替生成 `Thought → Action → Observation`:Thought 负责推理/更新计划,Action 调用外部工具(HotpotQA/FEVER 用 Wikipedia search/lookup,ALFWorld 用环境动作),Observation 是工具/环境反馈。核心贡献是"推理和行动互相支撑"——Thought 帮助 Action 更准确,Observation 帮助 Thought 修正错误认知,从而缓解纯 CoT 的幻觉和错误传播问题。论文本身**不含显式的失败检测或重试机制**——就是单轮跑到 Finish 或步数上限。

### 1.2 Reflexion (Shinn et al., arXiv:2303.11366, NeurIPS 2023)
在 ReAct 之上加一层:一轮 episode 结束(成功或失败/halted)后,若失败,LLM 生成一段 verbal reflection(自然语言诊断"这次为什么没做对"),存入 memory,下一轮把这些 reflection 文本拼进 prompt 里重新跑,如此重复直到成功或达到最大轮数。核心是 **post-hoc、episode 粒度**的反思,不在 episode 内部改变行为。

### 1.3 ADaPT (Prasad et al., arXiv:2311.05772, NAACL 2024 Findings)
Executor 尝试直接执行任务/子任务;执行器自己判断成功或失败(prompt 里显式要求输出"task completed"/"task failed",不依赖外部 gold reward)。一旦判定失败,Planner 把当前任务分解成 3-5 个子任务,并标注逻辑连接符(**And**=需要依次都成功,**Or**=任一子任务成功即可),Controller 递归地对每个子任务重新跑"Executor→(失败则)Planner分解"这套流程,直到子任务成功或达到最大递归深度 `d_max`(ALFWorld/WebShop 默认 3,TextCraft 默认 4)。是**failure-triggered、显式递归分解**,和"要不要反思"无关,单纯是"卡住了就拆解"。

### 1.4 ReflAct (Kim et al., arXiv:2505.15182, EMNLP 2025)
把 ReAct 里的 `Thought`(预测下一步该做什么)替换成 `Reflection`(先陈述当前状态与目标之间还差什么,再决定下一步动作)。论文给出的对照例子:

> ReAct: `Thought: Now I find a spraybottle 2. Next, I need to take it.`
> ReflAct: `Reflection: Currently, I am at cabinet 2 and have found a spraybottle 2, which brings me closer to completing the task of placing it on the toilet.`

即每一步都显式把"当前状态"和"任务目标"关联起来再决策,而不是只预测下一个动作。是**持续、每步粒度**的目标对齐检查,不像 Reflexion 是失败后才触发,也不像 ADaPT 是失败后才分解。

### 1.5 ReflexGrad (arXiv:2511.14584 **v4**)
四机制 dual-process 架构,全部在单个 episode 内运作(**无跨 episode 记忆**,论文原文:"ReflexGrad operates within a single episode without cross-episode learning"):
- **Hierarchical TODO planning**:任务开始时生成分层子目标列表
- **Progress evaluator** `E`:每步用 LLM 打分 `s_t = E(o_t, a_t, o_{t+1}, τ) ∈ [0,10]`
- **Fast process**(TextGrad 式局部修正):每 `k=3` 步跑一次,微调当前策略/prompt,不重新规划
- **Slow process**(causal reflection + 重新规划):当窗口内连续 `m=5` 个分数都低于 `θ_low=4` 时触发,生成因果诊断并重新规划
- **Cooldown**:slow process 触发后设 `c_t=5`,之后 5 步内 fast/slow 都暂停,防止 TextGrad 在计划刚重启、进展分数还没恢复时误伤刚生成的新计划
- 三态路由(Eq. 3):`c_t>0 → COOL`;`c_t=0 且窗口内连续 m 个低分 → SLOW`;否则每 `k` 步 `→ FAST`,其余 `→` 维持原策略不变

ALFWorld(Qwen-3-8B,134题,10 seeds):Zero-shot 35.1%,TextGrad-only 61.2%,Reflexion-only 42.5%,**完整 ReflexGrad 75.4%±2.2**。

---

## 2. 每个 baseline 是否有官方仓库

| Baseline | 官方仓库 | 状态 |
|---|---|---|
| ReAct | [ysymyth/ReAct](https://github.com/ysymyth/ReAct) | 有,notebook 形式 |
| Reflexion | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | 有——**就是当前 `reflexion-intra-episode` 这个仓库的上游**(origin 是它的 fork) |
| ADaPT | [archiki/ADaPT](https://github.com/archiki/ADaPT) | 有,但作者自述"文档尚在完善中"(preliminary) |
| ReflAct | **无**,已确认 | 论文里没有任何"code available at"或 GitHub 链接;搜索也没找到独立官方仓库。**你提到的 MPO 仓库(WeiminXiong/MPO)只是论文用来跑 baseline/环境的代码,不是 ReflAct 官方实现,不能称为官方复现** |
| ReflexGrad | [qpiai/reflexgrad](https://github.com/qpiai/reflexgrad) | 有,结构完整(见下) |

---

## 3. 官方仓库主要入口文件

| Baseline | 入口文件 | 说明 |
|---|---|---|
| ReAct | `hotpotqa.ipynb` / `FEVER.ipynb` / `alfworld.ipynb` / `WebShop.ipynb` | 逐任务 notebook,`wikienv.py`/`wrappers.py` 是支持代码 |
| Reflexion | `hotpotqa_runs/`、`alfworld_runs/` 等(本仓库就是这个上游的 fork,已经很熟悉,见第6节) | — |
| ADaPT | `run_alfworld.py`(另有 `run_textcraft.py`/`run_webshop.py`),ALFWorld 需要额外替换 `alfred_tw_env.py` | Planner/Executor/Controller 的具体类名文档没写全,需要直接读代码确认 |
| ReflAct | 无官方代码,只有论文正文 + Appendix K(prompt 在附录,论文正文没有逐字重印) | — |
| ReflexGrad | `main.py`(CLI:`--model_provider --num_trials --num_envs --run_name --env_type alfworld`) | 见下方组件表 |

ReflexGrad 组件对照(来自仓库调研):

| 组件 | 文件 | 作用 |
|---|---|---|
| Planner | `task_todo_manager.py` | 分层 TODO 分解 |
| Progress Evaluator | `reflexgrad_core_v12.py`, `reflexgrad_trial.py` | 滚动窗口打分聚合 |
| Fast Refiner | `dynamic_prompting.py` | k=3 步 TextGrad 式更新 |
| Slow Reflector | `generate_reflections.py` | m=5 触发的因果推理 |
| Router | `reflexgrad_core_v12.py::ReflexGradCore` | FAST/SLOW/COOL 三态路由 |
| Trial Engine | `reflexgrad_trial.py` | 编排路由与 priority merge(`plan ≻ gradient ≻ base policy`) |
| Env 适配层 | `universal_env_wrapper.py` | 抽象各环境接口 |
| LLM Client | `shared_model.py`(OpenAI)/`shared_model_openrouter.py`/`shared_model_gemini.py`/`shared_model_vllm.py` | 已有本地 vLLM 后端可以直接抄 |

---

## 4. 所需依赖和环境

| Baseline | 依赖 |
|---|---|
| ReAct | `openai` 包,ALFWorld 需另装 `alfworld` |
| Reflexion | 本仓库已有 `reflexion_hotpot`/`alfworld035` conda env,依赖已装好 |
| ADaPT | Python 3.7,`requirements.txt`,ALFWorld 需要替换其自带的 `alfred_tw_env.py`(等于官方仓库对 ALFWorld 底层做了 patch,接入时要弄清这个 patch 具体改了什么,不能囫囵吞枣直接抄) |
| ReflAct | 无独立依赖(论文没放代码),复用我们自己 ALFWorld runner 的依赖即可 |
| ReflexGrad | `requirements.txt` + Docker 支持;`alfworld-download` 拉数据;`shared_model_vllm.py` 已经支持本地 vLLM,和我们现有 `llm.py::AnyOpenAILLM` 的定位一样,可以对照复用其 prompt/evaluator 逻辑但换成我们自己的 client |

---

## 5. 是否原生支持 ALFWorld

| Baseline | ALFWorld 支持 |
|---|---|
| ReAct | 原生支持(`alfworld.ipynb`),本仓库 `alfworld_runs_ae/agents.py::ALFWorldAgent` 已经是照这个模式做的干净复刻 |
| Reflexion | 原生支持(`alfworld_runs/`),本仓库 `alfworld_runs_ae/agents.py::ALFWorldReflectAgent` 同样已复刻 |
| ADaPT | 原生支持,但需要给 ALFWorld 底层打官方提供的 patch(`alfred_tw_env.py`) |
| ReflAct | 论文里有 ALFWorld 实验(134题/6任务类型,93.3% 成功率),但**没有配套代码**,需要完全自己实现 |
| ReflexGrad | 原生支持,是论文的主要 benchmark(134题) |

---

## 6. 与当前项目代码的可复用部分

沿用 Stage 0 审计的发现,结合这次读论文的结果:

- **ReAct**:`alfworld_runs_ae/agents.py::ALFWorldAgent`(单轮纯 ReAct 类)和 `hotpotqa_runs/agents.py::ReactAgent` 的 Thought/Action/Observation 循环结构与论文一致,**核心流程确认吻合,可直接复用,不需要重新迁移官方仓库**。
- **Reflexion**:`hotpotqa_runs/agents.py::ReactReflectAgent` 和 `alfworld_runs_ae/agents.py::ALFWorldReflectAgent` 都实现了"失败→生成反思→下一轮注入→达到 max_trials 或成功为止停止"的完整闭环,和论文核心机制一致。当前 `run_practice.py`/`run_exam.py` 的调用方式额外耦合了 RulePool(`rules_text` 参数),但这个参数**默认可以传空字符串**,不需要改 `agents.py` 本身,只需要新写一个不 import `consolidation`/`schedulers` 的调用入口。⚠️ 注意:这份实现和 thesis 目前报告的 Reflexion 分数(来自 `AE/baselines/reflexion` 官方 langchain 版)不是同一份代码,上次审计已经记录、你也已确认新方向用这份自建版打底,这里只是再强调一次。
- **ADaPT**:当前代码里没有任何"执行器判断失败→分解子任务"的先例,ALFWorld 的 `ALFWorldAgent.run()` 目前是纯单轮执行,没有失败检测和递归分解,需要新写 Planner/Controller,但可以复用 `environment.py::make_alfworld_env` 和 `llm.py::AnyOpenAILLM`。
- **ReflAct**:当前 `ALFWorldAgent._build_base_prompt()` 生成的 prompt 结构(few-shot examples + task)和把 `Thought` 换成 `Reflection` 只是 prompt 模板和一步生成逻辑的改动,底层 env/LLM 调用可以完全复用,改动量小但因为没有官方代码要自己对着论文摘录的例子写 prompt。
- **ReflexGrad**:是四个 baseline 里改动量最大的,当前代码没有 progress evaluator、没有 TODO 分层规划、没有 fast/slow 双进程路由、没有 cooldown,这些都要新写;能复用的只有底层的 `AnyOpenAILLM`/`make_alfworld_env`/日志落盘模式。

---

## 7. 需要新增或修改的模块

沿用 Stage 0 审计的模块清单,细化到这次 5 个 baseline:

```
agents/
    react_baseline.py     薄封装,复用 ALFWorldAgent/ReactAgent,不新写逻辑
    reflexion_baseline.py 薄封装,复用 ALFWorldReflectAgent/ReactReflectAgent,rules_text 固定传空
    adapt.py               新写:Executor 失败判定 + Planner 分解(And/Or) + 递归 Controller
    reflact.py              新写:Reflection-first 单步 prompt(取代 Thought),按论文摘录的例子改写 few-shot
    reflexgrad/
        todo_manager.py     新写:分层 TODO
        progress_eval.py    新写:LLM-judged 0-10 打分
        fast_refine.py       新写:k=3 步 TextGrad 式局部修正
        slow_reflect.py       新写:m=5 连续低分触发的因果反思+重新规划
        router.py             新写:FAST/SLOW/COOL 三态路由 + cooldown

runners/
    run_pilot_alfworld.py  统一入口,--baseline {react,reflexion,adapt,reflact,reflexgrad,ae} 切换
```

`ADaPT`/`ReflexGrad` 需要给"任务是否成功/失败"一个显式判定接口(ADaPT 靠 executor 自报,ReflexGrad 靠 progress evaluator 打分),这两个接口设计上要和后面 Stage 5 的 AE controller 共享同一套 `StepContext`,否则以后没法公平对比。

---

## 8. 预计实现难度

| Baseline | 难度 | 主要工作量来源 |
|---|---|---|
| ReAct | 低 | 只需确认+写统一入口,核心逻辑已验证一致 |
| Reflexion | 低 | 同上,只需摘掉 RulePool 耦合参数 |
| ADaPT | 中 | And/Or 递归分解 + 深度限制,官方仓库文档不全,细节要读代码而非只读文档 |
| ReflAct | 中 | 没有代码可抄,prompt 需要自己按论文摘录的例子 + Appendix K 描述重建,存在"复现不精确"的天然风险,必须标注 paper-based reproduction |
| ReflexGrad | 高 | 四机制协同(TODO + fast + slow + cooldown + 三态路由),官方仓库结构复杂,要先完整读 `reflexgrad_core_v12.py`/`reflexgrad_trial.py` 才能保证路由逻辑一比一 |

---

## 9. 推荐的接入顺序

1. **ReAct**(确认现有实现,写统一入口)—— 半天量级,几乎是核对性质的工作
2. **Reflexion**(同上,去掉 RulePool 耦合)—— 半天量级
3. **ReflexGrad**(优先级最高的新 baseline,和 AE 关系最紧密,必须先吃透避免后续 AE 设计撞车)—— 建议先花时间通读 `reflexgrad_core_v12.py`/`reflexgrad_trial.py`/`dynamic_prompting.py`/`generate_reflections.py` 四个核心文件,再动手写适配版
4. **ADaPT**(有官方仓库兜底,递归分解逻辑相对独立,和 ReflexGrad 不冲突,可以并行或紧接着做)
5. **ReflAct**(没有代码,细节最不确定,适合放最后,趁前面几个 baseline 的 runner/日志框架都搭好之后,用最小改动接进去)

这个顺序和你消息里给的优先级(ReAct/Reflexion 复用 → ReflexGrad → ReflAct → ADaPT)基本一致,唯一的调整建议是 **ADaPT 排在 ReflAct 前面**——因为 ADaPT 有官方仓库可以对照,风险和不确定性更低,先做风险低的能更快验证 runner/日志框架是否搭对,再去处理 ReflAct 这种"没有代码、只能靠论文摘录"的高不确定性项。

---

## 10. Exact reproduction vs Adapted reproduction

| Baseline | 分类 | 理由 |
|---|---|---|
| ReAct | **Adapted reproduction**(但核心流程一致) | 底层用的是本仓库已有的自建 `AnyOpenAILLM` + 本地 vLLM,不是论文原版 `davinci-002` + OpenAI API,且论文用 notebook,我们用脚本;Thought/Action/Observation 循环结构本身与论文一致 |
| Reflexion | **Adapted reproduction** | 同样换了 LLM 后端和调用方式;反思生成→注入→重试的核心机制与论文/官方仓库一致;**且这份实现本身就是官方仓库的 fork 改出来的,不是从零重写**,但因为模型换了(论文用 GPT系列,我们用 Qwen2.5-32B 本地 vLLM),不能称为 exact reproduction |
| ADaPT | **Adapted reproduction** | 有官方仓库可对照,递归分解算法本身可以做到严格一致,但 LLM 后端、ALFWorld 环境 patch 细节、prompt 格式会因为要接入我们自己的 runner/logging 而调整 |
| ReflAct | **Paper-based reproduction**(不是 official repository reproduction,论文里没有代码,只能算按论文文字描述的复现,精度上限低于其他几项) | 无官方仓库,prompt 需要自己按论文摘录例子 + Appendix K 描述重建,这个标注必须在最终报告/thesis 里保留,不能写成"官方复现" |
| ReflexGrad | **Adapted reproduction**,力争核心路由逻辑(FAST/SLOW/COOL 三态 + cooldown + 参数 k=3/m=5/θ_low=4)与 v4 论文严格一致 | 有官方仓库,但要接入我们自己的 vLLM client 和实验框架,论文默认在 GPT-5/Qwen-3-8B/Gemini 上跑,我们要换成 Qwen2.5-32B |

---

## 待确认/风险提示

1. **ADaPT 官方仓库文档不完整**(作者自述 preliminary),Planner/Executor/Controller 的具体类名需要直接读源码确认,不能只信 README。
2. **ReflAct 完全没有代码**,唯一可靠信息来源是论文正文 + Appendix K(prompt 部分),目前只拿到正文摘录的两个对照例子(ReAct vs ReflAct 的 Thought/Reflection 对比),完整的 few-shot demonstration 还需要单独去读 Appendix K 原文再抄。
3. **ReflexGrad 的"Full ReflexGrad" vs 你后续可能要的"scalar progress gate"消歧**:官方仓库的消融实验是 Zero-shot / TextGrad-only / Reflexion-only / Full,**没有**一个现成的"只保留 progress-gate 触发逻辑、去掉 TODO 规划和 TextGrad 精修"的消融版本——如果以后要做这个对照,需要我们自己从 `router.py` 的路由逻辑里摘出来,不是官方仓库直接提供的,做的时候要在文档里明确写清楚这是"我们自己抽出的简化版",不是官方消融结果。
4. 五个 baseline 都要和后面 Stage 5 的 AE controller 共用同一套 `StepContext`/日志格式,这个统一接口设计目前还没定,建议在真正动手写 ADaPT/ReflexGrad 代码前先把 `StepContext` 定下来,避免五个 baseline 各写一套、以后要返工对齐。
