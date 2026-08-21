# external/ADaPT — provenance

Upstream: https://github.com/archiki/ADaPT.git
Pinned commit: `ecdc4ab0030b4be9be122622d8ea78f8c59c44c4` (2024-01-03)
LICENSE: present in the clone, unmodified.

The clone itself is **not** committed (see `.gitignore`). To reconstruct:

```bash
git clone https://github.com/archiki/ADaPT.git external/ADaPT
cd external/ADaPT
git checkout ecdc4ab0030b4be9be122622d8ea78f8c59c44c4
git apply ../_provenance/ADaPT/ae3_local.patch
```

## What `ae3_local.patch` changes

All edits are confined to `run_alfworld.py` and fall into four groups. Upstream's
`std` / `cross` / `common` react-types still produce **byte-identical** prompts.

1. **Local vLLM instead of the OpenAI API.** The original hard-required a `KEY.txt` and
   used the pre-1.0 module-level `openai.Completion.create`. Replaced with an
   `openai>=1.0` client object pointed at `$OPENAI_BASE_URL`. New `'qwen' in LM.lower()`
   branches are placed *first* in `llm()` and `plan_llm()`, so the original OpenAI call
   sites are unreachable in our runs rather than removed.

2. **alfworld 0.2.2 → 0.4 API drift.** Upstream targets `alfworld==0.2.2`, which
   re-exported `AlfredTWEnv` at the `alfworld.agents.environment` module level and offered
   `init_game(batch_size, game_file=...)`. Neither exists in the installed version. Both
   are worked around exactly as `alfworld_runs_ae/environment.py::_resolve_env_cls` /
   `make_single_task_env` already do for our own pipeline.

3. **Planner format fix (2026-07-27).** A diagnostic audit found `plan_llm()` **never
   once** produced the required `Step N:` / `Execution Order:` format across all 134 tasks
   — it looped the robot capability-list bullets until `max_tokens=400` truncated it, so
   `plan_to_args()` always parsed an empty step list and no decomposition ever happened.
   (The 8/134 successes were single-shot atomic-executor wins, not ADaPT recursion.) A
   12–24-task planner-only A/B isolated two independent causes:
   - constraint lines used `- ` dash bullets identical in shape to the capability list in
     the same prompt → rewritten as plain sentences (18/18 looping → 0/18);
   - the model then free-associated a preamble that hit the `"\n\n"` stop before reaching
     `Step 1:` → the prompt now **ends mid-line** with `"\nStep 1:"`, forcing continuation.

   Together: **18/18 non-empty, correctly-parsed plans.** Constraint *semantics*, few-shot
   content, model, temperature and `max_tokens` are untouched — the change is format only.

4. **New `--react-type oneshot`.** Upstream's `fetch_react_prompt()` is hardcoded
   **two-shot** for every react-type (`d[react_{v}_1] + d[react_{v}_0]`, under a literal
   "Here are two examples." header). Every other arm in this project is one-shot, so a
   two-shot ADaPT executor would turn part of any gap into a demo-count artifact. The new
   branch serves a single demo using the project's per-type index mapping
   (`put`/`examine`→0, `heat`/`cool`/`clean`→2, `puttwo`→1), i.e. the byte-identical
   demonstration ReAct and AE saw.

   Scope note: this affects the **executor** only. ADaPT's planner prompts
   (`alfworld_plan_filled_prompts.json`) are intrinsic to the method — the same way
   Reflexion's reflection template is — and are deliberately left alone.

## Verified claim about the demo pool

ADaPT's bundled ALFWorld demos were checked against ours: all 18 are our demos
**verbatim**, plus one trailing `'\n> think: Task completed!'` line. "Demos consistent with
ReAct" is therefore a checkable statement, not an assumption.
