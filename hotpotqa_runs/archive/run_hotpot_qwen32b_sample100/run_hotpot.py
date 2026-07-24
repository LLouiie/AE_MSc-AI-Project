"""
run_hotpot.py — scripted version of notebooks/ReactQA.ipynb (official Reflexion).

Zero changes to the official code (agents.py / prompts.py / util.py are used
as-is); this file only replaces the notebook driver:

  - LLM: any OpenAI-compatible endpoint (e.g. local vLLM) via a thin wrapper
    passed in as react_llm / reflect_llm. The repo's own AnyOpenAILLM is
    unusable here (langchain 0.0.162 calls the pre-1.0 openai API, but the
    Reflexion env has openai 2.x installed).
  - Data: CLI path (.json list of dicts, or .joblib pandas DataFrame such as
    data/hotpot-qa-distractor-sample.joblib). Only question/answer are used;
    retrieval is the official langchain Wikipedia docstore (live wiki API).
  - Wikipedia requests carry the User-Agent from $WIKIPEDIA_USER_AGENT
    (required; injected into the `wikipedia` package).
  - Checkpoint after every trial to runs/<run-name>/checkpoint.joblib;
    rerunning the same command resumes from the last completed trial.

Usage:
    export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (you@example.com)'
    python run_hotpot.py \
        --data data/hotpot-qa-distractor-sample.joblib \
        --run-name reflexion_qwen32b \
        [--strategy reflexion] [--trials 5] [--max-steps 6] \
        [--model Qwen/Qwen2.5-32B-Instruct] \
        [--base-url http://localhost:8000/v1] \
        [--api completions] [--limit 5]
"""

import argparse
import json
import os
import re
import sys
import time

# agents.py builds default AnyOpenAILLM objects at import time and reads
# OPENAI_API_KEY; the defaults are never called, they just need to construct.
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

import joblib

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data",      required=True,           help=".json or .joblib question file")
p.add_argument("--run-name",  required=True,           help="output directory name under runs/")
p.add_argument("--strategy",  default="reflexion",
               choices=["none", "last_trial", "reflexion", "last_trial_and_reflexion"])
p.add_argument("--trials",    type=int, default=5,     help="number of trials (notebook uses 5)")
p.add_argument("--max-steps", type=int, default=6,     help="max ReAct steps per trial (official default 6)")
p.add_argument("--limit",     type=int, default=None,  help="cap number of questions")
p.add_argument("--model",     default="Qwen/Qwen2.5-32B-Instruct")
p.add_argument("--base-url",  default="http://localhost:8000/v1")
p.add_argument("--api-key",   default="EMPTY")
p.add_argument("--api",       default="completions", choices=["completions", "chat"],
               help="completions: few-shot continuation like the user's other runs; "
                    "chat: single user message like the official gpt-3.5-turbo setup")
args = p.parse_args()

# ── Wikipedia User-Agent (fail fast) ──────────────────────────────────────────
UA = os.environ.get("WIKIPEDIA_USER_AGENT", "").strip()
if not UA:
    sys.exit(
        "ERROR: set the WIKIPEDIA_USER_AGENT environment variable, e.g.\n"
        "  export WIKIPEDIA_USER_AGENT='AE-thesis-project/1.0 (you@example.com)'\n"
        "Refusing to send anonymous requests to Wikipedia."
    )
import wikipedia
wikipedia.wikipedia.USER_AGENT = UA  # used by every _wiki_request header

import tiktoken
from langchain import Wikipedia
import agents as agents_module
from agents import ReactAgent, ReactReflectAgent, ReflexionStrategy
from util import summarize_react_trial, log_react_trial

# Two compatibility fixes, applied here so the official files stay untouched:
# 1. agents.parse_action returns None on malformed actions and the caller
#    unpacks it -> TypeError kills the whole episode. Additionally, Qwen often
#    appends prompt-marker junk after a well-formed action ("Finish[X] (END OF
#    RESPONSE)"), which the official ^...$ regex rejects; that wasted 32% of
#    all steps in a full run. Salvage the leading 'Type[arg]' if present,
#    otherwise route to the official 'Invalid Action' observation branch.
_orig_parse_action = agents_module.parse_action
def _safe_parse_action(s):
    r = _orig_parse_action(s)
    if r is not None:
        return r
    m = re.match(r'\s*(\w+)\[(.+)\]', s)  # greedy: last ']' in the string
    if m:
        return m.group(1), m.group(2)
    return ("Invalid", s)
agents_module.parse_action = _safe_parse_action

# 2. wikipedia.page() defaults to auto_suggest=True, which mangles exact
#    titles ("League of Nations" -> PageError) through langchain's docstore.
_orig_page = wikipedia.page
def _page_no_suggest(*a, **kw):
    kw["auto_suggest"] = False
    return _orig_page(*a, **kw)
wikipedia.page = _page_no_suggest

ENC = tiktoken.encoding_for_model("text-davinci-003")

def dump_checkpoint(path, trial, agents):
    # agent.enc (tiktoken CoreBPE) is not picklable; detach and restore
    for a in agents:
        a.enc = None
    joblib.dump({"trial": trial, "agents": agents}, path)
    for a in agents:
        a.enc = ENC

STRATEGY = {
    "none":                     ReflexionStrategy.NONE,
    "last_trial":               ReflexionStrategy.LAST_ATTEMPT,
    "reflexion":                ReflexionStrategy.REFLEXION,
    "last_trial_and_reflexion": ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
}[args.strategy]

# ── LLM: OpenAI-compatible endpoint behind the AnyOpenAILLM call interface ────
llm_calls = 0

class EndpointLLM:
    """Callable prompt->text wrapper, pickle-safe (client rebuilt lazily)."""
    def __init__(self, model, base_url, api_key, api, max_tokens, stop=None):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.api, self.max_tokens, self.stop = api, max_tokens, stop
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def __call__(self, prompt: str) -> str:
        global llm_calls
        llm_calls += 1
        client = self._get_client()
        if self.api == "completions":
            r = client.completions.create(
                model=self.model, prompt=prompt, temperature=0,
                max_tokens=self.max_tokens, stop=self.stop)
            return r.choices[0].text
        r = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=self.max_tokens, stop=self.stop)
        return r.choices[0].message.content

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_client"] = None
        return state

# official settings: react 100 tokens + stop "\n"; reflect 250 tokens, no stop
react_llm   = EndpointLLM(args.model, args.base_url, args.api_key, args.api,
                          max_tokens=100, stop=["\n"])
reflect_llm = EndpointLLM(args.model, args.base_url, args.api_key, args.api,
                          max_tokens=250)

# ── output dir / config snapshot ──────────────────────────────────────────────
RUN_DIR = os.path.join("runs", args.run_name)
os.makedirs(RUN_DIR, exist_ok=True)
CKPT      = os.path.join(RUN_DIR, "checkpoint.joblib")
TRIAL_LOG = os.path.join(RUN_DIR, "trial_log.txt")
RESULTS   = os.path.join(RUN_DIR, "results.jsonl")

json.dump({**vars(args), "user_agent": UA,
           "data_path": os.path.abspath(os.path.expanduser(args.data)),
           "started": time.strftime("%F %T")},
          open(os.path.join(RUN_DIR, "config.json"), "w"), indent=2)

# ── data ──────────────────────────────────────────────────────────────────────
data_path = os.path.expanduser(args.data)
if data_path.lower().endswith(".joblib"):
    records = joblib.load(data_path)
    if hasattr(records, "to_dict"):
        records = records.reset_index(drop=True).to_dict("records")
else:
    records = json.load(open(data_path))
if args.limit:
    records = records[: args.limit]
qids = [str(r.get("id", i)) for i, r in enumerate(records)]

# ── agents (fresh, or resumed from checkpoint) ────────────────────────────────
start_trial = 0
if os.path.exists(CKPT):
    ck = joblib.load(CKPT)
    agents, start_trial = ck["agents"], ck["trial"]
    assert len(agents) == len(records), \
        f"checkpoint has {len(agents)} agents but --data/--limit gives {len(records)}"
    for a in agents:
        a.enc = ENC
    print(f"resuming from checkpoint: {start_trial} trials done")
else:
    docstore = Wikipedia()
    if STRATEGY == ReflexionStrategy.NONE:
        agents = [ReactAgent(r["question"], r["answer"], max_steps=args.max_steps,
                             docstore=docstore, react_llm=react_llm)
                  for r in records]
    else:
        agents = [ReactReflectAgent(r["question"], r["answer"], max_steps=args.max_steps,
                                    docstore=docstore, react_llm=react_llm,
                                    reflect_llm=reflect_llm)
                  for r in records]

# ── trial loop (same control flow as notebooks/ReactQA.ipynb) ─────────────────
for trial in range(start_trial + 1, args.trials + 1):
    t0, c0 = time.time(), llm_calls
    for i, agent in enumerate(agents):
        if agent.is_correct():
            continue
        print(f"\n=== Trial {trial} | Q{i + 1}/{len(agents)} | qid={qids[i]} ===")
        try:
            if STRATEGY != ReflexionStrategy.NONE:
                agent.run(reflect_strategy=STRATEGY)
            else:
                agent.run()
        except Exception as e:
            from openai import APIConnectionError, APITimeoutError, InternalServerError
            if isinstance(e, (APIConnectionError, APITimeoutError, InternalServerError)):
                # LLM endpoint is down: abort instead of burning through the
                # remaining agents; checkpoint keeps the last completed trial
                sys.exit(f"ABORT: LLM endpoint unreachable ({type(e).__name__}: {e}). "
                         f"Restart the same command to resume from the checkpoint.")
            print(f"Q{i + 1} FAILED: {type(e).__name__}: {e}")

    correct, incorrect, halted = summarize_react_trial(agents)
    with open(TRIAL_LOG, "a") as f:
        f.write(log_react_trial(agents, trial))
    with open(RESULTS, "a") as f:
        f.write(json.dumps({
            "trial": trial,
            "correct": len(correct), "incorrect": len(incorrect), "halted": len(halted),
            "em": round(len(correct) / len(agents), 4),
            "llm_calls": llm_calls - c0, "wall_s": round(time.time() - t0, 1),
            "per_question": [{"qid": qids[i], "correct": a.is_correct(),
                              "finished": a.is_finished(), "halted": a.is_halted(),
                              "answer": a.answer,
                              "n_reflections": len(getattr(a, "reflections", []))}
                             for i, a in enumerate(agents)],
        }, ensure_ascii=False) + "\n")
    dump_checkpoint(CKPT, trial, agents)
    print(f"\nFinished Trial {trial}: Correct {len(correct)}, "
          f"Incorrect {len(incorrect)}, Halted {len(halted)} "
          f"(EM so far {len(correct) / len(agents):.3f}, "
          f"calls {llm_calls - c0}, {round(time.time() - t0, 1)}s)")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n===== DONE =====")
for line in open(RESULTS):
    r = json.loads(line)
    print(f"trial {r['trial']}: EM={r['em']:.3f} "
          f"(correct {r['correct']}, incorrect {r['incorrect']}, halted {r['halted']})")
