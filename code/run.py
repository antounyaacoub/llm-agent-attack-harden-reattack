"""Run the defense ablation on a fixed AgentDojo sample. Resumable: rerun the same command to continue.

Examples
  python run.py --provider openai --model gpt-4o-mini-2024-07-18 --pilot
  python run.py --provider openai --model gpt-4o-mini-2024-07-18
  python run.py --provider groq   --model llama-3.3-70b-versatile --keys-file groq_keys.txt
  python run.py --provider openrouter --model openai/gpt-oss-120b      (needs OPENROUTER_API_KEY)

Groq keys: one per line in --keys-file, or comma-separated in GROQ_API_KEYS.
--outdir: put it on Google Drive in Colab so nothing is lost if the runtime resets.
"""
import argparse, json, os, random, time, traceback, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import openai
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.attacks import base_attacks  # noqa: F401  (registers attacks)
from agentdojo.attacks import important_instructions_attacks  # noqa: F401
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import Logger, NullLogger, OutputLogger
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suite

import defenses as D
import hardening as H
from keys import AllKeysExhausted, RotatingClient, is_malformed

VERSION = "v1.2.1"
ATTACK = "important_instructions"
SAMPLE = {"workspace": (12, 4), "banking": (16, 4)}  # (user tasks, injection tasks) per suite
SEED = 2026
HERE = Path(__file__).parent
OUT = HERE / "results"          # overridden by --outdir
RESULTS = OUT / "results.jsonl"


class QuietLogger(NullLogger):
    """Saves traces to logdir without printing every message."""
    def __init__(self, logdir):
        self._dir = logdir

    def __enter__(self):
        Logger.__enter__(self)
        self.messages, self.logdir = [], self._dir
        return self

    def __exit__(self, *exc):
        return Logger.__exit__(self, *exc)


def load_groq_keys(keys_file):
    keys = []
    if keys_file:
        keys += [l.strip() for l in Path(keys_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    keys += [k for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
    if os.environ.get("GROQ_API_KEY"):
        keys.append(os.environ["GROQ_API_KEY"])
    return list(dict.fromkeys(keys))  # dedupe, keep order


def make_client(provider, keys_file=None):
    if provider == "openai":
        return openai.OpenAI(max_retries=8, timeout=120)
    if provider == "groq":
        keys = load_groq_keys(keys_file)
        print(f"Using {len(keys)} Groq key(s)")
        return RotatingClient(keys, base_url="https://api.groq.com/openai/v1")
    if provider == "openrouter":
        print("Using OpenRouter")
        return RotatingClient([os.environ["OPENROUTER_API_KEY"]], base_url="https://openrouter.ai/api/v1")
    raise ValueError(provider)


def get_sample(root):
    path = root / "sample.json"
    if path.exists():
        return json.loads(path.read_text())
    rng, sample = random.Random(SEED), {}
    for s, (nu, ni) in SAMPLE.items():
        suite = get_suite(VERSION, s)
        u = sorted(suite.user_tasks, key=lambda k: int(k.split("_")[-1]))
        i = sorted(suite.injection_tasks, key=lambda k: int(k.split("_")[-1]))
        sample[s] = {"user": sorted(rng.sample(u, min(nu, len(u))), key=lambda k: int(k.split("_")[-1])),
                     "inj": sorted(rng.sample(i, min(ni, len(i))), key=lambda k: int(k.split("_")[-1]))}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(sample, indent=2))
    return sample


def done_keys():
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            r = json.loads(line)
            if r["status"] in ("ok", "model_error"):
                keys.add((r["model"], r["config"], r.get("attack", "none"), r["suite"], r["user_task"], r["injection_task"]))
    return keys


def available_models(raw):
    first = raw.clients[0] if isinstance(raw, RotatingClient) else raw
    try:
        return sorted(m.id for m in first.models.list().data)
    except Exception as e:
        return [f"(could not list models: {e})"]


def preflight(raw, model):
    """One tiny call before anything else: stop at once if the model name or key is wrong."""
    try:
        raw.chat.completions.create(model=model, max_tokens=5,
                                    messages=[{"role": "user", "content": "Reply OK."}])
        print(f"Preflight OK: {model}")
    except AllKeysExhausted:
        raise SystemExit("All keys are used for today. Rerun tomorrow.")
    except (openai.NotFoundError, openai.BadRequestError, openai.AuthenticationError,
            openai.PermissionDeniedError) as e:
        print(f"Preflight FAILED for model '{model}': {type(e).__name__}: {e}")
        print("Models available to this key:")
        for m in available_models(raw):
            print("   ", m)
        raise SystemExit("Fix the model name (cell 2) and rerun. Nothing was run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "groq", "openrouter"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", default=["none", "lp_monitor", "hardened"])
    ap.add_argument("--n-inj", type=int, default=4)
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--attacks", nargs="+", default=["important_instructions", "justify", "exhaust"])
    ap.add_argument("--pilot", action="store_true",
                    help="1 user task x 1 injection per suite, all configs, saved in <outdir>/pilot")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--outdir", default=str(HERE / "results"))
    a = ap.parse_args()

    global OUT, RESULTS
    OUT = Path(a.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    sample_root = OUT  # the pilot uses the same fixed sample as the full run
    if a.pilot:
        OUT = OUT / "pilot"
        OUT.mkdir(parents=True, exist_ok=True)
    RESULTS = OUT / "results.jsonl"
    # ---- guard: refuse to run with old code ----
    if not hasattr(D, "ToolAllowlist"):
        raise SystemExit("STOPPED: defenses.py is the OLD version (no ToolAllowlist). Rerun the fix cell. Nothing was run.")
    if not hasattr(H, "TaintGateExecutor") or "hardened" not in D.CONFIGS:
        raise SystemExit("STOPPED: chapter code not loaded. Rerun the code cells. Nothing was run.")
    print("Code check OK: chapter v1")
    raw = make_client(a.provider, a.keys_file)
    preflight(raw, a.model)
    client = D.CountingClient(raw)
    name = a.model.replace("/", "_")
    MODEL_NAMES.setdefault(name, "AI assistant")  # name used inside the attack text
    sample, done = get_sample(sample_root), done_keys()
    logdir = OUT / "traces"

    jobs = []
    for cfg in a.configs:
        for s, smp in sample.items():
            users = smp["user"][:1] if a.pilot else smp["user"]
            injs = smp["inj"][:1] if a.pilot else smp["inj"][:a.n_inj]
            for u in users:
                if not a.no_benign: jobs.append((cfg, "none", s, u, "none"))
                for atk in a.attacks:
                    for i in injs:
                        jobs.append((cfg, atk, s, u, i))
    todo = [j for j in jobs if (name, *j) not in done]
    print(f"Results file: {RESULTS}")
    print(f"{len(jobs)} episodes, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    pipelines = {c: D.build_pipeline(c, client, a.model, name) for c in a.configs}
    log_ctx = OutputLogger(str(logdir)) if a.verbose else QuietLogger(str(logdir))
    bad_streak = 0
    idle_streak = 0
    with log_ctx:
        for n, (cfg, atk, s, u, i) in enumerate(todo, 1):
            suite, pipe = get_suite(VERSION, s), pipelines[cfg]
            D.reset_stats()
            t0 = time.time()
            row = dict(model=name, config=cfg, attack=atk, suite=s, user_task=u, injection_task=i)
            H.set_oracle(suite.get_user_task_by_id(u), suite)
            try:
                if i == "none":
                    util, _ = run_task_without_injection_tasks(
                        suite, pipe, suite.get_user_task_by_id(u), logdir, True, VERSION)
                    sec = None
                else:
                    attack = load_attack(atk, suite, pipe)
                    inj_text = attack.attack(suite.get_user_task_by_id(u), suite.get_injection_task_by_id(i))
                    if not inj_text or not all(H.ATTACK_MARKERS[atk] in v for v in inj_text.values()):
                        print(f"\nSTOPPED: attack {atk} did not produce its payload for {s} {u} {i}. Nothing saved.")
                        return
                    ur, sr = run_task_with_injection_tasks(
                        suite, pipe, suite.get_user_task_by_id(u), attack, logdir, True, [i], VERSION)
                    util, sec = ur[(u, i)], sr[(u, i)]
                row.update(status="ok", utility=bool(util), attack_success=None if sec is None else bool(sec))
            except AllKeysExhausted:
                print(f"\nAll keys used for today. Stopped after {n - 1} new episodes. "
                      "Rerun the same command tomorrow; finished episodes are saved in", RESULTS)
                return
            except Exception as e:
                # model_error = the model itself produced an invalid action 4 times in a row: a task failure, not retried
                row.update(status="model_error" if is_malformed(e) else "error",
                           error=f"{type(e).__name__}: {e}"[:500])
                if isinstance(e, openai.RateLimitError):
                    print("Rate limit reached. Stop now and rerun the same command later; finished episodes are kept.")
            row.update(seconds=round(time.time() - t0, 2), monitor_seconds=round(D.STATS["monitor_s"], 2),
                       calls_checked=D.STATS["checked"], calls_blocked=D.STATS["blocked"],
                       llm_calls=D.STATS["llm_calls"], prompt_tokens=D.STATS["prompt_tokens"],
                       completion_tokens=D.STATS["completion_tokens"], allowlist_size=D.STATS["allowlist_size"],
                       allowlist_raw=D.STATS["allowlist_raw"], tool_calls=D.STATS["tool_calls"],
                       escalated=D.STATS["escalated"], approved=D.STATS["approved"], locked=D.STATS["locked"],
                       gate_log=list(D.STATS["gate_log"]))
            # ---- guards: stop at the first sign of a broken component, before saving ----
            if cfg in ("least_priv", "all", "lp_monitor", "hardened"):
                print(f"      selector said: {row['allowlist_raw']!r}")
            for g in row["gate_log"]:
                print(f"      gate: {g}")
            if cfg in ("least_priv", "all", "lp_monitor", "hardened") and row["status"] == "ok" and not row["allowlist_size"]:
                print(f"\nSTOPPED: empty allowlist in {cfg} {s} {u} {i}. The selector is broken. Episode NOT saved.")
                return
            idle_streak = idle_streak + 1 if row["status"] == "ok" and row["tool_calls"] == 0 and not row.get("utility") else 0
            if idle_streak >= 11:  # > the 5 episodes of two user tasks: at least 3 tasks idle and failing in a row
                print(f"\nSTOPPED: the agent made NO tool call in 11 failed episodes in a row ({cfg}). Episode NOT saved.")
                return
            bad_streak = bad_streak + 1 if row["status"] != "ok" else 0
            if bad_streak >= 3:
                print(f"\nSTOPPED: 3 failed episodes in a row. Last error: {row.get('error')}")
                return
            with RESULTS.open("a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
            print(f"[{n}/{len(todo)}] {cfg:10s} {atk:22s} {s:9s} {u:13s} {i:17s} -> {row['status']}"
                  f" util={row.get('utility')} asr={row.get('attack_success')} {row['seconds']}s"
                  f" tok={row['prompt_tokens'] + row['completion_tokens']}"
                  f" tools={row['tool_calls']} allow={row['allowlist_size']} blocked={row['calls_blocked']}/{row['calls_checked']}"
                  f" esc={row['escalated']} ok={row['approved']}{' LOCKED' if row['locked'] else ''}"
                  + (f" ERROR {row['error']}" if row["status"] == "error" else ""))


if __name__ == "__main__":
    main()
