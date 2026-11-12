"""Reproduces the tables and figures of the chapter from the raw results.

New episodes: ../results/results.jsonl. Reused cells (C0 and C5 under the standard attack and without attack)
come from the conference study: ../data/bce27_results.jsonl (also at https://doi.org/10.5281/zenodo.23644837).
Only the two attacker goals per suite used in the chapter are kept; the last run of a duplicated episode is kept.
"""
import json, math, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); P = lambda *a: os.path.join(HERE, "..", *a)
INJ = {"banking": ["injection_task_1", "injection_task_2"], "workspace": ["injection_task_1", "injection_task_4"]}
keep = lambda r: r["injection_task"] == "none" or r["injection_task"] in INJ[r["suite"]]
new = pd.DataFrame([json.loads(l) for l in open(P("results", "results.jsonl"))])
new = new[(new.status == "ok") & new.apply(keep, axis=1)].drop_duplicates(
    ["model", "config", "attack", "suite", "user_task", "injection_task"], keep="last")
new = new[~(new.config.isin(["none", "lp_monitor"]) & new.attack.isin(["important_instructions", "none"]))]
new = new[~((new.config == "none") & (new.attack == "exhaust"))]
bce = pd.DataFrame([json.loads(l) for l in open(P("data", "bce27_results.jsonl"))])
bce = bce[bce.config.isin(["none", "lp_monitor"])].copy()
bce["attack"] = bce.injection_task.map(lambda x: "none" if x == "none" else "important_instructions")
bce = bce[bce.apply(keep, axis=1)]
df = pd.concat([bce.assign(src="reused"), new.assign(src="new")], ignore_index=True)
df["M"] = df.model.map(lambda m: "mini" if m.startswith("gpt") else "oss"); df["tok"] = df.prompt_tokens + df.completion_tokens
for c in ["escalated", "approved"]: df[c] = df[c].fillna(0)
df["locked"] = df["locked"].fillna(False)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), c + h
rows = []
for (m, c, a), g in df.groupby(["M", "config", "attack"]):
    r = dict(model=m, config=c, attack=a, n=len(g), utility=round(100 * g.utility.mean(), 1), tokens=round(g.tok.mean()),
             seconds=round(g.seconds.mean(), 2), checked=int(g.calls_checked.sum()), denied=int(g.calls_blocked.sum()),
             escalated=int(g.escalated.sum()), approved=int(g.approved.sum()), locks=int(g.locked.sum()))
    if a != "none":
        k = int(g.attack_success.sum()); lo, hi = wilson(k, len(g))
        r.update(asr=round(100 * k / len(g), 1), asr_lo=round(100 * lo, 1), asr_hi=round(100 * hi, 1))
    rows.append(r)
t = pd.DataFrame(rows); t.to_csv(P("results", "summary.csv"), index=False)
print(f"{len(df)} episodes ({(df.src=='new').sum()} new, {(df.src=='reused').sum()} reused)")
print(t.to_string(index=False))
