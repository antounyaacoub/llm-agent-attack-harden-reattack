"""Pilot check for the chapter experiment: parses every component's output. Prints PASS/WARN/FAIL."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import hardening as H

def first_text(m):
    c = m.get("content")
    return "".join(b.get("content") or "" for b in c) if isinstance(c, list) else str(c or "")

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", required=True); a = ap.parse_args()
pdir = Path(a.outdir) / "pilot"
df = pd.DataFrame([json.loads(l) for l in (pdir / "results.jsonl").read_text().splitlines()])
df = df.drop_duplicates(["model", "config", "attack", "suite", "user_task", "injection_task"], keep="last")
ok_all = True
def check(c, label):
    global ok_all; ok_all &= bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {label}")
def warn(c, label): print(f"  [{'PASS' if c else 'WARN'}] {label}")

for m, g in df.groupby("model"):
    print(f"\n=== {m} ===")
    okg = g[g.status == "ok"]
    check(len(okg) == 24, f"{len(okg)}/24 episodes completed")
    for _, r in g[g.status != "ok"].iterrows(): print("     error:", r.config, r.attack, r.suite, str(r.get("error"))[:150])
    check(set(okg.attack) == {"none", "important_instructions", "justify", "exhaust"}, f"attacks ran: {sorted(set(okg.attack))}")
    check(set(okg.config) == {"none", "lp_monitor", "hardened"}, f"configs ran: {sorted(set(okg.config))}")
    check((okg.tool_calls > 0).groupby(okg.config).any().all(), "every configuration makes tool calls")
    lp = okg[okg.config.isin(["lp_monitor", "hardened"])]
    check(lp.allowlist_size.notna().all() and (lp.allowlist_size > 0).all(), "allowlist computed and non-empty")
    check(okg[okg.config == "lp_monitor"].calls_checked.sum() > 0, "LLM monitor active in lp_monitor")
    check(okg[okg.config != "hardened"].escalated.sum() == 0, "taint gate inactive outside 'hardened'")
    hatt = okg[(okg.config == "hardened") & (okg.attack != "none")]
    check(hatt.escalated.sum() > 0, f"taint gate escalated {int(hatt.escalated.sum())} calls in attacked episodes")
    # payload reached the agent, per attack, in the undefended config
    tdir = pdir / "traces" / f"{m}-none"
    for atk, marker in H.ATTACK_MARKERS.items():
        files = list(tdir.rglob(f"{atk}/*.json")) if tdir.exists() else []
        seen = sum(any(x["role"] == "tool" and marker in first_text(x) for x in json.loads(f.read_text()).get("messages", [])) for f in files)
        check(seen > 0, f"{atk}: payload reached the agent in {seen}/{len(files)} undefended episodes")
    print("  Gate decisions (hardened):")
    for _, r in okg[okg.config == "hardened"].iterrows():
        for gl in (r.gate_log or []): print(f"    {r.attack:22s} {r.suite:9s} {gl}")
    hb = okg[(okg.config == "hardened") & (okg.attack == "none")]
    warn(not any("REFUSED" in gl for gls in hb.gate_log for gl in (gls or [])), "no refusal in benign hardened episodes")
    tok = okg.prompt_tokens + okg.completion_tokens
    full = 28 * (1 + 3 * 4) * 3
    print(f"  Pilot: {int(tok.sum()):,} tokens. Full run: {full} episodes, about {tok.mean() * full / 1e6:.1f}M tokens")
print("\nPILOT", "PASSED: go for the full run." if ok_all else "FAILED: send this output before running more.")
