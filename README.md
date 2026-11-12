# Attack, Harden, Re-Attack — code, data and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24171175.svg)](https://doi.org/10.5281/zenodo.24171175)

Artifact of: A. Yaacoub and D. Ammar, *Attack, Harden, Re-Attack: A Security Engineering Lifecycle for LLM Agent Defenses Against Indirect Prompt Injection*, book chapter, Springer.
It builds on the conference study *Defense in Depth for LLM Agents* (BCE'27), whose artifact is at https://doi.org/10.5281/zenodo.23644837.

## Contents
| Path | What it is |
|---|---|
| `code/hardening.py` | The two adaptive attacks (A1 *justify*, monitor-aware; A2 *exhaust*, resource exhaustion) and the deterministic **taint gate** with escalation to a simulated user and fail-safe lock. |
| `code/defenses.py` | Tool allowlist (least privilege), LLM monitor (C5) and configuration `hardened` (H). |
| `code/run.py` | Runner; resumable; prints every component's decision; stops on anomalies. |
| `code/check_chapter.py` | Validates a pilot run component by component. |
| `code/analyze_chapter.py` | Reproduces all numbers of the chapter (`results/summary.csv`). |
| `code/keys.py`, `code/requirements.txt` | API client and dependencies. |
| `code/reproduce_colab.ipynb` | Self-contained Colab notebook. |
| `data/sample.json` | Fixed task sample (seed 2026), AgentDojo v1.2.1. |
| `data/bce27_results.jsonl` | Results of the conference study, from which the C0/C5 cells under the standard attack and without attack are reused. |
| `results/results.jsonl` | One line per new episode: configuration, attack, task, utility, attack success, tokens, time, monitor and gate decisions (`gate_log`, `escalated`, `approved`, `locked`). |
| `results/traces/` | Not in this repository (too many files): see the Zenodo archive. |
| `results/pilot/` | Validated pilot. |
| `figures/` | Figures of the chapter (PDF). |

## Setup
AgentDojo 0.1.35, benchmark v1.2.1, suites workspace and banking, 28 user tasks, 2 attacker goals per suite. Models: `gpt-4o-mini-2024-07-18` (OpenAI) and `openai/gpt-oss-120b` (OpenRouter, `:nitro`), temperature 0. Configurations: C0 (none), C5 (allowlist + LLM monitor), H (allowlist + taint gate). Attacks: A0 `important_instructions`, A1 `justify`, A2 `exhaust`. 1,064 episodes analyzed (728 new, 336 reused).

## Reproduce
```
cd code && pip install -r requirements.txt
python analyze_chapter.py
```
Full experiment: `code/reproduce_colab.ipynb` (OpenAI and OpenRouter keys required).

## License
Code: MIT. Data and results: CC BY 4.0.
