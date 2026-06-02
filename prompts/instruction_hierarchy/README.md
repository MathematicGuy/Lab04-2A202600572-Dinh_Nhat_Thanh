# Instruction Hierarchy — Prompt Templates

This folder contains structured system-prompt templates derived from the OpenAI *"Instruction Hierarchy"* research paper. They are used by the **Instruction Hierarchy Lab** tab in `app.py` to run side-by-side before/after comparisons on `meta-llama/llama-3.2-3b-instruct`.

---

## Priority Tier Mapping (from the paper)

| Tier | Source | Privilege | Description |
|------|--------|-----------|-------------|
| **Priority 0** | System Message | **Highest** | Developer-set safety rules & business constraints |
| **Priority 10** | User Message | Medium | End-user queries, followed only if aligned with P0 |
| **Priority 20** | Multi-modal Inputs | Medium/Low | Images, audio — treated as structured data only |
| **Priority 30** | Tool / API Outputs | **Lowest** | RAG chunks, DB results — text only, never executable |

---

## Files

| File | Purpose |
|------|---------|
| `naive_system.txt` | Minimal undefended system prompt — **BEFORE** baseline |
| `ih_system.txt` | Full IH-hardened system prompt — **AFTER** treatment |
| `base_system.json` | Machine-readable version of the IH framework config |
| `scenarios.json` | Six test scenarios covering jailbreak, injection, extraction, and benign queries |

---

## How to Use

The `src/core/ih_runner.py` helper loads these templates and calls `meta-llama/llama-3.2-3b-instruct` (via OpenRouter) once with each system prompt for a given scenario. Results are returned to `app.py` for side-by-side visualization.

```python
from src.core.ih_runner import run_ih_comparison

result = run_ih_comparison(scenario_id="jailbreak_01")
# result = {"naive": str, "ih": str, "judge_naive": dict, "judge_ih": dict}
```
