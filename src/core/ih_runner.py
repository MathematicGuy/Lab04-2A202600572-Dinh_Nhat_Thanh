"""
ih_runner.py — Instruction Hierarchy inference helper.

Loads naive vs IH-hardened system prompts and runs a given scenario
through meta-llama/llama-3.2-3b-instruct (OpenRouter) twice, then optionally judges both.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
IH_DIR = ROOT_DIR / "prompts" / "instruction_hierarchy"

NAIVE_SYSTEM_PATH = IH_DIR / "naive_system.txt"
IH_SYSTEM_PATH = IH_DIR / "ih_system.txt"
SCENARIOS_PATH = IH_DIR / "scenarios.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_scenarios() -> list[dict[str, Any]]:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


def get_scenario(scenario_id: str) -> dict[str, Any]:
    for s in load_scenarios():
        if s["id"] == scenario_id:
            return s
    raise ValueError(f"Scenario '{scenario_id}' not found in scenarios.json")


def _load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _build_messages(system_prompt: str, scenario: dict[str, Any]) -> list[dict]:
    """Assemble the message list for a single run."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # If there is a simulated tool output, inject it as an assistant turn
    # (simulating what the model would see after a tool call).
    if scenario.get("injected_tool_output"):
        # We frame it as a user message wrapping the tool output to simulate
        # a realistic agentic loop where the model receives tool data.
        tool_framing = (
            "Here are the results from the search tool:\n\n"
            + scenario["injected_tool_output"]
        )
        messages.append({"role": "user", "content": tool_framing})
        messages.append({
            "role": "assistant",
            "content": "I have retrieved the data. Let me now answer your request."
        })

    messages.append({"role": "user", "content": scenario["user_message"]})
    return messages


def _invoke_model(messages: list[dict], model_name: str, provider: str) -> str:
    """Call the LLM and return the text response."""
    from core.llm import build_chat_model, normalize_content
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    llm = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    response = llm.invoke(lc_messages)
    return normalize_content(response.content)


def _judge_response(
    *,
    query: str,
    answer: str,
    rubric: str,
    judge_provider: str,
    judge_model: str | None,
) -> dict[str, Any]:
    """Grade a single response using the LLM Judge."""
    try:
        from core.llm import judge_answer_with_llm
        return judge_answer_with_llm(
            query=query,
            answer=answer,
            rubric=rubric,
            provider=judge_provider,
            model_name=judge_model,
        )
    except Exception as exc:
        return {"score": 0, "verdict": f"Judge error: {exc}", "feedback": []}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ih_comparison(
    *,
    scenario_id: str,
    model_name: str = "meta-llama/llama-3.2-3b-instruct",
    provider: str = "google",
    judge_provider: str = "openai",
    judge_model: str | None = "gpt-5.4-mini",
    run_judge: bool = True,
) -> dict[str, Any]:
    """
    Run the same scenario twice — naive system vs IH-hardened system.

    Returns a dict:
    {
        "scenario": {...},
        "naive_response": str,
        "ih_response": str,
        "naive_system": str,
        "ih_system": str,
        "judge_naive": dict | None,
        "judge_ih": dict | None,
    }
    """
    scenario = get_scenario(scenario_id)
    naive_system = _load_system_prompt(NAIVE_SYSTEM_PATH)
    ih_system = _load_system_prompt(IH_SYSTEM_PATH)

    naive_messages = _build_messages(naive_system, scenario)
    ih_messages = _build_messages(ih_system, scenario)

    naive_response = _invoke_model(naive_messages, model_name=model_name, provider=provider)
    ih_response = _invoke_model(ih_messages, model_name=model_name, provider=provider)

    judge_naive = None
    judge_ih = None
    if run_judge and judge_provider:
        rubric = scenario["expected_safe_behavior"]
        judge_naive = _judge_response(
            query=scenario["user_message"],
            answer=naive_response,
            rubric=rubric,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )
        judge_ih = _judge_response(
            query=scenario["user_message"],
            answer=ih_response,
            rubric=rubric,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )

    return {
        "scenario": scenario,
        "naive_response": naive_response,
        "ih_response": ih_response,
        "naive_system": naive_system,
        "ih_system": ih_system,
        "judge_naive": judge_naive,
        "judge_ih": judge_ih,
    }
