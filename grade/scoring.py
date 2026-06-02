from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT_DIR / "grade" / "cache"
RESULTS_DIR = ROOT_DIR / "grade" / "results"
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.llm import judge_answer_with_llm
from core.schemas import AgentResult, ToolCallRecord


@dataclass
class CaseScore:
    case_id: str
    score: float
    max_score: float
    feedback: list[str]


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(model_name: str, case_id: str) -> str:
    """Stable filename-safe key derived from model + case."""
    raw = f"{model_name}::{case_id}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    safe_model = model_name.replace("/", "_").replace(":", "-")
    return f"{safe_model}__{case_id}__{digest}"


def _cache_path(model_name: str, case_id: str) -> Path:
    return CACHE_DIR / f"{_cache_key(model_name, case_id)}.json"


def load_cached_result(model_name: str, case_id: str) -> AgentResult | None:
    """Return a cached AgentResult if one exists, else None."""
    path = _cache_path(model_name, case_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruct tool_calls list from plain dicts
        from core.schemas import ToolCallRecord
        data["tool_calls"] = [
            ToolCallRecord(name=tc["name"], args=tc.get("args", {}), output=tc.get("output", tc.get("result", "")))
            for tc in data.get("tool_calls", [])
        ]
        data.pop("_cached_at", None)  # remove internal metadata before constructing
        return AgentResult(**data)
    return None


def save_cached_result(model_name: str, case_id: str, result: AgentResult) -> None:
    """Persist an AgentResult to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model_name, case_id)
    payload = {
        "query": result.query,
        "final_answer": result.final_answer,
        "provider": result.provider,
        "model_name": result.model_name,
        "tool_calls": [
            {"name": tc.name, "args": tc.args, "output": tc.output}
            for tc in result.tool_calls
        ],
        "_cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_run_summary(summary: dict[str, Any], model_name: str) -> Path:
    """Save the final summary JSON with a timestamp to grade/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_name.replace("/", "_").replace(":", "-")
    out_path = RESULTS_DIR / f"{safe_model}__{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def coerce_result(raw: Any, *, query: str, provider: str, model_name: str | None) -> AgentResult:
    if isinstance(raw, AgentResult):
        return raw
    if isinstance(raw, str):
        return AgentResult(query=query, final_answer=raw, provider=provider, model_name=model_name)
    if isinstance(raw, dict):
        return AgentResult(**raw)
    raise TypeError(f"Unsupported run_agent result type: {type(raw)!r}")


def grade_result(
    result: AgentResult,
    case: dict[str, Any],
    *,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
) -> CaseScore:
    expected = case["expected"]
    weights = case["weights"]
    earned = 0.0
    feedback: list[str] = []
    answer = result.final_answer.lower()

    required_keywords = [item.lower() for item in expected.get("required_keywords", [])]
    keyword_hits = sum(1 for keyword in required_keywords if keyword in answer)
    if required_keywords:
        keyword_score = weights["keywords"] * (keyword_hits / len(required_keywords))
        earned += keyword_score
        if keyword_hits < len(required_keywords):
            missing = [keyword for keyword in required_keywords if keyword not in answer]
            feedback.append(f"Missing required keywords: {missing}.")
    else:
        earned += weights["keywords"]

    forbidden_keywords = [item.lower() for item in expected.get("forbidden_keywords", [])]
    violations = [keyword for keyword in forbidden_keywords if keyword in answer]
    if not violations:
        earned += weights["safety"]
    else:
        feedback.append(f"Answer contains forbidden keywords: {violations}.")

    required_tools = expected.get("required_tools", [])
    tool_names = [tool.name for tool in result.tool_calls]
    if all(tool in tool_names for tool in required_tools):
        earned += weights["tools"]
    else:
        missing_tools = [tool for tool in required_tools if tool not in tool_names]
        feedback.append(f"Missing required tools: {missing_tools}.")

    if judge_provider:
        judge = judge_answer_with_llm(
            query=result.query,
            answer=result.final_answer,
            rubric=expected.get("grading_rubric", ""),
            provider=judge_provider,
            model_name=judge_model_name,
        )
        earned += weights["llm_judge"] * (judge["score"] / 10)
        feedback.extend(judge["feedback"])
    else:
        earned += weights["llm_judge"]

    return CaseScore(
        case_id=case["id"],
        score=round(earned, 2),
        max_score=float(sum(weights.values())),
        feedback=feedback,
    )


def summarize_scores(scores: list[CaseScore]) -> dict[str, Any]:
    total_earned = sum(item.score for item in scores)
    total_max = sum(item.max_score for item in scores)
    overall = round((total_earned / total_max) * 100, 2) if total_max else 0.0
    return {
        "overall_score": overall,
        "total_earned": total_earned,
        "total_max": total_max,
        "cases": [
            {
                "case_id": item.case_id,
                "score": item.score,
                "max_score": item.max_score,
                "feedback": item.feedback,
            }
            for item in scores
        ],
    }

# uv run python grade/scoring.py --model-name meta-llama/llama-3.2-3b-instruct --provider openrouter --judge-provider openai --judge-model-name gpt-5.4-mini
def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Grade final answers for the TravelBuddy create_agent lab")
    parser.add_argument("--module", default="agent.graph")
    parser.add_argument("--cases", default=str(ROOT_DIR / "data" / "graded_cases.json"))
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "openai", "ollama", "google"])
    parser.add_argument(
        "--model-name",
        default="meta-llama/llama-3.2-3b-instruct",
        help="Model name. Default is meta-llama/llama-3.2-3b-instruct via OpenRouter. Can also be OpenAI ('gpt-5.4-mini') or Ollama ('phi3')."
    )
    parser.add_argument("--today", default="2026-05-31")
    parser.add_argument("--pass-threshold", type=float, default=80.0)
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "openrouter", "ollama"])
    parser.add_argument("--judge-model-name", default="gpt-5.4-mini")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any cached agent responses and force fresh LLM calls."
    )
    args = parser.parse_args()

    use_cache = not args.no_cache
    if use_cache:
        print(f"📦 Cache enabled  →  {CACHE_DIR}")
        print(f"💾 Results saved  →  {RESULTS_DIR}")
    else:
        print("🚫 Cache disabled  — running fresh LLM calls")
    print("=" * 60)

    module = importlib.import_module(args.module)
    if not hasattr(module, "run_agent"):
        raise SystemExit(f"Module {args.module} does not expose run_agent()")

    cases = load_cases(Path(args.cases))
    scores = []
    cache_hits = 0

    for case in cases:
        case_id = case["id"]

        # ── Try cache first ────────────────────────────────────────────────
        result = None
        if use_cache:
            result = load_cached_result(args.model_name, case_id)
            if result is not None:
                cache_hits += 1
                print(f"\n⚡ [Case: {case_id}] Loaded from cache (skipping LLM call)")

        # ── Fresh LLM call if no cache hit ─────────────────────────────────
        if result is None:
            raw_result = module.run_agent(
                case["query"],
                provider=args.provider,
                model_name=args.model_name,
                today=args.today,
            )
            result = coerce_result(
                raw_result,
                query=case["query"],
                provider=args.provider,
                model_name=args.model_name,
            )
            if use_cache:
                save_cached_result(args.model_name, case_id, result)
                print(f"\n🔖 [Case: {case_id}] Response cached")

        # ── Grade ──────────────────────────────────────────────────────────
        score = grade_result(
            result,
            case,
            judge_provider=args.judge_provider,
            judge_model_name=args.judge_model_name,
        )
        scores.append(score)

        status_emoji = "✅" if score.score >= score.max_score else "⚠️" if score.score > 0 else "❌"
        print(f"{status_emoji} [Case: {score.case_id}] Score: {score.score} / {score.max_score}")
        if score.feedback:
            print("   Feedback:")
            for fb in score.feedback:
                print(f"     - {fb}")
        print("-" * 60)

    summary = summarize_scores(scores)
    summary["_meta"] = {
        "model_name": args.model_name,
        "provider": args.provider,
        "judge_provider": args.judge_provider,
        "judge_model_name": args.judge_model_name,
        "cache_hits": cache_hits,
        "total_cases": len(cases),
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Save results for demo ──────────────────────────────────────────────
    out_path = save_run_summary(summary, args.model_name)
    print(f"\n📊 Summary saved → {out_path}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["overall_score"] >= args.pass_threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
