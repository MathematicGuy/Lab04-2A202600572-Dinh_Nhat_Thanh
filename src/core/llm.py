from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "openai",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    # 1. Local GGUF or custom model check (ends with .gguf or contains .model)
    if model_name and (str(model_name).endswith(".gguf") or ".model" in str(model_name)):
        from langchain_ollama import ChatOllama
        model_id = "phi3"
        if "qwen" in str(model_name).lower():
            model_id = "qwen3.5:0.8b"
        return ChatOllama(
            model=model_id,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    # 2. Explicit OpenRouter provider — goes directly to openrouter.ai, no OpenAI key needed.
    #    Use this when model_name is an OpenRouter model like 'openai/gpt-oss-20b',
    #    'deepseek/deepseek-v4-flash', etc.
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Cannot use provider='openrouter'.")
        return ChatOpenAI(
            model=model_name or "openai/gpt-oss-20b",
            temperature=temperature,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    # 3. Native OpenAI provider — uses OPENAI_API_KEY directly.
    #    Use model_name values like 'gpt-4o-mini', 'gpt-5.4-mini', etc.
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is not set. Cannot use provider='openai'.")
        return ChatOpenAI(
            model=model_name or "gpt-5.4-mini",
            temperature=temperature,
            openai_api_key=openai_key,
        )

    # 4. Legacy 'google' provider alias — tries OpenAI key first, falls back to OpenRouter.
    #    Kept for backward compatibility with earlier versions of this codebase.
    if provider == "google":
        from langchain_openai import ChatOpenAI

        openai_key = os.getenv("OPENAI_API_KEY")
        # Proactively validate the OpenAI key to avoid 401 AuthenticationError failures
        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                client.models.list()
            except Exception:
                openai_key = None

        openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")

        primary_model = None
        fallback_model = None

        if openai_key:
            primary_model = ChatOpenAI(
                model=model_name or "gpt-5-nano-2025-08-07",
                temperature=temperature,
                openai_api_key=openai_key,
            )

        if openrouter_key:
            fallback_model = ChatOpenAI(
                model=model_name or "openai/gpt-oss-20b",
                temperature=temperature,
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )

        if primary_model and fallback_model:
            return primary_model.with_fallbacks([fallback_model])
        elif primary_model:
            return primary_model
        elif fallback_model:
            return fallback_model
        else:
            raise ValueError("No active API keys found (OpenAI or OpenRouter).")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError(f"Provider '{provider}' is not supported. Choose: openrouter, openai, ollama, google.")



def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    prompt = f"""
You are grading a student travel-agent answer.
Return JSON only with:
- score: integer from 0 to 10
- verdict: short string
- feedback: short list of strings

Rubric:
{rubric}

User query:
{query}

Student answer:
{answer}
""".strip()
    payload = extract_json_object(model.invoke(prompt).content)
    score = max(0, min(10, int(payload.get("score", 0))))
    return {
        "score": score,
        "verdict": str(payload.get("verdict", "")).strip(),
        "feedback": [str(item).strip() for item in payload.get("feedback", []) if str(item).strip()],
    }
