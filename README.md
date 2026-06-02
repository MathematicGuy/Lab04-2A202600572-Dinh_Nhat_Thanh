# TravelBuddy Lab: Tool-Calling Agent with `create_agent`

This lab introduces a practical pattern for building an LLM application that can reason over tool outputs and produce a grounded final response. You will implement a travel assistant for TravelBuddy that can search flights, estimate budget feasibility, and suggest hotels based on the remaining budget.

## Prerequisite Knowledge

Before starting this lab, you should be comfortable with:

- basic Python functions and modules
- reading JSON data
- environment setup with `uv`
- the idea of LLM tools / function calling
- prompt basics for system and user instructions

## Learning Outcomes

After completing this lab, you should be able to:

- build a prebuilt tool-calling agent with `create_agent`
- design clear tool schemas and tool descriptions
- write a system prompt that controls tool usage and answer style
- keep the final answer grounded in tool outputs
- evaluate agent quality with answer-based grading

## Lab Deliverable

Complete [src/agent/graph.py](/Users/duongnh59.al1/Documents/Project/Vin20K/Cohort2/Day-4-Lab/labs/src/agent/graph.py) so that the agent:

- calls the correct tools when information is available
- asks for clarification when key trip details are missing
- refuses unsafe or illegal requests
- returns a concise final answer in Vietnamese

## Project Layout

- `task.txt`: assignment brief
- `guide.md`: implementation guide
- `rubric.md`: grading rubric
- `src/agent/graph.py`: lab scaffold
- `src/core/`: model helpers and result schema
- `src/utils/`: dataset helpers
- `grade/scoring.py`: grading script
- `data/`: datasets and grading cases

## Setup

```bash
cd labs
uv sync --extra dev
```

## Run the Grader

You can test and grade the TravelBuddy agent using three different model backends:

### 1. Default: OpenRouter Llama 3.2 3B Instruct (Recommended)
Tests using `meta-llama/llama-3.2-3b-instruct` via OpenRouter (requires `OPENROUTER_API_KEY`).
No flags needed — this is the default:
```bash
uv run python grade/scoring.py
```

### 2. OpenRouter — any model
Explicitly route through OpenRouter with a specific model:
```bash
uv run python grade/scoring.py --provider openrouter --model-name meta-llama/llama-3.2-3b-instruct
uv run python grade/scoring.py --provider openrouter --model-name deepseek/deepseek-v4-flash
```

### 3. Native OpenAI Model
Tests using a model served directly by OpenAI (requires `OPENAI_API_KEY`):
```bash
uv run python grade/scoring.py --provider openai --model-name gpt-5.4-mini
```

### 4. Local Ollama Model
Tests using a locally running Ollama model:
```bash
uv run python grade/scoring.py --provider ollama --model-name phi3
```

## Optional LLM Judge

Add `--judge-provider` and `--judge-model-name` to enable LLM-based quality grading.
The judge provider is **independent** of the tested model provider:

```bash
# OpenRouter model tested, OpenAI judge (default)
uv run python grade/scoring.py

# Explicit flags version of the same command
uv run python grade/scoring.py --provider openrouter --model-name openai/gpt-oss-20b --judge-provider openai --judge-model-name gpt-5.4-mini

# Force fresh run (skip cache)
uv run python grade/scoring.py --no-cache
```

> **Provider quick reference:**
> | Provider flag | Routes to | Key required |
> |---|---|---|
> | `openrouter` | openrouter.ai API | `OPENROUTER_API_KEY` |
> | `openai` | api.openai.com | `OPENAI_API_KEY` |
> | `ollama` | localhost:11434 | none |
> | `google` | auto-detect (legacy) | either |
