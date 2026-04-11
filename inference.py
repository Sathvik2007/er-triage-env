#!/usr/bin/env python3
"""
inference.py — ER Triage OpenEnv Agent
========================================
Runs an LLM agent through all 3 triage tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL      LLM API endpoint
    API_KEY           API key
    MODEL_NAME        Model identifier (default: gpt-4o-mini)
    ENV_BASE_URL      Environment server URL

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import re
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from openenv.core.generic_client import GenericEnvClient as HTTPEnvClient
except Exception:
    from openenv_core import HTTPEnvClient

from models import TriageAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME    = os.getenv("TASK_NAME", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sathvik2007-er-triage-env.hf.space")
BENCHMARK    = "er-triage"

MAX_STEPS               = 70
TEMPERATURE             = 0.15
MAX_TOKENS              = 220
SUCCESS_SCORE_THRESHOLD = 0.45

TASKS = [
    "simple-triage",
    "resource-constraint",
    "critical-overload",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Emergency Room Triage AI.
    Goal: Maximize survival rate, minimize average waiting time, use resources efficiently.
    Rules:
    - Always treat highest severity first, especially trauma and cardiac with ICU/doctor.
    - ICU is precious — reserve for severity > 0.75.
    - Ignored patients worsen and may die.
    - Output ONLY a single valid JSON object in this exact schema:
      {"action_type":"assign","patient_id":<int>,"resource":"bed|icu|doctor"}
      or {"action_type":"wait"}
      (no extra text, no markdown).
""").strip()

# ---------------------------------------------------------------------------
# Logging helpers — must match the OpenEnv spec exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def normalize_action(raw_action: dict) -> dict:
    action = dict(raw_action)
    raw_type = action.get("action_type")
    if raw_type is None and "action" in action:
        alias = str(action.get("action")).strip().lower()
        if alias in {"treat", "assign", "allocate"}:
            raw_type = "assign"
        elif alias in {"wait", "idle", "noop"}:
            raw_type = "wait"
    if raw_type is None and action.get("patient_id") is not None and action.get("resource") is not None:
        raw_type = "assign"
    if isinstance(action.get("resource"), str):
        resource = action["resource"].strip().lower()
        resource_aliases = {"physician": "doctor", "doc": "doctor", "icu_bed": "icu"}
        action["resource"] = resource_aliases.get(resource, resource)
    action["action_type"] = raw_type or "wait"
    if action["action_type"] != "assign":
        return {"action_type": "wait"}
    return {
        "action_type": "assign",
        "patient_id": action.get("patient_id"),
        "resource": action.get("resource"),
    }

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, observation: dict) -> dict:
    user_prompt = f"Current observation:\n{json.dumps(observation, indent=2)}\nChoose next action."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip <think>...</think> reasoning blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.strip()

        print(f"[LLM] Response: {text}", flush=True)
        action_dict = normalize_action(json.loads(text))
        return action_dict

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc} — defaulting to wait", flush=True)
        return {"action_type": "wait"}

# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

import asyncio

async def run_task(client: OpenAI, task_name: str) -> None:
    rewards:    List[float] = []
    steps_taken = 0
    score        = 0.0
    success      = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        env = HTTPEnvClient(base_url=ENV_BASE_URL)
        await env.connect()

        result = await env.reset(task=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_data = result.observation
            obs_dict = obs_data.model_dump() if hasattr(obs_data, "model_dump") else obs_data

            action_dict = get_model_action(client, obs_dict)
            action_str  = json.dumps(action_dict)

            try:
                action  = TriageAction(**action_dict)
                result  = await env.step(action)
                last_error = None
            except Exception as exc:
                last_error = str(exc)
                result.done = True

            reward = result.reward or 0.0
            done   = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                break

        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = max(0.01, min(score, 0.99))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)
        last_error = str(exc)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — iterate all tasks
# ---------------------------------------------------------------------------

def get_task_list() -> List[str]:
    if TASK_NAME.strip():
        return [TASK_NAME.strip()]
    return TASKS


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in get_task_list():
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())