import asyncio
import os
import textwrap
import json
from typing import List, Optional
from openai import OpenAI
try:
    from openenv.core.generic_client import GenericEnvClient as HTTPEnvClient
except Exception:
    from openenv_core import HTTPEnvClient
from models import TriageAction

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME = os.getenv("TASK_NAME", "")
BENCHMARK = "er-triage"
MAX_STEPS = 70
TEMPERATURE = 0.15
MAX_TOKENS = 220
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sathvik2007-er-triage-env.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str] = None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_task_list() -> List[str]:
    if TASK_NAME.strip():
        return [TASK_NAME.strip()]
    return ["simple-triage", "resource-constraint", "critical-overload"]


async def run_task(task_name: str) -> None:
    log_start(task_name, BENCHMARK, MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    env = None

    try:
        env = HTTPEnvClient(base_url=ENV_BASE_URL)
        await env.connect()
        print(f"[DEBUG] Connected to env successfully for task: {task_name}", flush=True)

        result = await env.reset(task=task_name)
        print(f"[DEBUG] Reset successful for {task_name}, done={result.done}", flush=True)

        try:
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                obs_data = result.observation
                obs_dict = obs_data.model_dump() if hasattr(obs_data, "model_dump") else obs_data
                user_prompt = f"Current observation:\n{json.dumps(obs_dict, indent=2)}\nChoose next action."

                print(f"[LLM] Calling proxy at {API_BASE_URL} ...", flush=True)
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    text = (completion.choices[0].message.content or "").strip()
                    print(f"[LLM] Response: {text}", flush=True)
                    action_dict = normalize_action(json.loads(text))
                    action = TriageAction(**action_dict)
                    action_str = json.dumps(action_dict)
                except Exception as e:
                    print(f"[FATAL] LLM call failed: {e}", flush=True)
                    raise e

                result = await env.step(action)
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_str, reward, result.done, error=None)
                if result.done:
                    break

            # Ensure score is strictly between 0 and 1 (not 0.0 or 1.0)
            raw_score = sum(rewards) / max(len(rewards), 1)
            epsilon = 1e-6
            score = max(epsilon, min(1 - epsilon, raw_score))
            success = bool(score >= 0.45)

        finally:
            if env is not None:
                await env.close()
                print(f"[DEBUG] Closed env for task: {task_name}", flush=True)

    except Exception as exc:
        print(f"[FATAL] Inference failed for {task_name}: {exc}", flush=True)

    finally:
        log_end(success, steps_taken, rewards)
        # Add delay between tasks to ensure proper cleanup
        await asyncio.sleep(1)


async def main():
    print("[DEBUG] Starting inference for all tasks...", flush=True)
    for task_name in get_task_list():
        await run_task(task_name)
    print("[DEBUG] All tasks completed", flush=True)

if __name__ == "__main__":
    asyncio.run(main())