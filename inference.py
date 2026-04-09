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

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME = os.getenv("TASK_NAME", "simple-triage")
BENCHMARK = "er-triage"
MAX_STEPS = 70
TEMPERATURE = 0.15
MAX_TOKENS = 220
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

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

async def main():
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    env = None
    try:
        env = HTTPEnvClient(base_url=ENV_BASE_URL)
        await env.connect()
        result = await env.reset(task=TASK_NAME)
        try:
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                obs_data = result.observation
                obs_dict = obs_data.model_dump() if hasattr(obs_data, "model_dump") else obs_data
                user_prompt = f"Current observation:\n{json.dumps(obs_dict, indent=2)}\nChoose next action."
                print("Calling LLM...", flush=True)
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
                action_dict = normalize_action(json.loads(text))
                action = TriageAction(**action_dict)
                action_str = json.dumps(action_dict)
                result = await env.step(action)
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                log_step(step, action_str, reward, result.done, error=None)
                if result.done:
                    break
            score = sum(rewards) / max(len(rewards), 1)
            success = bool(score >= 0.45)
        finally:
            if env is not None:
                await env.close()
    except Exception as exc:
        print(f"[DEBUG] Inference failed before/at reset: {exc}", flush=True)
    finally:
        log_end(success, steps_taken, rewards)

if __name__ == "__main__":
    asyncio.run(main())