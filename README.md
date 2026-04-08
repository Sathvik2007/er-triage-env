---
title: ER Triage RL
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
app_file: inference.py
pinned: false
---

# ER-Triage-Env

**OpenEnv Hackathon Submission**

**Team Members:**
- Nikhil Krishna Sathvik
- Disha K S
- Harshini Bhushan

Realistic Emergency Room (ER) triage and resource allocation simulator.  
Patients have different severity levels and types (trauma, cardiac, etc.). Resources are limited. Patients worsen over time if ignored. The agent must learn to make ethical, optimal decisions under scarcity.

## Why This Environment
- Models a real ER bottleneck: continuous arrivals with limited beds, ICU, and doctors.
- Captures sequential trade-offs: saving resources now can improve later survival.
- Includes deterministic scenario seeds for reproducible benchmark scores.

## API and OpenEnv Spec
- `reset(task=...)` starts one episode and returns typed `Observation`.
- `step(TriageAction)` applies allocation and returns `StepResult`.
- `state()` returns current typed `Observation` without advancing time.
- Environment server entrypoint: `server.app:create_app`.
- Tasks declared in `openenv.yaml`: `simple-triage`, `resource-constraint`, `critical-overload`.

## Observation Space
The observation is:
- `patients`: list of patients with `id`, `severity`, `waiting_time`, `patient_type`, `treated`.
- `resources`: currently available `beds`, `icu`, `doctors`.
- `time_step`: current episode step.
- `message`: environment status text.

## Action Space
Use the typed model `TriageAction`:
- `{"action_type":"assign","patient_id":<int>,"resource":"bed|icu|doctor"}`
- `{"action_type":"wait"}`

## Reward Design
- Positive: treat severe patients, correct ICU allocation, stabilize high-risk patients.
- Negative: patient death, invalid resource allocation, excessive waiting, idle behavior.
- Episode grader score in `[0.0, 1.0]` combines:
  - survival rate (50%)
  - waiting-time quality score (30%)
  - resource utilization (20%)

## Tasks and Difficulty Progression
- `simple-triage` (easy): low load, enough resources, no arrivals.
- `resource-constraint` (medium): resource scarcity with moderate stochastic arrivals.
- `critical-overload` (hard): severe arrivals with sustained pressure and limited ICU.

## Run Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
python -m server.app
```

In another terminal:
```bash
TASK_NAME=simple-triage ENV_BASE_URL=http://localhost:8000 python inference.py
```

## Docker
```bash
docker build -t er-triage .
docker run --rm -p 8000:8000 er-triage
```

## Pre-Submission Validation
```bash
./venv/bin/openenv validate
```

If deploying to Hugging Face Spaces, set:
- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY`