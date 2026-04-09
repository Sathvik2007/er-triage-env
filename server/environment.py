import uuid
import random
from typing import Optional, Tuple

from openenv.core.env_server import Environment

from models import (
    TriageAction,
    Observation,
    Patient,
    Resources,
    EpisodeMetrics,
    TaskGrade,
    State,
)
from server.scenarios import get_scenario


def _clamp(x: float) -> float:
    return max(0.01, min(0.99, round(x, 4)))


class ERTriageEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.episode_id: Optional[str] = None
        self.scenario: Optional[dict] = None
        self.patients: dict[int, Patient] = {}
        self.resources: dict[str, int] = {}
        self.resource_capacity: dict[str, int] = {}
        self.resource_usage: dict[str, int] = {}
        self.time_step: int = 0
        self.survived: int = 0
        self.deaths: int = 0
        self.total_wait_time_sum: float = 0.0
        self.total_wait_updates: int = 0
        self.total_patients_seen: int = 0
        self._rng: random.Random = random.Random(0)
        self._task: str = "simple-triage"
        self._last_grade: Optional[TaskGrade] = None

    def reset(self, task: str = "simple-triage", **kwargs) -> Observation:
        # handle task passed via difficulty= or other kwargs
        if task == "simple-triage":
            task = kwargs.get("difficulty", kwargs.get("task_name", task))
        self.episode_id = str(uuid.uuid4())
        self._task = task
        self.scenario = get_scenario(task)
        self.patients = {p["id"]: Patient(**p) for p in self.scenario["initial_patients"]}
        self.resources = self.scenario["initial_resources"].copy()
        self.resource_capacity = self.scenario["initial_resources"].copy()
        self.resource_usage = {"beds": 0, "icu": 0, "doctors": 0}
        self.time_step = 0
        self.survived = 0
        self.deaths = 0
        self.total_wait_time_sum = 0.0
        self.total_wait_updates = 0
        self.total_patients_seen = len(self.patients)
        self._rng = random.Random(self.scenario.get("seed", 0))
        self._last_grade = None

        obs = self._build_observation("New episode started. Prioritize highest severity patients.")
        obs.reward = 0.01
        obs.done = False
        return obs

    def state(self) -> State:
        return State(
            episode_id=self.episode_id or "",
            step_count=self.time_step,
        )

    def step(self, action: TriageAction) -> Observation:
        # guard against None scenario
        if self.scenario is None:
            self.scenario = get_scenario(self._task)
            self.patients = {p["id"]: Patient(**p) for p in self.scenario["initial_patients"]}
            self.resources = self.scenario["initial_resources"].copy()
            self.resource_capacity = self.scenario["initial_resources"].copy()

        reward = 0.0
        message = "Action processed."
        step_deaths = 0
        step_survivors = 0

        if action.action_type == "assign" and action.patient_id is not None and action.resource:
            alloc_reward, msg, survivors = self._allocate_resource(action.patient_id, action.resource)
            reward += alloc_reward
            step_survivors += survivors
            message = msg
        else:
            reward -= 0.15
            message = "No allocation this step."

        new_deaths = self._update_patient_conditions()
        step_deaths += new_deaths
        self._add_new_patients()
        self.time_step += 1

        reward += self._compute_reward(step_deaths=step_deaths, step_survivors=step_survivors)
        self._reset_step_resources()

        done = self.time_step >= self.scenario.get("max_steps", 50) or len(self.patients) == 0
        if done:
            self._last_grade = self.grade_task()
            message = f"{message} Episode score={self._last_grade.score:.3f}"

        obs = self._build_observation(message)
        final_reward = round(reward, 2)
        if done:
            final_reward = _clamp(self._last_grade.score)
        obs.reward = final_reward
        obs.done = done
        return obs

    def _resource_key(self, action_resource: str) -> str:
        mapping = {"bed": "beds", "icu": "icu", "doctor": "doctors"}
        return mapping[action_resource]

    def _allocate_resource(self, pid: int, res: str) -> Tuple[float, str, int]:
        resource_key = self._resource_key(res)
        if pid not in self.patients:
            return -0.7, "Invalid patient id - penalty applied.", 0
        if self.resources.get(resource_key, 0) <= 0:
            return -0.7, "Selected resource unavailable - penalty applied.", 0

        patient = self.patients[pid]
        bonus = 0.0

        if patient.severity > 0.8 and res == "icu":
            bonus = 1.0
        elif patient.severity > 0.55:
            bonus = 0.55

        if res == "doctor":
            if patient.patient_type in ["trauma", "cardiac"]:
                bonus += 0.35

        patient.treated = True
        patient.severity = max(0.0, patient.severity - 0.70)
        self.resources[resource_key] -= 1
        self.resource_usage[resource_key] += 1
        survivors = 0

        if patient.severity <= 0.3:
            self.survived += 1
            survivors = 1
            del self.patients[pid]
            return 0.5 + bonus + 0.5, f"Assigned {res} to patient {pid}; patient stabilized and discharged.", survivors

        return 0.5 + bonus, f"Assigned {res} to patient {pid}.", survivors

    def _update_patient_conditions(self) -> int:
        to_remove = []
        deaths_this_step = 0
        for pid, p in list(self.patients.items()):
            if not p.treated:
                p.waiting_time += 5
                rates = {"trauma": 0.065, "cardiac": 0.052, "infection": 0.038, "general": 0.028}
                p.severity += rates[p.patient_type] * (p.waiting_time / 25.0)
                if p.severity > 1.0:
                    self.deaths += 1
                    deaths_this_step += 1
                    to_remove.append(pid)
            else:
                p.treated = False

        for pid in to_remove:
            del self.patients[pid]
        return deaths_this_step

    def _add_new_patients(self):
        if self.scenario is None:
            return
        max_patients = self.scenario.get("max_patients", 999)
        if self.total_patients_seen >= max_patients:
            return
        if self._rng.random() < self.scenario.get("arrival_rate", 0.0):
            new_id = max(self.patients.keys(), default=0) + 1
            sev = self._rng.uniform(0.55, 0.97)
            typ = self._rng.choice(["trauma", "cardiac", "infection", "general"])
            self.patients[new_id] = Patient(id=new_id, severity=sev, waiting_time=0, patient_type=typ)
            self.total_patients_seen += 1

    def _compute_reward(self, step_deaths: int, step_survivors: int) -> float:
        r = 0.0
        r -= step_deaths * 2.0
        avg_wait = sum(p.waiting_time for p in self.patients.values()) / max(len(self.patients), 1)
        r -= min(avg_wait / 120.0, 1.2) * 0.35
        r += step_survivors * 0.8
        return r

    def _reset_step_resources(self) -> None:
        self.resources = self.resource_capacity.copy()

    def _episode_metrics(self) -> EpisodeMetrics:
        avg_waiting_time = self.total_wait_time_sum / max(self.total_wait_updates, 1)
        total_usage = sum(self.resource_usage.values())
        max_possible_usage = max(self.time_step, 1) * max(sum(self.resource_capacity.values()), 1)
        utilization = min(total_usage / max_possible_usage, 1.0)
        survival_rate = self.survived / max(self.total_patients_seen, 1)
        wait_score = max(0.0, 1.0 - min(avg_waiting_time / 60.0, 1.0))
        raw_score = (0.5 * survival_rate) + (0.3 * wait_score) + (0.2 * utilization)
        final_score = _clamp(raw_score)
        return EpisodeMetrics(
            survived=self.survived,
            deaths=self.deaths,
            total_patients_seen=self.total_patients_seen,
            avg_waiting_time=round(avg_waiting_time, 2),
            utilization=round(utilization, 4),
            survival_rate=round(survival_rate, 4),
            wait_score=round(wait_score, 4),
            final_score=final_score,
        )

    def grade_task(self) -> TaskGrade:
        metrics = self._episode_metrics()
        return TaskGrade(
            task=self._task,
            score=_clamp(metrics.final_score),
            metrics={
                "survival_rate": metrics.survival_rate,
                "wait_score": metrics.wait_score,
                "utilization": metrics.utilization,
            },
        )

    def _build_observation(self, message: Optional[str] = None) -> Observation:
        active = list(self.patients.values())
        self.total_wait_time_sum += sum(p.waiting_time for p in active)
        self.total_wait_updates += len(active)
        return Observation(
            patients=active,
            resources=Resources(**self.resources),
            time_step=self.time_step,
            message=message or f"Step {self.time_step} - {len(active)} patients active."
        )