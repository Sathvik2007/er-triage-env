from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class Patient(BaseModel):
    id: int
    severity: float = Field(..., ge=0.0, le=1.5)
    waiting_time: int = 0
    patient_type: Literal["trauma", "cardiac", "infection", "general"] = "general"
    treated: bool = False

class Resources(BaseModel):
    beds: int = 0
    icu: int = 0
    doctors: int = 0

class TriageAction(BaseModel):
    action_type: Literal["assign", "wait"]
    patient_id: Optional[int] = None
    resource: Optional[Literal["bed", "icu", "doctor"]] = None
    reason: Optional[str] = None

class Observation(BaseModel):
    patients: List[Patient]
    resources: Resources
    time_step: int
    message: Optional[str] = None
    reward: Optional[float] = None
    done: bool = False

class EnvInfo(BaseModel):
    episode_id: str
    task: str


class EpisodeMetrics(BaseModel):
    survived: int
    deaths: int
    total_patients_seen: int
    avg_waiting_time: float
    utilization: float = Field(..., ge=0.0, le=1.0)
    survival_rate: float = Field(..., ge=0.0, le=1.0)
    wait_score: float = Field(..., ge=0.0, le=1.0)
    final_score: float = Field(..., ge=0.0, le=1.0)


class TaskGrade(BaseModel):
    task: str
    score: float = Field(..., ge=0.0, le=1.0)
    metrics: Dict[str, float]