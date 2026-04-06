from typing import Dict, Any

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "simple-triage": {
        "initial_patients": [
            {"id": 1, "severity": 0.85, "waiting_time": 0, "patient_type": "trauma"},
            {"id": 2, "severity": 0.45, "waiting_time": 0, "patient_type": "general"},
            {"id": 3, "severity": 0.65, "waiting_time": 0, "patient_type": "cardiac"},
        ],
        "initial_resources": {"beds": 3, "icu": 2, "doctors": 4},
        "max_steps": 25,
        "arrival_rate": 0.0,
        "seed": 7,
        "max_patients": 6,
    },
    "resource-constraint": {
        "initial_patients": [
            {"id": 1, "severity": 0.92, "waiting_time": 0, "patient_type": "trauma"},
            {"id": 2, "severity": 0.78, "waiting_time": 0, "patient_type": "cardiac"},
            {"id": 3, "severity": 0.55, "waiting_time": 0, "patient_type": "infection"},
            {"id": 4, "severity": 0.62, "waiting_time": 0, "patient_type": "general"},
            {"id": 5, "severity": 0.48, "waiting_time": 0, "patient_type": "general"},
        ],
        "initial_resources": {"beds": 2, "icu": 1, "doctors": 3},
        "max_steps": 45,
        "arrival_rate": 0.25,
        "seed": 23,
        "max_patients": 14,
    },
    "critical-overload": {
        "initial_patients": [
            {"id": 1, "severity": 0.95, "waiting_time": 0, "patient_type": "trauma"},
            {"id": 2, "severity": 0.89, "waiting_time": 0, "patient_type": "cardiac"},
            {"id": 3, "severity": 0.82, "waiting_time": 0, "patient_type": "trauma"},
            {"id": 4, "severity": 0.71, "waiting_time": 0, "patient_type": "infection"},
        ],
        "initial_resources": {"beds": 3, "icu": 1, "doctors": 2},
        "max_steps": 65,
        "arrival_rate": 0.55,
        "seed": 61,
        "max_patients": 20,
    }
}

def get_scenario(task: str) -> Dict[str, Any]:
    return SCENARIOS.get(task, SCENARIOS["simple-triage"]).copy()