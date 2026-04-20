from __future__ import annotations

from itertools import product
from math import exp
from typing import Dict, Iterable, List, Tuple

# Canonical variable ordering used across the project.
VARIABLE_STATES: Dict[str, List[str]] = {
    "Location": ["Local", "Foreign"],
    "Time": ["Day", "Night"],
    "Device": ["Known", "Unknown"],
    "Amount": ["Low", "Medium", "High"],
    "Frequency": ["Normal", "Unusual"],
    "Fraud": ["No", "Yes"],
}

# DAG edges (all roots point to Fraud in baseline model).
EDGES: List[Tuple[str, str]] = [
    ("Location", "Fraud"),
    ("Time", "Fraud"),
    ("Device", "Fraud"),
    ("Amount", "Fraud"),
    ("Frequency", "Fraud"),
]

# Priors for root nodes.
PRIORS: Dict[str, Dict[str, float]] = {
    "Location": {"Local": 0.85, "Foreign": 0.15},
    "Time": {"Day": 0.70, "Night": 0.30},
    "Device": {"Known": 0.80, "Unknown": 0.20},
    "Amount": {"Low": 0.60, "Medium": 0.30, "High": 0.10},
    "Frequency": {"Normal": 0.88, "Unusual": 0.12},
}

BASELINE_LOG_ODDS = -2.0
RISK_WEIGHTS = {
    "Location": {"Local": 0.0, "Foreign": 1.2},
    "Time": {"Day": 0.0, "Night": 0.6},
    "Device": {"Known": 0.0, "Unknown": 1.0},
    "Amount": {"Low": 0.0, "Medium": 0.5, "High": 1.3},
    "Frequency": {"Normal": 0.0, "Unusual": 0.9},
}

PARENT_ORDER = ["Location", "Time", "Device", "Amount", "Frequency"]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def fraud_probability_yes(evidence: Dict[str, str]) -> float:
    """Return P(Fraud=Yes | evidence) from the expert weighted rule."""
    score = BASELINE_LOG_ODDS
    for parent in PARENT_ORDER:
        score += RISK_WEIGHTS[parent][evidence[parent]]
    return _sigmoid(score)


def _parent_combinations() -> Iterable[Tuple[str, str, str, str, str]]:
    return product(*(VARIABLE_STATES[parent] for parent in PARENT_ORDER))


def build_fraud_cpt() -> List[Dict[str, object]]:
    """Build full CPT rows for Fraud node over all parent combinations."""
    rows: List[Dict[str, object]] = []
    for combo in _parent_combinations():
        evidence = dict(zip(PARENT_ORDER, combo))
        p_yes = fraud_probability_yes(evidence)
        rows.append(
            {
                **evidence,
                "P(Fraud=No)": round(1.0 - p_yes, 6),
                "P(Fraud=Yes)": round(p_yes, 6),
            }
        )
    return rows


def validate_priors() -> None:
    """Raise ValueError if any prior distribution is invalid."""
    for variable, distribution in PRIORS.items():
        total = sum(distribution.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Prior probabilities for {variable} sum to {total}, not 1.0")


def sanity_examples() -> Dict[str, float]:
    """Canonical scenarios for quick manual sanity checks."""
    scenarios = {
        "low_risk": {
            "Location": "Local",
            "Time": "Day",
            "Device": "Known",
            "Amount": "Low",
            "Frequency": "Normal",
        },
        "high_risk": {
            "Location": "Foreign",
            "Time": "Night",
            "Device": "Unknown",
            "Amount": "High",
            "Frequency": "Unusual",
        },
        "mixed_risk": {
            "Location": "Local",
            "Time": "Night",
            "Device": "Known",
            "Amount": "Medium",
            "Frequency": "Normal",
        },
    }
    return {name: round(fraud_probability_yes(evidence), 6) for name, evidence in scenarios.items()}


PHASE1_FRAUD_CPT = build_fraud_cpt()


if __name__ == "__main__":
    validate_priors()
    print("Phase 1 model artifacts initialized successfully.")
    print(f"Total Fraud CPT rows: {len(PHASE1_FRAUD_CPT)}")
    print("Sanity examples P(Fraud=Yes):")
    for scenario_name, probability in sanity_examples().items():
        print(f"  - {scenario_name}: {probability:.4f}")
