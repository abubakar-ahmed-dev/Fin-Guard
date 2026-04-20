from __future__ import annotations

import sys
from typing import Dict, Tuple

from model import VARIABLE_STATES, get_fraud_probability, get_node_marginal


Scenario = Tuple[str, Dict[str, str], float, float]


SCENARIOS: list[Scenario] = [
    (
        "low_risk_baseline",
        {
            "Location": "Local",
            "Time": "Day",
            "Device": "Known",
            "Amount": "Low",
            "Frequency": "Normal",
        },
        0.0,
        0.30,
    ),
    (
        "high_risk_profile",
        {
            "Location": "Foreign",
            "Time": "Night",
            "Device": "Unknown",
            "Amount": "High",
            "Frequency": "Unusual",
        },
        0.70,
        1.0,
    ),
    (
        "mixed_profile",
        {
            "Location": "Local",
            "Time": "Night",
            "Device": "Known",
            "Amount": "Medium",
            "Frequency": "Normal",
        },
        0.20,
        0.50,
    ),
]


def check_probability_bounds() -> None:
    for scenario_name, evidence, _, _ in SCENARIOS:
        probability = get_fraud_probability(evidence)
        if not 0.0 <= probability <= 1.0:
            raise AssertionError(f"{scenario_name}: fraud probability out of bounds: {probability}")


def check_node_marginals() -> None:
    for node in VARIABLE_STATES:
        posterior = get_node_marginal(node, {"Fraud": "Yes"})
        total = sum(posterior.values())
        if abs(total - 1.0) > 1e-9:
            raise AssertionError(f"{node}: posterior does not sum to 1, got {total}")


def check_scenarios() -> None:
    for scenario_name, evidence, lower, upper in SCENARIOS:
        probability = get_fraud_probability(evidence)
        if not (lower <= probability <= upper):
            raise AssertionError(
                f"{scenario_name}: expected probability between {lower:.2f} and {upper:.2f}, got {probability:.6f}"
            )
        print(f"{scenario_name}: {probability:.6f} OK")


def main() -> int:
    print("Running Phase 5 validation suite...\n")
    check_probability_bounds()
    check_node_marginals()
    check_scenarios()
    print("\nAll Phase 5 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())