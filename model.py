from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp
from typing import Dict, Iterable, List, Tuple

VARIABLE_STATES: Dict[str, List[str]] = {
    "Location": ["Local", "Foreign"],
    "Time": ["Day", "Night"],
    "Device": ["Known", "Unknown"],
    "Amount": ["Low", "Medium", "High"],
    "Frequency": ["Normal", "Unusual"],
    "Fraud": ["No", "Yes"],
}

EDGES: List[Tuple[str, str]] = [
    ("Location", "Fraud"),
    ("Time", "Fraud"),
    ("Device", "Fraud"),
    ("Amount", "Fraud"),
    ("Frequency", "Fraud"),
]

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


@dataclass(frozen=True)
class BayesianNetworkModel:
    """Bayesian network container for exact inference."""

    variables: Dict[str, List[str]]
    edges: List[Tuple[str, str]]
    priors: Dict[str, Dict[str, float]]
    fraud_cpt: List[Dict[str, object]]

    def validate(self) -> None:
        validate_priors()
        fraud_rows = len(self.fraud_cpt)
        expected_rows = 1
        for parent in PARENT_ORDER:
            expected_rows *= len(self.variables[parent])
        if fraud_rows != expected_rows:
            raise ValueError(f"Fraud CPT has {fraud_rows} rows but expected {expected_rows}.")

        for row in self.fraud_cpt:
            if abs((row["P(Fraud=No)"] + row["P(Fraud=Yes)"]) - 1.0) > 1e-9:
                raise ValueError("Fraud CPT row does not sum to 1.0.")

    def _root_assignment_probability(self, assignment: Dict[str, str]) -> float:
        probability = 1.0
        for node in PARENT_ORDER:
            probability *= self.priors[node][assignment[node]]
        return probability

    def _fraud_probability_yes(self, assignment: Dict[str, str]) -> float:
        for row in self.fraud_cpt:
            if all(row[parent] == assignment[parent] for parent in PARENT_ORDER):
                return float(row["P(Fraud=Yes)"])
        raise KeyError("No Fraud CPT row matched the provided assignment.")

    def enumerate_root_assignments(self) -> Iterable[Dict[str, str]]:
        for combo in _parent_combinations():
            yield dict(zip(PARENT_ORDER, combo))

    def query_distribution(self, node: str, evidence: Dict[str, str] | None = None) -> Dict[str, float]:
        evidence = evidence or {}
        _validate_evidence(evidence)

        if node not in self.variables:
            raise ValueError(f"Unknown node requested: {node}")

        states = self.variables[node]
        unnormalized: Dict[str, float] = {state: 0.0 for state in states}

        for assignment in self.enumerate_root_assignments():
            if any(assignment[parent] != value for parent, value in evidence.items() if parent in PARENT_ORDER):
                continue

            prior = self._root_assignment_probability(assignment)
            fraud_yes = self._fraud_probability_yes(assignment)
            evidence_probability = 1.0
            if "Fraud" in evidence:
                evidence_probability = fraud_yes if evidence["Fraud"] == "Yes" else 1.0 - fraud_yes

            if node == "Fraud":
                if "Fraud" in evidence:
                    unnormalized[evidence["Fraud"]] += prior * evidence_probability
                else:
                    unnormalized["Yes"] += prior * fraud_yes
                    unnormalized["No"] += prior * (1.0 - fraud_yes)
                continue

            for state in states:
                if assignment[node] != state:
                    continue
                unnormalized[state] += prior * evidence_probability

        total_probability = sum(unnormalized.values())
        if total_probability == 0.0:
            raise ValueError("Evidence is inconsistent with the model.")
        return {state: value / total_probability for state, value in unnormalized.items()}


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
        rows.append({**evidence, "P(Fraud=No)": 1.0 - p_yes, "P(Fraud=Yes)": p_yes})
    return rows


def validate_priors() -> None:
    """Raise ValueError if any prior distribution is invalid."""
    for variable, distribution in PRIORS.items():
        total = sum(distribution.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Prior probabilities for {variable} sum to {total}, not 1.0")


def build_model() -> BayesianNetworkModel:
    """Build and validate the Bayesian Network used for exact inference."""
    model = BayesianNetworkModel(
        variables=VARIABLE_STATES,
        edges=EDGES,
        priors=PRIORS,
        fraud_cpt=build_fraud_cpt(),
    )
    model.validate()
    return model


def _validate_evidence(evidence: Dict[str, str]) -> None:
    for node, state in evidence.items():
        if node == "Fraud":
            if state not in VARIABLE_STATES["Fraud"]:
                raise ValueError(f"Invalid state '{state}' for node 'Fraud'.")
            continue
        if node not in VARIABLE_STATES:
            raise ValueError(f"Unknown node in evidence: {node}")
        if state not in VARIABLE_STATES[node]:
            raise ValueError(f"Invalid state '{state}' for node '{node}'.")


def get_fraud_probability(evidence_dict: Dict[str, str] | None = None) -> float:
    """Return posterior P(Fraud=Yes | evidence) using exact inference."""
    evidence = evidence_dict or {}
    _validate_evidence(evidence)
    return build_model().query_distribution("Fraud", evidence)["Yes"]


def get_node_marginal(node: str, evidence_dict: Dict[str, str] | None = None) -> Dict[str, float]:
    """Return posterior distribution for any node as {state: probability}."""
    evidence = evidence_dict or {}
    _validate_evidence(evidence)
    return build_model().query_distribution(node, evidence)


if __name__ == "__main__":
    build_model()
    print("Backend initialized successfully.")
