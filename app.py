from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgb
import streamlit as st

from model import (
    EDGES,
    VARIABLE_STATES,
    get_fraud_probability,
    get_node_marginal,
    initialize_variables,
    ML_VARIABLE_STATES,
    USE_CACHE,
)

st.set_page_config(
    page_title="AI Fraud Risk Estimator",
    page_icon="\U0001F4B3",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f4ef 0%, #ffffff 45%, #f2f6fb 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #102a43 0%, #1f4e79 55%, #2c7a7b 100%);
        color: white;
        box-shadow: 0 14px 40px rgba(16, 42, 67, 0.22);
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.45rem 0 0;
        opacity: 0.92;
        font-size: 1.0rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: white;
        border: 1px solid rgba(16, 42, 67, 0.09);
        box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
    }
    .risk-low {
        color: #0f7a3a;
        font-weight: 700;
    }
    .risk-medium {
        color: #b7791f;
        font-weight: 700;
    }
    .risk-high {
        color: #c53030;
        font-weight: 700;
    }
    .insight-card {
        padding: 0.95rem 1rem;
        border-radius: 0.9rem;
        background: #ffffff;
        border: 1px solid rgba(16, 42, 67, 0.1);
        box-shadow: 0 8px 20px rgba(16, 42, 67, 0.07);
        margin-bottom: 0.75rem;
    }
    .chip {
        display: inline-block;
        margin: 0.22rem 0.28rem 0 0;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .chip-low {
        background: #def7e7;
        color: #136f3f;
    }
    .chip-mid {
        background: #fff1d6;
        color: #9a640f;
    }
    .chip-high {
        background: #fde0e0;
        color: #a12626;
    }
    .model-toggle {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 0.75rem;
        border: 1px solid rgba(16, 42, 67, 0.1);
    }
    .model-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .badge-rule {
        background: #e3f2fd;
        color: #1565c0;
    }
    .badge-ml {
        background: #f3e5f5;
        color: #6a1b9a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Initialize Model Selection
# ============================================================

if "model_type" not in st.session_state:
    st.session_state.model_type = "rule-based"

# Check if ML model is available
ml_available = USE_CACHE

GRAPH_POSITIONS = {
    "Location": (-2.15, 0.95),
    "Time": (-0.95, 1.82),
    "Device": (0.95, 1.82),
    "Amount": (2.15, 0.95),
    "Frequency": (0.0, -0.15),
    "Fraud": (0.0, 0.68),
}

NODE_COLOR_PALETTE = {
    "Location": {"Local": "#1a925a", "Foreign": "#e75151"},
    "Time": {"Day": "#1a925a", "Night": "#e75151"},
    "Device": {"Known": "#1a925a", "Unknown": "#e75151"},
    "Amount": {"Low": "#1a925a", "Medium": "#e7ab51", "High": "#e75151"},
    "Frequency": {"Frequent": "#1a925a", "Rare": "#e75151"},
}

# Support for ML-based variable states
if ml_available:
    NODE_COLOR_PALETTE["Frequency"] = {
        "Frequent": "#1a925a",
        "Rare": "#e75151",
    }

NODE_LABELS = {
    "Location": "Location",
    "Time": "Time",
    "Device": "Device",
    "Amount": "Amount",
    "Frequency": "Frequency",
    "Fraud": "Fraud",
}

EVIDENCE_SIGNAL_SCORE = {
    "Location": {"Local": 0, "Foreign": 2},
    "Time": {"Day": 0, "Night": 1},
    "Device": {"Known": 0, "Unknown": 2},
    "Amount": {"Low": 0, "Medium": 1, "High": 2},
    "Frequency": {"Frequent": 0, "Rare": 2},
}

# Add ML-based signal scores if available
if ml_available:
    EVIDENCE_SIGNAL_SCORE["Frequency"] = {
        "Frequent": 0,
        "Rare": 2,
    }


def build_evidence(model_type: str) -> Dict[str, str]:
    """Build evidence dictionary based on the selected model type."""
    current_vars = VARIABLE_STATES if VARIABLE_STATES else (ML_VARIABLE_STATES if ml_available else {})
    
    return {
        "Location": st.session_state.get("location", current_vars.get("Location", ["Local"])[0]),
        "Time": st.session_state.get("time_of_day", current_vars.get("Time", ["Day"])[0]),
        "Device": st.session_state.get("device", current_vars.get("Device", ["Known"])[0]),
        "Amount": st.session_state.get("amount", current_vars.get("Amount", ["Low"])[0]),
        "Frequency": st.session_state.get("frequency", current_vars.get("Frequency", ["Frequent"])[0]),
    }


def risk_metadata(fraud_probability: float) -> tuple[str, str, str]:
    if fraud_probability < 0.30:
        return (
            "Low",
            "risk-low",
            "The current evidence pattern is closer to normal transaction behavior.",
        )
    if fraud_probability < 0.70:
        return (
            "Medium",
            "risk-medium",
            "The model sees mixed evidence, so the fraud risk is moderate.",
        )
    return (
        "High",
        "risk-high",
        "Multiple risk signals are aligned, so the model flags a high fraud likelihood.",
    )


def state_probability_text(node: str, posterior: Dict[str, float], evidence: Dict[str, str]) -> str:
    if node in evidence:
        return f"{NODE_LABELS[node]}:\n{evidence[node]}"

    top_state = max(posterior, key=posterior.get)
    if node == "Fraud":
        return f"Fraud\nYes {posterior['Yes'] * 100:.1f}%"

    return f"{NODE_LABELS[node]}\n{top_state}"


def fraud_fill_color(probability: float) -> str:
    if probability < 0.30:
        return "#95d5b2"
    if probability < 0.70:
        return "#ffd166"
    return "#ef476f"


def blend_with_white(hex_color: str, amount: float) -> tuple[float, float, float]:
    red, green, blue = to_rgb(hex_color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def node_style(
    node: str,
    evidence: Dict[str, str],
    posterior: Dict[str, float],
    fraud_probability: float,
) -> tuple[tuple[float, float, float], str, str, float]:
    if node == "Fraud":
        base_color = fraud_fill_color(fraud_probability)
        border_color = "#1b998b" if fraud_probability < 0.30 else "#f4a261" if fraud_probability < 0.70 else "#d62828"
        return blend_with_white(base_color, 0.18), border_color, "#102a43", 0.26

    if node == "Frequency":
        if node in evidence:
            base_color = NODE_COLOR_PALETTE[node].get(evidence[node], "#1a925a")
            return blend_with_white(base_color, 0.0), base_color, "white", 0.22

        top_state = max(posterior, key=posterior.get)
        base_color = NODE_COLOR_PALETTE[node].get(top_state, "#1a925a")
        confidence = posterior[top_state]
        lightness = 0.78 - min(confidence, 0.9) * 0.28
        return blend_with_white(base_color, lightness), base_color, "#102a43", 0.21

    if node in evidence:
        base_color = NODE_COLOR_PALETTE[node].get(evidence[node], "#1a925a")
        return blend_with_white(base_color, 0.0), base_color, "white", 0.19

    top_state = max(posterior, key=posterior.get)
    base_color = NODE_COLOR_PALETTE[node].get(top_state, "#1a925a")
    confidence = posterior[top_state]
    lightness = 0.78 - min(confidence, 0.9) * 0.28
    return blend_with_white(base_color, lightness), base_color, "#102a43", 0.18


def render_bayesian_network(evidence: Dict[str, str], model_type: str) -> None:
    graph = nx.DiGraph()
    current_vars = VARIABLE_STATES if VARIABLE_STATES else (ML_VARIABLE_STATES if ml_available else {})
    
    graph.add_nodes_from(current_vars.keys())
    graph.add_edges_from(EDGES)

    fraud_probability = get_fraud_probability(evidence, model_type=model_type)
    node_posteriors = {node: get_node_marginal(node, evidence, model_type=model_type) for node in current_vars}

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    ax.set_title("Bayesian Network with Live Evidence Propagation", fontsize=11.2, pad=8, fontweight="bold")
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(-0.55, 2.05)

    for source, target in graph.edges():
        start_x, start_y = GRAPH_POSITIONS[source]
        end_x, end_y = GRAPH_POSITIONS[target]
        source_state = evidence.get(source, max(node_posteriors[source], key=node_posteriors[source].get))
        source_color = NODE_COLOR_PALETTE[source].get(source_state, "#1a925a")
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="->", color=source_color, lw=1.8, shrinkA=22, shrinkB=20, alpha=0.9),
        )

    for node, (x_coord, y_coord) in GRAPH_POSITIONS.items():
        face_color, edge_color, text_color, radius = node_style(node, evidence, node_posteriors[node], fraud_probability)
        line_width = 2.0 if node == "Fraud" else 1.6 if node in evidence else 1.25

        circle = plt.Circle((x_coord, y_coord), radius, facecolor=face_color, edgecolor=edge_color, linewidth=line_width, zorder=2)
        ax.add_patch(circle)
        ax.text(
            x_coord,
            y_coord,
            state_probability_text(node, node_posteriors[node], evidence),
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=text_color,
            zorder=3,
            linespacing=1.0,
        )

    fig.tight_layout(pad=0.15)
    st.pyplot(fig, width="stretch", clear_figure=True)


# ============================================================
# UI LAYOUT
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI Fraud Risk Estimator</h1>
        <p>Adjust transaction evidence and see the fraud probability update instantly using Bayesian reasoning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Model Toggle Section
st.markdown("<div class='model-toggle'>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 0.5])

with col1:
    st.write("**Select Analysis Model:**")

with col2:
    model_options = ["Rule-Based Model", "ML-Based Model"] if ml_available else ["Rule-Based Model"]
    default_idx = 0 if st.session_state.model_type == "rule-based" else 1
    selected_model = st.radio(
        "Model Type",
        options=model_options,
        index=min(default_idx, len(model_options) - 1),
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Update model type in session state
if selected_model == "Rule-Based Model":
    st.session_state.model_type = "rule-based"
elif selected_model == "ML-Based Model":
    st.session_state.model_type = "ml-based"

# Initialize variables for the selected model
initialize_variables(st.session_state.model_type)

# Display model info badge
if st.session_state.model_type == "rule-based":
    st.markdown(
        '<span class="model-badge badge-rule">📊 Using Rule-Based Expert System</span>',
        unsafe_allow_html=True,
    )
    model_description = "This model uses expert-defined rules and weights to estimate fraud probability based on transaction characteristics."
else:
    st.markdown(
        '<span class="model-badge badge-ml">🤖 Using ML-Trained Model</span>',
        unsafe_allow_html=True,
    )
    model_description = "This model uses a machine learning approach trained on real fraud data to provide data-driven fraud probability estimates."

st.caption(model_description)

st.divider()

# Main Input Section
left_col, right_col = st.columns([1.05, 0.95], gap="large")

current_vars = VARIABLE_STATES if VARIABLE_STATES else (ML_VARIABLE_STATES if ml_available else {})

with left_col:
    st.subheader("Transaction Evidence")
    st.selectbox("Location", current_vars.get("Location", ["Local"]), index=0, key="location")
    st.selectbox("Time", current_vars.get("Time", ["Day"]), index=0, key="time_of_day")
    st.selectbox("Device", current_vars.get("Device", ["Known"]), index=0, key="device")
    st.selectbox("Amount", current_vars.get("Amount", ["Low"]), index=0, key="amount")
    st.selectbox("Frequency", current_vars.get("Frequency", ["Frequent"]), index=0, key="frequency")

    evidence = build_evidence(st.session_state.model_type)

with right_col:
    st.subheader("Risk Output")
    fraud_probability = get_fraud_probability(evidence, model_type=st.session_state.model_type)
    risk_percent = fraud_probability * 100
    risk_label, risk_class, summary = risk_metadata(fraud_probability)

    metric_a, metric_b = st.columns(2)
    with metric_a:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.95rem;color:#52616b;">Fraud Probability</div>
                <div style="font-size:2.2rem;font-weight:800;line-height:1.1;">{risk_percent:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_b:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.95rem;color:#52616b;">Risk Level</div>
                <div class="{risk_class}" style="font-size:2.2rem;line-height:1.1;">{risk_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.95rem;color:#52616b;margin-bottom:0.35rem;">Model Interpretation</div>
            <div style="font-size:1rem;line-height:1.6;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader("Selected Evidence")
selected_cols = st.columns(5)
for column, (name, value) in zip(selected_cols, evidence.items()):
    with column:
        st.metric(name, value)

st.subheader("Bayesian Network Visualization")
render_bayesian_network(evidence, st.session_state.model_type)

st.subheader("Bayesian Insight")
fraud_posterior = get_node_marginal("Fraud", evidence, model_type=st.session_state.model_type)
yes_prob = fraud_posterior["Yes"]
no_prob = fraud_posterior["No"]

signal_total = sum(EVIDENCE_SIGNAL_SCORE.get(node, {}).get(value, 0) for node, value in evidence.items())
if signal_total <= 2:
    signal_label = "Mostly Safe Pattern"
    signal_class = "chip-low"
elif signal_total <= 5:
    signal_label = "Mixed Risk Pattern"
    signal_class = "chip-mid"
else:
    signal_label = "High Alert Pattern"
    signal_class = "chip-high"

insight_left, insight_right = st.columns([1.15, 0.85], gap="large")

with insight_left:
    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:0.92rem;color:#52616b;">Fraud Posterior Breakdown</div>
            <div style="font-size:1.5rem;font-weight:800;color:#102a43;margin-top:0.15rem;">Yes: {yes_prob * 100:.2f}%</div>
            <div style="font-size:0.9rem;color:#52616b;margin-top:0.15rem;">No: {no_prob * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(yes_prob * 100), text="Fraud Yes probability")

    amount_posterior = get_node_marginal("Amount", evidence, model_type=st.session_state.model_type)
    top_amount = max(amount_posterior, key=amount_posterior.get)
    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:0.92rem;color:#52616b;">Most Likely Transaction Profile Under Current Evidence</div>
            <div style="font-size:1rem;font-weight:700;color:#102a43;margin-top:0.3rem;">Amount: {top_amount} ({amount_posterior[top_amount] * 100:.1f}%)</div>
            <div style="font-size:0.9rem;color:#52616b;margin-top:0.2rem;">Center node displays P(Fraud=Yes), while input nodes show selected state only.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with insight_right:
    chips_html = ""
    for node, value in evidence.items():
        level = EVIDENCE_SIGNAL_SCORE.get(node, {}).get(value, 0)
        chip_class = "chip-high" if level == 2 else "chip-mid" if level == 1 else "chip-low"
        chips_html += f'<span class="chip {chip_class}">{node}: {value}</span>'

    st.markdown(
        f"""
        <div class="insight-card">
            <div style="font-size:0.92rem;color:#52616b;">Evidence Impact Summary</div>
            <div style="margin-top:0.3rem;"><span class="chip {signal_class}">{signal_label}</span></div>
            <div style="margin-top:0.4rem;">{chips_html}</div>
            <div style="font-size:0.88rem;color:#52616b;margin-top:0.55rem;">Red-style chips represent stronger fraud signals, green-style chips represent safer signals.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Node colors update with your selected evidence, while the center Fraud node reflects the current risk state.")

if not ml_available:
    st.warning("⚠️ ML-Based Model is not available. To enable it, ensure `fraud_model.pkl` is in the project directory.")
