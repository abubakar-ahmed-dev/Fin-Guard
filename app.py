from __future__ import annotations

import streamlit as st

from model import (
    VARIABLE_STATES,
    get_fraud_probability,
    get_node_marginal,
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AI Fraud Risk Estimator</h1>
        <p>Adjust transaction evidence and see the fraud probability update instantly using Bayesian reasoning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Transaction Evidence")

    location = st.selectbox("Location", VARIABLE_STATES["Location"], index=0)
    time_of_day = st.selectbox("Time", VARIABLE_STATES["Time"], index=0)
    device = st.selectbox("Device", VARIABLE_STATES["Device"], index=0)
    amount = st.selectbox("Amount", VARIABLE_STATES["Amount"], index=0)
    frequency = st.selectbox("Frequency", VARIABLE_STATES["Frequency"], index=0)

    evidence = {
        "Location": location,
        "Time": time_of_day,
        "Device": device,
        "Amount": amount,
        "Frequency": frequency,
    }

with right_col:
    st.subheader("Risk Output")
    fraud_probability = get_fraud_probability(evidence)
    risk_percent = fraud_probability * 100

    if fraud_probability < 0.30:
        risk_label = "Low"
        risk_class = "risk-low"
        summary = "The current evidence pattern is closer to normal transaction behavior."
    elif fraud_probability < 0.70:
        risk_label = "Medium"
        risk_class = "risk-medium"
        summary = "The model sees mixed evidence, so the fraud risk is moderate."
    else:
        risk_label = "High"
        risk_class = "risk-high"
        summary = "Multiple risk signals are aligned, so the model flags a high fraud likelihood."

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

st.subheader("Bayesian Insight")
insight_cols = st.columns(3)
posterior_queries = ["Location", "Time", "Device"]
for column, node in zip(insight_cols, posterior_queries):
    with column:
        posterior = get_node_marginal(node, {"Fraud": "Yes"})
        top_state = max(posterior, key=posterior.get)
        st.caption(f"Posterior given Fraud=Yes")
        st.write(f"{node}: {top_state} ({posterior[top_state] * 100:.1f}%)")

st.caption("Phase 3 focuses on the interactive UI. The network visualization will be added in the next phase.")
