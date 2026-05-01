# Fin-Guard: AI Fraud Risk Estimator (Merged Edition)

A unified web application that provides fraud probability estimation using two complementary approaches: **Rule-Based Expert System** and **ML-Based Trained Model**.

## Overview

This merged project combines the strengths of two approaches to fraud detection:

### 🎯 Rule-Based Model
- **Approach**: Expert-weighted probabilistic system using Rule Based fraud risk weights
- **Advantages**: 
  - Fully interpretable and transparent
  - Fast inference with no training required
  - Deterministic and reproducible
  - Great for understanding expert domain knowledge
- **How it works**: Uses weighted evidence from transaction characteristics (location, time, device, amount, frequency) to compute fraud probability via sigmoid transformation

### 🤖 ML-Based Model
- **Approach**: Machine learning model trained on real fraud transaction data from kaggle.
- **Advantages**:
  - Data-driven insights from actual fraud patterns
  - Learns complex relationships automatically
  - Adaptive to real-world fraud behavior
  - Can improve with more training data
- **How it works**: Uses pgmpy's Bayesian Network with maximum likelihood estimation trained on historical fraud data

## Features

✨ **Interactive Toggle**: Switch between models in real-time to compare predictions

📊 **Live Bayesian Network Visualization**: See how evidence propagates through the network

📈 **Comprehensive Risk Analysis**: 
- Fraud probability with risk classification (Low/Medium/High)
- Node-by-node posterior probability distributions
- Evidence impact summary with visual indicators

🎨 **Beautiful UI**: Modern Streamlit interface with gradient backgrounds and responsive design

## Installation

1. **Clone/Setup the project**:
```bash
cd d:\Fin-Guard-main
```

2. **Create and activate a virtual environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **(Optional) For ML-Based Model**: Ensure `fraud_model.pkl` is in the project root
   - If available, the ML-Based Model toggle will be enabled
   - If not available, only the Rule-Based Model will be accessible

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. **Select a Model**
   - Use the radio button at the top to choose between:
     - **Rule-Based Model**: Expert system with predefined weights
     - **ML-Based Model**: Trained on real fraud data (if available)

### 2. **Configure Transaction Evidence**
   - **Location**: Choose between "Local" (rule-based) or your region
   - **Time**: Select "Day" or "Night"
   - **Device**: Indicate if device is "Known" or "Unknown"
   - **Amount**: Choose transaction size ("Low", "Medium", "High")
   - **Frequency**: Select user activity pattern ("Normal"/"Unusual" for rule-based, "Frequent"/"Rare" for ML-based)

### 3. **Interpret Results**
   - **Fraud Probability**: The model's estimate of fraud likelihood (0-100%)
   - **Risk Level**: Categorical assessment (Low < 30%, Medium 30-70%, High > 70%)
   - **Bayesian Network**: Visualizes how evidence affects fraud node
   - **Insight Cards**: Shows probability distributions and evidence impact

## Model Comparison

| Aspect | Rule-Based | ML-Based |
|--------|-----------|----------|
| **Data Dependency** | No training data required | Requires training data |
| **Interpretability** | Fully transparent weights | Black-box (pgmpy structure visible) |
| **Accuracy** | Based on expert judgment | Based on real patterns |
| **Speed** | Very fast | Slightly slower inference |
| **Adaptability** | Manual weight updates needed | Can retrain with new data |
| **Setup** | Immediate | Requires model file |

## Technical Details

### Rule-Based Model Components
- **Priors**: Base probability distributions for each variable
- **Risk Weights**: Expert-assigned weights for evidence values (e.g., Foreign location = +1.2 log-odds)
- **Baseline Log-Odds**: Starting point (-2.0) before adding evidence
- **Inference**: Exact Bayesian inference via enumeration

### ML-Based Model Components
- **Framework**: pgmpy (Python Graphical Models)
- **Training Method**: Maximum Likelihood Estimation (MLE)
- **Network**: Bayesian Network with fraud as child node
- **Inference**: Variable Elimination algorithm
- **Caching**: Pickled model for fast loading

### Variable States

- Location: [Local, Foreign]
- Time: [Day, Night]
- Device: [Known, Unknown]
- Amount: [Low, Medium, High]
- Frequency: [Frequent, Rare]

