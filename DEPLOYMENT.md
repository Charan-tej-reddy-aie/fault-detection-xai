# 🏭 Industrial Machine Fault Detection & Explainable AI (XAI) System
## Production Deployment, Validation & Architecture Guide

---

### 📌 Table of Contents
1. [Executive Summary & Key Benchmarks](#1-executive-summary--key-benchmarks)
2. [Data Leakage & Model Validation Audit (Step 1)](#2-data-leakage--model-validation-audit-step-1)
3. [Production Real-Time Prediction Engine (`predictor.py`) (Step 2)](#3-production-real-time-prediction-engine-predictorpy-step-2)
4. [Explainable AI (XAI) Dynamic SHAP Root-Cause Attribution (Step 3)](#4-explainable-ai-xai-dynamic-shap-root-cause-attribution-step-3)
5. [Interactive Streamlit Diagnostic Dashboard (`app.py`) (Step 4)](#5-interactive-streamlit-diagnostic-dashboard-apppy-step-4)
6. [Deployment Verification & Latency Benchmarks (Step 5)](#6-deployment-verification--latency-benchmarks-step-5)
7. [Directory Structure & Single Outputs Standard](#7-directory-structure--single-outputs-standard)
8. [Quick Start & Command Reference](#8-quick-start--command-reference)

---

### 1. Executive Summary & Key Benchmarks

This system delivers an end-to-end, high-accuracy predictive maintenance and explainable root-cause diagnosis solution for smart manufacturing environments.

| Metric | Target | 1D-CNN | BiLSTM | CNN+BiLSTM Ensemble | Bayesian Network | Final Hybrid System |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | > 90.0% | **92.23%** | **92.06%** | **92.37%** | **92.30%** | **92.30%** |
| **Weighted Precision** | > 90.0% | **93.28%** | **93.12%** | **93.39%** | **93.35%** | **93.35%** |
| **Weighted Recall** | > 90.0% | **92.23%** | **92.06%** | **92.37%** | **92.30%** | **92.30%** |
| **Weighted F1 Score** | > 90.0% | **92.74%** | **92.57%** | **92.86%** | **92.81%** | **92.81%** |
| **Inference Latency** | < 500 ms | ~1.5 ms | ~2.5 ms | ~3.8 ms | ~0.006 ms (VE) | ~278 ms (Batch) |

---

### 2. Data Leakage & Model Validation Audit (Step 1)

Before deployment, a rigorous validation and data leakage audit was executed (`validate_leakage.py`) across three critical dimensions:

#### 2.1 Feature Mutual Information Analysis
We audited operational metadata (`machine_status`, `anomaly_flag`, `downtime_risk`, `maintenance_required`) versus continuous physical sensors (`temperature`, `vibration`, `humidity`, `pressure`, `energy_consumption`).

* **Operational Indicators:** `machine_status` (Mutual Information: 0.252) and `maintenance_required` (Mutual Information: 0.149) exhibit high correlation with the *existence* of an equipment anomaly.
* **Fault-Mode Discriminability:** Operational flags indicate *that* a machine has failed, but they **cannot differentiate** between specific fault types (e.g. `Overheating` vs `Vibration Issue` vs `Pressure Drop` vs `Electrical Fault`). The physical sensor streams are strictly necessary for root-cause classification.

#### 2.2 Unseen Machine Generalization (Machines 41–50)
The models were validated on a partition of completely unseen machines that were never encountered during training:
* **Pure Physical Sensors Only:** **91.84% Test Accuracy** (Confirms that physical sensors alone generalize cleanly without leakage).
* **Multi-Modal Feature Fusion:** **92.42% Test Accuracy** (Operational metadata provides high-confidence confirmation).

#### 2.3 Chronological & Temporal Integrity
* **Temporal Sequence Windows:** Evaluated on late-chronology telemetry segments to ensure future time steps never leak into past predictions.
* **Temporal Validation Result:** **92.46% Test Accuracy**.

---

### 3. Production Real-Time Prediction Engine (`predictor.py`) (Step 2)

The production inference engine (`predictor.py`) encapsulates the trained artifacts into a single high-performance class: `FaultPredictor`.

#### Key Capabilities:
1. **Preprocessing Pipeline:** Applies stored StandardScaler and LabelEncoder identical to training order.
2. **Flexible Input Acceptance:** Accepts Python dictionaries or Pandas DataFrames. Missing operational flags are automatically imputed from physical sensor thresholds.
3. **Multi-Model Consensus:**
   - 1D-CNN localized temporal spike prediction
   - BiLSTM long-range progression prediction
   - Soft-voting probabilistic integration
   - Discrete Bayesian Network causal verification
4. **Automated Severity Scoring:**
   - `HEALTHY` (Green): Nominal continuous operation.
   - `LOW` (Yellow): Borderline signals; monitor trend.
   - `MEDIUM` (Orange): Emerging mechanical/hydraulic issue; schedule maintenance.
   - `HIGH` (Orange): Persistent fault confirmed; immediate inspection.
   - `CRITICAL` (Red): Thermal runaway or power surge; urgent safety risk.

#### Python API Usage:
```python
from predictor import FaultPredictor

# Initialize predictor (loads neural networks, Bayesian DAG, and scalers once)
predictor = FaultPredictor()

# Input raw sensor readings
telemetry = {
    "temperature": 98.5,
    "vibration": 52.0,
    "humidity": 58.0,
    "pressure": 3.1,
    "energy_consumption": 4.1,
    "machine_status": 2,
    "anomaly_flag": 1,
    "downtime_risk": 0.92,
    "maintenance_required": 1
}

result = predictor.predict(telemetry, explain=True)

print("Condition :", result["machine_condition"])
print("Diagnosis :", result["predicted_fault"], f"({result['confidence_pct']}%)")
print("Severity  :", result["severity_level"])
for reason in result["diagnostic_narrative"]:
    print(" -", reason)
```

---

### 4. Explainable AI (XAI) Dynamic SHAP Root-Cause Attribution (Step 3)

The platform answers the core operator question: *"Why did this fault occur and which sensor triggered it?"*

1. **GradientExplainer Attribution:** Calculates the exact gradient attribution vector $\phi_i$ across all sensor channels and temporal time steps.
2. **Dynamic Root-Cause Breakdown:**
   - **Primary Trigger:** The specific sensor contributing the highest positive attribution towards the diagnosed failure mode.
   - **Secondary Contributors:** Supporting sensor deviations reinforcing the anomalous state.
   - **Mitigating Factors:** Sensor channels operating in safe baseline ranges that opposed the fault diagnosis.
3. **Automated Diagnostic Narrative:** Generates human-readable explanations ready for maintenance dispatch logs.

---

### 5. Interactive Streamlit Diagnostic Dashboard (`app.py`) (Step 4)

The application provides a dedicated, real-time diagnostic simulation dashboard running at `http://localhost:8501`.

#### Key Features:
* **Simulation Scenarios & Case Presets:** Quickly load baseline nominal or specific fault presets (Healthy, Overheating, Mechanical Vibration, Hydraulic Pressure Drop, Electrical Surge) or custom manual adjustments.
* **On-Demand "Fetch Fault Diagnosis" Button:** Smooth slider adjustments without lag; inferences and dynamic SHAP gradient calculations run only when requested by clicking the button.
* **Physical Telemetry Auto-Synchronization:** Operational states (`machine_status`, `anomaly_flag`, `downtime_risk`, `maintenance_required`) are automatically inferred from continuous physical sensor readings, ensuring high sensor readings accurately trigger the appropriate fault mode.
* **Individual Model Breakdown:** Side-by-side decision cards for **1D-CNN**, **BiLSTM**, and **Bayesian Network (BN)** with confidence percentages.
* **SHAP Root-Cause Explainability:** Automated reasoning narrative log and a dynamic horizontal bar chart visually mapping the exact attribution of each sensor.

---

### 6. Deployment Verification & Latency Benchmarks (Step 5)

A comprehensive test suite was executed via `test_deployment.py`:

```
================================================================================
 BENCHMARKING LATENCY (100 CONSECUTIVE INFERENCES) 
================================================================================
Average Inference Latency (without SHAP) : 277.95 ms
Median Inference Latency                 : 297.31 ms
P95 Inference Latency                    : 534.99 ms
Throughput                               : 3.6 predictions/sec

Real-time deployment test suite passed successfully with ZERO errors!
```

---

### 7. Directory Structure & Single Outputs Standard

The project adheres strictly to the single `outputs/` directory and single `notebook/new_prob.ipynb` standard:

```
c:\Users\devin\Desktop\prob_project\
│
├── dataset\
│   └── smart_manufacturing_data.csv       # Industrial IoT telemetry (100,000 records)
│
├── notebook\
│   └── new_prob.ipynb                     # Clean, single Jupyter notebook (>92% Accuracy)
│
├── outputs\                               # STRICTLY SINGLE outputs directory
│   ├── best_cnn_model.keras               # Trained 1D-CNN model (>92% Accuracy)
│   ├── best_lstm_model.keras              # Trained BiLSTM model (>92% Accuracy)
│   ├── bayesian_network.pkl               # Discrete Bayesian Network DAG
│   ├── scaler.pkl                         # Fitted StandardScaler (9 features)
│   ├── label_encoder.pkl                  # Fitted LabelEncoder (5 classes)
│   ├── bin_edges.pkl                      # Discretization quantile edges
│   ├── model_comparison_table.csv         # Benchmark metrics table
│   ├── validation_leakage_audit.csv       # Unseen machine & temporal audit
│   ├── feature_mutual_information.csv     # Mutual information scores
│   ├── shap_global_sensor_importance.csv  # Global SHAP ranking
│   ├── shap_per_class_importance.csv      # Per-class SHAP matrix
│   ├── shap_why_fault_occurred_reports.txt# Natural-language XAI reports
│   └── *.png                              # Confusion matrices, ROC/PR curves, heatmaps
│
├── app.py                                 # Interactive Streamlit dashboard
├── new_prob.py                            # Complete automated training & evaluation script
├── predictor.py                           # Production real-time inference engine
├── validate_leakage.py                    # Data leakage audit & validation script
├── test_deployment.py                     # Real-time deployment test suite & benchmark
├── DEPLOYMENT.md                          # Production deployment & architecture guide
└── README.md                              # Project overview
```

---

### 8. Quick Start & Command Reference

#### 1. Run Complete Training & SHAP Evaluation:
```powershell
C:\Users\devin\anaconda3\envs\faultdiag_new\python.exe new_prob.py
```

#### 2. Run Data Leakage Validation Audit:
```powershell
C:\Users\devin\anaconda3\envs\faultdiag_new\python.exe validate_leakage.py
```

#### 3. Run Real-Time Deployment Test & Latency Benchmark:
```powershell
C:\Users\devin\anaconda3\envs\faultdiag_new\python.exe test_deployment.py
```

#### 4. Launch Streamlit Web Dashboard:
```powershell
C:\Users\devin\anaconda3\envs\faultdiag_new\python.exe -m streamlit run app.py
```
*Access locally at:* `http://localhost:8501`

#### 5. Open Jupyter Notebook:
Launch Jupyter Notebook and open:
```
notebook/new_prob.ipynb
```
Each cell can be executed sequentially to train, evaluate, and visualize the entire pipeline interactively.
