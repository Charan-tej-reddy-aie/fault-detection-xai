# Explainable Machine Fault Detection & Root-Cause Diagnosis (new_prob)

An industrial-grade, end-to-end Machine Fault Detection and Explainable AI (XAI) pipeline using **Hybrid Deep Learning (1D-CNN + BiLSTM)**, **Probabilistic Graphical Models (Discrete Bayesian Network)**, and **SHAP (SHapley Additive exPlanations)**.

Trained and evaluated on the real dataset: `dataset/smart_manufacturing_data.csv` (100,000 multi-sensor IoT readings across 50 machines).

---

## 🌟 Interactive Web Dashboard (`app.py`)

A full interactive Streamlit dashboard is included to monitor machine health, run real-time "What-If" simulations, query the Bayesian Network, and view SHAP root-cause explanations.

### How to Launch the Dashboard:
```powershell
conda activate faultdiag_new
cd C:\Users\devin\Desktop\prob_project
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### Dashboard Features:
1. **🏠 Overview & KPI Summary:** Real-time metrics on monitored machines, accuracy, and fault mode breakdown.
2. **⚡ Live Fault Diagnosis Simulator:** Interactive sliders for physical sensors (`temperature`, `vibration`, `humidity`, `pressure`, `energy_consumption`) and operational states with live multi-model prediction and automated root-cause reports.
3. **📊 Industrial Telemetry Explorer:** Filter and inspect the 100,000 telemetry records by machine ID or failure mode with distribution plots.
4. **🏆 Model Evaluation (>92% Accuracy):** Interactive confusion matrix viewer, ROC/PR curves, per-class F1 breakdown, and inference speed benchmarks.
5. **🔍 SHAP Root-Cause Explainability:** Global feature rankings, per-fault-class attribution heatmaps, temporal progression heatmaps, and downloadable operator reports.
6. **🕸️ Bayesian Network Causal Graph:** Interactive causal DAG query interface calculating exact posterior distributions via Variable Elimination.

---

## 📊 Evaluation & Model Performance (>92% Accuracy)

Evaluated on the independent test split (2,974 sequences):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **1. 1D-CNN** | **0.9223 (92.23%)** | 0.9310 | 0.9223 | **0.9263** |
| **2. BiLSTM** | **0.9206 (92.06%)** | 0.9340 | 0.9206 | **0.9259** |
| **3. CNN+BiLSTM Ensemble** | **0.9237 (92.37%)** | 0.9336 | 0.9237 | **0.9282** |
| **4. Bayesian Network** | **0.9230 (92.30%)** | 0.9339 | 0.9230 | **0.9282** |
| **5. CNN+BiLSTM+BN (Final Hybrid)** | **0.9230 (92.30%)** | **0.9339** | **0.9230** | **0.9282** |

---

## ⚙️ How to Run via Notebook or CLI

### Run in Jupyter Notebook:
```powershell
conda activate faultdiag_new
cd C:\Users\devin\Desktop\prob_project
jupyter notebook notebook/new_prob.ipynb
```
Select the **`Python (faultdiag_new)`** kernel and run all cells.

### Run via Command Line:
```powershell
conda activate faultdiag_new
cd C:\Users\devin\Desktop\prob_project
python new_prob.py
```

### Run Model Validation & Data Leakage Audit:
```powershell
python validate_leakage.py
```

### Run Real-Time Deployment Benchmark:
```powershell
python test_deployment.py
```

For complete technical details, mathematical formulation, and architecture diagrams, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

