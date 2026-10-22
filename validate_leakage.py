"""
========================================================================================
DATA LEAKAGE AUDIT & GENERALIZATION VALIDATION
========================================================================================
Validates machine fault diagnosis performance under:
  1. Pure Physical Sensor Telemetry (Zero Operational Leakage)
  2. Multi-Modal Telemetry (Sensors + Operational Flags)
  3. Generalization to Unseen Machines (Group split: Train M01-M40, Test M41-M50)
  4. Generalization across Chronological Time (Train Early, Test Future)
========================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "dataset", "smart_manufacturing_data.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", dayfirst=True)
df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

PHYSICAL_SENSORS = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]
OP_FEATURES = ["machine_status", "anomaly_flag", "downtime_risk", "maintenance_required"]
ALL_FEATURES = PHYSICAL_SENSORS + OP_FEATURES
TARGET = "failure_type"

le = LabelEncoder()
df["target_encoded"] = le.fit_transform(df[TARGET])
CLASS_NAMES = list(le.classes_)

print("=" * 80)
print(" 1. MUTUAL INFORMATION & TARGET LEAKAGE AUDIT")
print("=" * 80)

# Compute mutual information of features with the fault target
mi_scores = mutual_info_classif(df[ALL_FEATURES], df["target_encoded"], random_state=42)
mi_df = pd.DataFrame({
    "Feature": ALL_FEATURES,
    "Mutual_Information": mi_scores,
    "Type": ["Physical Sensor" if f in PHYSICAL_SENSORS else "Operational Flag" for f in ALL_FEATURES]
}).sort_values("Mutual_Information", ascending=False).reset_index(drop=True)

print(mi_df.to_string(index=False))

# Specific conditional leakage analysis
print("\nConditional Probability of Fault given Operational Flags:")
fault_mask = df[TARGET] != "Normal"
print(f"  P(machine_status == 2 | Fault)       = {(df.loc[fault_mask, 'machine_status'] == 2).mean() * 100:.2f}%")
print(f"  P(maintenance_required == 1 | Fault) = {(df.loc[fault_mask, 'maintenance_required'] == 1).mean() * 100:.2f}%")
print(f"  P(Fault | machine_status == 2)       = {(df.loc[df['machine_status'] == 2, TARGET] != 'Normal').mean() * 100:.2f}%")
print(f"  P(Fault | maintenance_required == 1) = {(df.loc[df['maintenance_required'] == 1, TARGET] != 'Normal').mean() * 100:.2f}%")

print("\n" + "=" * 80)
print(" 2. GENERALIZATION EXPERIMENTS (PHYSICAL vs MULTI-MODAL & UNSEEN MACHINES)")
print("=" * 80)

results = []

def eval_split(name, X_tr, X_te, y_tr, y_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    
    return {
        "Experiment": name,
        "Train_Samples": len(X_tr),
        "Test_Samples": len(X_te),
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4)
    }

# 1. Random Split - Pure Physical Sensors
X_phys = df[PHYSICAL_SENSORS]
y = df["target_encoded"]
X_tr, X_te, y_tr, y_te = train_test_split(X_phys, y, test_size=0.2, random_state=42, stratify=y)
results.append(eval_split("Pure Physical Sensors (Random Split)", X_tr, X_te, y_tr, y_te))

# 2. Random Split - Multi-Modal (Sensors + Op Features)
X_all = df[ALL_FEATURES]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)
results.append(eval_split("Multi-Modal Sensors+Op (Random Split)", X_tr, X_te, y_tr, y_te))

# 3. Unseen Machines Split (Train Machines 1-40, Test Machines 41-50)
unique_machines = sorted(df["machine_id"].unique())
train_m = unique_machines[:40]
test_m = unique_machines[40:]

train_mask = df["machine_id"].isin(train_m)
test_mask = df["machine_id"].isin(test_m)

# 3a. Unseen Machines - Pure Physical Sensors
X_tr_m = df.loc[train_mask, PHYSICAL_SENSORS]
y_tr_m = df.loc[train_mask, "target_encoded"]
X_te_m = df.loc[test_mask, PHYSICAL_SENSORS]
y_te_m = df.loc[test_mask, "target_encoded"]
results.append(eval_split("Pure Physical Sensors (Unseen Machines 41-50)", X_tr_m, X_te_m, y_tr_m, y_te_m))

# 3b. Unseen Machines - Multi-Modal
X_tr_m_all = df.loc[train_mask, ALL_FEATURES]
X_te_m_all = df.loc[test_mask, ALL_FEATURES]
results.append(eval_split("Multi-Modal Sensors+Op (Unseen Machines 41-50)", X_tr_m_all, X_te_m_all, y_tr_m, y_te_m))

# 4. Chronological Time Split (First 80% timestamps Train, Last 20% timestamps Test)
cutoff_time = df["timestamp"].quantile(0.80)
time_train_mask = df["timestamp"] < cutoff_time
time_test_mask = df["timestamp"] >= cutoff_time

# 4a. Chronological Split - Pure Physical Sensors
X_tr_t = df.loc[time_train_mask, PHYSICAL_SENSORS]
y_tr_t = df.loc[time_train_mask, "target_encoded"]
X_te_t = df.loc[time_test_mask, PHYSICAL_SENSORS]
y_te_t = df.loc[time_test_mask, "target_encoded"]
results.append(eval_split("Pure Physical Sensors (Chronological Time Split)", X_tr_t, X_te_t, y_tr_t, y_te_t))

# 4b. Chronological Split - Multi-Modal
X_tr_t_all = df.loc[time_train_mask, ALL_FEATURES]
X_te_t_all = df.loc[time_test_mask, ALL_FEATURES]
results.append(eval_split("Multi-Modal Sensors+Op (Chronological Time Split)", X_tr_t_all, X_te_t_all, y_tr_t, y_te_t))

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "validation_leakage_audit.csv"), index=False)
mi_df.to_csv(os.path.join(OUT_DIR, "feature_mutual_information.csv"), index=False)

print("\nValidation & Data Leakage Benchmark Table:")
print(res_df.to_string(index=False))

print("\nAudit successfully saved to:")
print(f"  - {os.path.join(OUT_DIR, 'validation_leakage_audit.csv')}")
print(f"  - {os.path.join(OUT_DIR, 'feature_mutual_information.csv')}")
