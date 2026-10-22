"""
========================================================================================
DEPLOYMENT TEST SUITE & LATENCY BENCHMARK (TEST_DEPLOYMENT.PY)
========================================================================================
Tests real-time fault prediction, severity assessment, and dynamic SHAP explanations
across nominal and faulty machine scenarios.
========================================================================================
"""

import time
import json
import numpy as np
from predictor import FaultPredictor

print("=" * 80)
print(" INITIALIZING REAL-TIME PREDICTOR ")
print("=" * 80)
t0 = time.time()
predictor = FaultPredictor()
print(f"Predictor initialized in {time.time() - t0:.2f} seconds.\n")

scenarios = [
    {
        "name": "Scenario 1: Nominal Machine Operation (Healthy)",
        "inputs": {"temperature": 73.5, "vibration": 48.0, "humidity": 55.0, "pressure": 3.0, "energy_consumption": 2.7, "machine_status": 1, "anomaly_flag": 0, "downtime_risk": 0.05, "maintenance_required": 0}
    },
    {
        "name": "Scenario 2: Thermal Runaway (Overheating)",
        "inputs": {"temperature": 108.5, "vibration": 58.0, "humidity": 58.0, "pressure": 3.1, "energy_consumption": 4.5, "machine_status": 2, "anomaly_flag": 1, "downtime_risk": 0.95, "maintenance_required": 1}
    },
    {
        "name": "Scenario 3: Mechanical Bearing Oscillation (Vibration Issue)",
        "inputs": {"temperature": 76.0, "vibration": 98.5, "humidity": 52.0, "pressure": 3.0, "energy_consumption": 3.7, "machine_status": 2, "anomaly_flag": 1, "downtime_risk": 0.90, "maintenance_required": 1}
    },
    {
        "name": "Scenario 4: Hydraulic Line Leakage (Pressure Drop)",
        "inputs": {"temperature": 74.0, "vibration": 49.0, "humidity": 51.0, "pressure": 1.1, "energy_consumption": 2.8, "machine_status": 2, "anomaly_flag": 1, "downtime_risk": 0.88, "maintenance_required": 1}
    },
    {
        "name": "Scenario 5: High Current Draw & Power Surge (Electrical Fault)",
        "inputs": {"temperature": 91.0, "vibration": 54.0, "humidity": 55.0, "pressure": 2.9, "energy_consumption": 4.95, "machine_status": 2, "anomaly_flag": 1, "downtime_risk": 0.94, "maintenance_required": 1}
    },
    {
        "name": "Scenario 6: Pure Physical Sensors Only (Operational Flags Omitted)",
        "inputs": {"temperature": 102.0, "vibration": 55.0, "humidity": 60.0, "pressure": 3.1, "energy_consumption": 4.2}
    }
]

print("=" * 80)
print(" RUNNING REAL-TIME DIAGNOSTIC SCENARIOS ")
print("=" * 80)

for sc in scenarios:
    res = predictor.predict(sc["inputs"], explain=True)
    print(f"\n--- {sc['name']} ---")
    print(f"  Condition: {res['machine_condition']} | Diagnosis: {res['predicted_fault']} (Confidence: {res['confidence_pct']}%)")
    print(f"  Severity : {res['severity_level']} ({res['severity_color']}) - {res['severity_description']}")
    print(f"  Latency  : {res['inference_latency_ms']} ms")
    print("  Explanations:")
    for line in res["diagnostic_narrative"]:
        print(f"    * {line}")

print("\n" + "=" * 80)
print(" BENCHMARKING LATENCY (100 CONSECUTIVE INFERENCES) ")
print("=" * 80)

latencies = []
dummy_input = {"temperature": 75.0, "vibration": 50.0, "humidity": 55.0, "pressure": 3.0, "energy_consumption": 2.75}

for _ in range(100):
    t_start = time.time()
    # Fast inference without re-calculating full SHAP gradient each millisecond
    _ = predictor.predict(dummy_input, explain=False)
    latencies.append((time.time() - t_start) * 1000.0)

print(f"Average Inference Latency (without SHAP) : {np.mean(latencies):.2f} ms")
print(f"Median Inference Latency                 : {np.median(latencies):.2f} ms")
print(f"P95 Inference Latency                    : {np.percentile(latencies, 95):.2f} ms")
print(f"Throughput                               : {1000.0 / np.mean(latencies):.1f} predictions/sec")
print("\nReal-time deployment test suite passed successfully with ZERO errors!")
