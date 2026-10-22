"""
========================================================================================
INDUSTRIAL MACHINE FAULT DETECTION & EXPLAINABLE AI (XAI) SIMULATOR
========================================================================================
Production Diagnostic Application (Clean Industrial White Theme)
Powered by:
  - 1D Convolutional Neural Network (1D-CNN)
  - Bidirectional Long Short-Term Memory Network (BiLSTM)
  - Discrete Bayesian Network (pgmpy) Causal Reasoning
  - Dynamic Gradient SHAP Root-Cause Attribution
  - Prescriptive Maintenance & Root-Cause Avoidance Protocol
========================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from predictor import FaultPredictor, SENSOR_FEATURES, ALL_FEATURES

# --------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & WHITE THEME STYLING
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Machine Fault Diagnosis & XAI Simulator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enforce clean, professional, pristine white industrial theme
st.markdown("""
<style>
    /* Global White Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    
    /* Big Bold Header */
    .big-main-header {
        font-size: 2.35rem;
        font-weight: 850;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }
    .main-sub-header {
        font-size: 1.05rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    
    /* Clean Control Box */
    .control-panel {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.3rem;
    }
    
    /* Clean Cards */
    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .kpi-card-healthy {
        background-color: #F0FDF4;
        border: 1.5px solid #86EFAC;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
    }
    .kpi-card-fault {
        background-color: #FEF2F2;
        border: 1.5px solid #FCA5A5;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .kpi-sub {
        font-size: 0.85rem;
        color: #64748B;
    }
    
    /* Model Breakdown Cards */
    .model-card-white {
        background-color: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .model-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .model-prediction {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0.35rem 0;
    }
    .model-conf {
        font-size: 0.88rem;
        color: #64748B;
    }
    
    /* Alert Badge (The single allowed red alert symbol) */
    .red-alert-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #DC2626;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.25);
    }
    
    /* Reasoning Log Box */
    .reasoning-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1E293B;
    }
    
    /* Action Protocol Cards */
    .protocol-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# 2. LOAD PREDICTOR CACHE & ANTIQUE MACHINE IMAGE
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Deep Learning Models & Bayesian Network...")
def load_fault_predictor():
    return FaultPredictor()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(PROJECT_DIR, "outputs", "antique_machine.png")

try:
    predictor = load_fault_predictor()
    CLASS_NAMES = predictor.class_names
    predictor_ready = True
except Exception as e:
    st.error(f"Error initializing predictor models: {e}")
    predictor_ready = False
    CLASS_NAMES = ["Electrical Fault", "Normal", "Overheating", "Pressure Drop", "Vibration Issue"]

# --------------------------------------------------------------------------------------
# 3. PRESETS DEFINITION (NO SYMBOLS) & SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------------------
PRESETS = {
    "Case 1: Healthy Nominal Operation": {
        "temp": 72.5, "vib": 48.0, "hum": 54.0, "press": 3.0, "eng": 2.5
    },
    "Case 2: Severe Overheating (High Temperature)": {
        "temp": 112.0, "vib": 50.0, "hum": 58.0, "press": 3.0, "eng": 3.0
    },
    "Case 3: Mechanical Bearing Oscillation (High Vibration)": {
        "temp": 74.0, "vib": 105.0, "hum": 52.0, "press": 3.0, "eng": 3.2
    },
    "Case 4: Hydraulic Line Leakage (Low Pressure)": {
        "temp": 74.0, "vib": 50.0, "hum": 51.0, "press": 1.1, "eng": 2.8
    },
    "Case 5: High Current Draw and Power Surge (High Energy)": {
        "temp": 82.0, "vib": 54.0, "hum": 55.0, "press": 2.9, "eng": 5.0
    },
    "Case 6: Multi-Factor Critical Failure (All Features High)": {
        "temp": 122.0, "vib": 120.0, "hum": 85.0, "press": 5.5, "eng": 5.5
    }
}

if "sim_temp" not in st.session_state:
    st.session_state.sim_temp = 72.5
if "sim_vib" not in st.session_state:
    st.session_state.sim_vib = 48.0
if "sim_hum" not in st.session_state:
    st.session_state.sim_hum = 54.0
if "sim_press" not in st.session_state:
    st.session_state.sim_press = 3.0
if "sim_eng" not in st.session_state:
    st.session_state.sim_eng = 2.5

# --------------------------------------------------------------------------------------
# 4. HEADER ROW: BIG HEADING & HORIZONTAL ANTIQUE MACHINE HERO IMAGE
# --------------------------------------------------------------------------------------
st.markdown('<div class="big-main-header">INDUSTRIAL MACHINE FAULT DIAGNOSIS & EXPLAINABLE AI SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub-header">Multi-Modal Deep Learning (1D-CNN + BiLSTM), Bayesian Causal Reasoning & SHAP Root-Cause Analysis</div>', unsafe_allow_html=True)

if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True)
elif os.path.exists(os.path.join(PROJECT_DIR, "outputs", "antique_machine.jpg")):
    st.image(os.path.join(PROJECT_DIR, "outputs", "antique_machine.jpg"), use_container_width=True)

# --------------------------------------------------------------------------------------
# 5. SIMULATION CONTROLS (CLEAN WHITE THEME, NO SYMBOLS)
# --------------------------------------------------------------------------------------
with st.container():
    st.markdown("### Simulate Machine Sensor Telemetry")
    
    # Preset Selector
    st.markdown("**Step 1: Choose Scenario Preset (Optional)**")
    preset_choice = st.selectbox(
        "Select Scenario Preset to Auto-Fill Sliders:",
        ["-- Manual Custom Telemetry --"] + list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed"
    )

    if preset_choice in PRESETS:
        vals = PRESETS[preset_choice]
        st.session_state.sim_temp = vals["temp"]
        st.session_state.sim_vib = vals["vib"]
        st.session_state.sim_hum = vals["hum"]
        st.session_state.sim_press = vals["press"]
        st.session_state.sim_eng = vals["eng"]

    # Physical Sensor Sliders
    st.markdown("<br><b>Step 2: Adjust Continuous Physical Sensor Telemetry</b>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Thermal and Mechanical**")
        temp_val = st.slider("Temperature (°C)", min_value=30.0, max_value=125.0, value=float(st.session_state.sim_temp), step=0.5, key="slider_temp")
        vib_val = st.slider("Vibration (mm/s)", min_value=0.0, max_value=120.0, value=float(st.session_state.sim_vib), step=0.5, key="slider_vib")

    with c2:
        st.markdown("**Hydraulic and Electrical**")
        press_val = st.slider("Pressure (Bar)", min_value=0.5, max_value=5.5, value=float(st.session_state.sim_press), step=0.1, key="slider_press")
        eng_val = st.slider("Energy Consumption (kW/h)", min_value=0.5, max_value=5.5, value=float(st.session_state.sim_eng), step=0.1, key="slider_eng")

    with c3:
        st.markdown("**Atmospheric and Explainability**")
        hum_val = st.slider("Humidity (%)", min_value=25.0, max_value=85.0, value=float(st.session_state.sim_hum), step=0.5, key="slider_hum")
        compute_shap = st.checkbox("Compute Dynamic SHAP Attribution", value=True)

    # Big Action Button (Clean Text, No Emojis)
    st.markdown("<br>", unsafe_allow_html=True)
    fetch_btn = st.button("RUN FAULT DIAGNOSIS", type="primary", use_container_width=True)

# --------------------------------------------------------------------------------------
# 6. DIAGNOSTIC EXECUTION
# --------------------------------------------------------------------------------------
# Run inference when button clicked OR on first page load
if fetch_btn or ("last_diag" not in st.session_state):
    sensor_input = {
        "temperature": float(temp_val),
        "vibration": float(vib_val),
        "humidity": float(hum_val),
        "pressure": float(press_val),
        "energy_consumption": float(eng_val)
    }

    with st.spinner("Executing 1D-CNN, BiLSTM, Bayesian Network and SHAP inference..."):
        diag = predictor.predict(sensor_input, explain=compute_shap)
        st.session_state.last_diag = diag
        st.session_state.last_input = sensor_input.copy()

# --------------------------------------------------------------------------------------
# 7. DIAGNOSTIC RESULT PRESENTATION (WHITE THEME, EXPLICIT ALERT SYMBOL ONLY)
# --------------------------------------------------------------------------------------
if "last_diag" in st.session_state:
    diag = st.session_state.last_diag
    inp = st.session_state.get("last_input", {})
    is_healthy = (diag["machine_condition"] == "HEALTHY")

    st.markdown("---")
    st.markdown(f"#### Diagnostic Evaluation: `Temperature: {inp.get('temperature', 0):.1f}°C` | `Vibration: {inp.get('vibration', 0):.1f} mm/s` | `Pressure: {inp.get('pressure', 0):.1f} Bar` | `Energy: {inp.get('energy_consumption', 0):.1f} kW/h`")

    # Row 1: Primary Diagnostic KPI Cards
    b1, b2, b3, b4 = st.columns([1.3, 1.3, 1.1, 0.8])
    with b1:
        if is_healthy:
            st.markdown(f"""
            <div class="kpi-card-healthy">
                <div class="kpi-label">EQUIPMENT STATUS</div>
                <div class="kpi-value" style="color: #15803D;">HEALTHY</div>
                <div class="kpi-sub">Confidence: <b>{diag['confidence_pct']:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ONLY RED ALERT SYMBOL APPLIED HERE AS REQUESTED
            st.markdown(f"""
            <div class="kpi-card-fault">
                <div class="kpi-label" style="color: #991B1B;">EQUIPMENT STATUS</div>
                <div class="kpi-value" style="color: #B91C1C;">
                    <span class="red-alert-dot"></span>FAULT DETECTED
                </div>
                <div class="kpi-sub" style="color: #7F1D1D;">Confidence: <b>{diag['confidence_pct']:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">FINAL HYBRID DIAGNOSIS</div>
            <div class="kpi-value" style="color: #0F172A;">{diag['predicted_fault']}</div>
            <div class="kpi-sub">Physics-Guided Consensus (CNN + BiLSTM + BN)</div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        sev = diag["severity_level"]
        sev_color_map = {
            "HEALTHY": "#15803D",
            "LOW": "#B45309",
            "MEDIUM": "#C2410C",
            "HIGH": "#B91C1C",
            "CRITICAL": "#991B1B"
        }
        sev_color = sev_color_map.get(sev, "#0F172A")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">SEVERITY LEVEL</div>
            <div class="kpi-value" style="color: {sev_color};">{diag['severity_level']}</div>
            <div class="kpi-sub">{diag['severity_description']}</div>
        </div>
        """, unsafe_allow_html=True)

    with b4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">LATENCY</div>
            <div class="kpi-value" style="color: #0F172A;">{diag['inference_latency_ms']:.1f}<span style="font-size: 1rem; color: #64748B;">ms</span></div>
            <div class="kpi-sub" style="color: #15803D; font-weight: 600;">Real-Time Ready</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Model Decision Breakdown (1D-CNN, BiLSTM, Bayesian Network)
    st.subheader("Multi-Model Decision Breakdown (CNN, LSTM, Bayesian Network)")
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(f"""
        <div class="model-card-white">
            <div class="model-title">1. 1D-CNN (Temporal Spikes)</div>
            <div class="model-prediction">{diag['cnn_diagnosis']}</div>
            <div class="model-conf">Confidence: <b>{diag['cnn_confidence_pct']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="model-card-white">
            <div class="model-title">2. BiLSTM (Temporal Sequence)</div>
            <div class="model-prediction">{diag['lstm_diagnosis']}</div>
            <div class="model-conf">Confidence: <b>{diag['lstm_confidence_pct']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="model-card-white">
            <div class="model-title">3. Bayesian Network (Causal DAG)</div>
            <div class="model-prediction">{diag['bayesian_diagnosis']}</div>
            <div class="model-conf">Posterior Certainty: <b>{diag['bayesian_confidence_pct']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Class Probability Distributions (Grouped Horizontal Bar Chart on Pure White)
    st.subheader("Fault Mode Probability Distribution Across All Classes")
    prob_rows = []
    for cls in CLASS_NAMES:
        prob_rows.append({
            "Fault Class": cls,
            "Ensemble Probability (%)": diag["probabilities"].get(cls, 0.0),
            "Bayesian Posterior (%)": diag["bayesian_posterior"].get(cls, 0.0)
        })
    prob_df = pd.DataFrame(prob_rows)

    c_table, c_chart = st.columns([1.1, 1.4])
    with c_table:
        row_list = []
        for cls in CLASS_NAMES:
            ens_prob = diag["probabilities"].get(cls, 0.0)
            bn_prob = diag["bayesian_posterior"].get(cls, 0.0)
            is_selected = (cls == diag["predicted_fault"])

            if is_selected:
                if is_healthy:
                    row_bg = "#F0FDF4"
                    border_style = "border-left: 4px solid #16A34A;"
                    class_cell = f"<b style='color: #166534;'>{cls}</b> <span style='background-color: #16A34A; color: #FFFFFF; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; margin-left: 6px;'>HEALTHY BASELINE</span>"
                    val1_cell = f"<b style='color: #15803D; font-size: 1.02rem;'>{ens_prob:.2f}%</b>"
                    val2_cell = f"<b style='color: #15803D; font-size: 1.02rem;'>{bn_prob:.2f}%</b>"
                else:
                    row_bg = "#FEF2F2"
                    border_style = "border-left: 4px solid #DC2626;"
                    class_cell = f"<b style='color: #991B1B;'>{cls}</b> <span style='background-color: #DC2626; color: #FFFFFF; font-size: 0.72rem; font-weight: 700; padding: 2px 7px; border-radius: 4px; margin-left: 6px;'>DIAGNOSED FAULT</span>"
                    val1_cell = f"<b style='color: #B91C1C; font-size: 1.02rem;'>{ens_prob:.2f}%</b>"
                    val2_cell = f"<b style='color: #B91C1C; font-size: 1.02rem;'>{bn_prob:.2f}%</b>"
            else:
                row_bg = "#FFFFFF"
                border_style = "border-left: 4px solid transparent;"
                class_cell = f"<span style='color: #334155;'>{cls}</span>"
                val1_cell = f"<span style='color: #475569;'>{ens_prob:.2f}%</span>"
                val2_cell = f"<span style='color: #475569;'>{bn_prob:.2f}%</span>"

            row_list.append(
                f'<tr style="background-color: {row_bg}; {border_style} border-bottom: 1px solid #E2E8F0;">'
                f'<td style="padding: 9px 12px; text-align: left;">{class_cell}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-family: monospace;">{val1_cell}</td>'
                f'<td style="padding: 9px 12px; text-align: right; font-family: monospace;">{val2_cell}</td>'
                f'</tr>'
            )

        table_body = "".join(row_list)
        table_html = (
            '<div style="background-color: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04); margin-bottom: 0.5rem;">'
            '<table style="width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; font-size: 0.90rem;">'
            '<thead>'
            '<tr style="background-color: #F1F5F9; border-bottom: 2px solid #CBD5E1;">'
            '<th style="padding: 10px 12px; text-align: left; color: #1E293B; font-weight: 700; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Fault Class</th>'
            '<th style="padding: 10px 12px; text-align: right; color: #1E293B; font-weight: 700; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Ensemble Prob (%)</th>'
            '<th style="padding: 10px 12px; text-align: right; color: #1E293B; font-weight: 700; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em;">Bayesian Posterior (%)</th>'
            '</tr>'
            '</thead>'
            f'<tbody>{table_body}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

    with c_chart:
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        y_indices = np.arange(len(CLASS_NAMES))
        bar_height = 0.35
        ens_vals = [diag["probabilities"].get(c, 0.0) for c in CLASS_NAMES]
        bn_vals = [diag["bayesian_posterior"].get(c, 0.0) for c in CLASS_NAMES]

        ax.barh(y_indices - bar_height/2, ens_vals, height=bar_height, label="Ensemble (DL)", color="#2563EB")
        ax.barh(y_indices + bar_height/2, bn_vals, height=bar_height, label="Bayesian Network", color="#059669")
        ax.set_yticks(y_indices)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9, color="#0F172A")
        ax.set_xlabel("Probability (%)", fontsize=9, color="#475569")
        ax.tick_params(colors="#475569")
        ax.set_xlim(0, 105)
        ax.legend(loc="lower right", fontsize=8, facecolor="#FFFFFF", edgecolor="#CBD5E1", labelcolor="#0F172A")
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="#E2E8F0")
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 4: Dynamic SHAP Root-Cause Explainability (Physical Sensors Only)
    if compute_shap and "feature_contributions" in diag and diag["feature_contributions"]:
        st.markdown("---")
        st.subheader("SHAP Root-Cause Explainability (Why Did This Fault Occur?)")

        shap_col1, shap_col2 = st.columns([1.2, 1.1])

        with shap_col1:
            st.markdown("**Automated Diagnostic Reasoning Log**")
            narrative_text = "<br>".join([f"• {line}" for line in diag["diagnostic_narrative"]])
            st.markdown(f"""
            <div class="reasoning-box">
                <span style="color: #2563EB; font-weight: bold;">[PHYSICAL SENSOR ATTRIBUTION LOG]</span><br><br>
                {narrative_text}
            </div>
            """, unsafe_allow_html=True)

            if diag.get("primary_trigger"):
                pt = diag["primary_trigger"]
                if is_healthy:
                    st.markdown(f"""
                    <div style="background-color: #F0FDF4; border: 1.5px solid #86EFAC; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.8rem; color: #166534; font-size: 0.92rem;">
                        <b>Key Nominal Stabilizer Sensor:</b> {pt['feature'].replace('_', ' ').title()} 
                        (Telemetry: <b>{pt['raw_value']}</b>, Stabilizing SHAP Attribution: <b style="color: #059669;">{pt['attribution']:+.4f}</b>)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #FEF2F2; border: 1.5px solid #FCA5A5; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.8rem; color: #991B1B; font-size: 0.92rem;">
                        <b>Primary Fault Driver Sensor:</b> {pt['feature'].replace('_', ' ').title()} 
                        (Telemetry: <b>{pt['raw_value']}</b>, Fault Trigger SHAP Attribution: <b style="color: #DC2626;">{pt['attribution']:+.4f}</b>)
                    </div>
                    """, unsafe_allow_html=True)

        with shap_col2:
            st.markdown("**Physical Sensor Impact on Diagnosis (SHAP Attribution)**")
            contribs = diag["feature_contributions"]
            contribs_sorted = sorted(contribs, key=lambda x: x["attribution"], reverse=True)

            feat_names = [c["feature"].replace("_", " ").title() for c in contribs_sorted]
            attr_values = [c["attribution"] for c in contribs_sorted]

            # Precise Color assignment:
            # In Healthy: all positive attributions stabilize the Normal condition -> Vivid Green (#059669)
            # In Fault: positive attributions push toward the fault -> Vivid Red (#DC2626), negative attributions pull away -> Vivid Green (#059669)
            if is_healthy:
                bar_colors = ["#059669" if v >= 0 else "#DC2626" for v in attr_values]
            else:
                bar_colors = ["#DC2626" if v > 0 else "#059669" for v in attr_values]

            fig_shap, ax_shap = plt.subplots(figsize=(6, 3.0), facecolor="#FFFFFF")
            ax_shap.set_facecolor("#FFFFFF")
            y_pos = np.arange(len(feat_names))
            
            # Render horizontal bars with clear border
            bars = ax_shap.barh(y_pos, attr_values, color=bar_colors, align="center", height=0.52, edgecolor="#CBD5E1", linewidth=0.8)
            
            # Auto scale x limits with padding so text labels are never cut off
            max_val = max([abs(v) for v in attr_values]) if attr_values else 0.05
            max_val = max(max_val, 0.03)
            ax_shap.set_xlim(-max_val * 1.45, max_val * 1.45)
            
            # Label each bar with its exact attribution value in bold
            for idx, (bar, val) in enumerate(zip(bars, attr_values)):
                color = bar_colors[idx]
                offset = max_val * 0.04
                if val >= 0:
                    ax_shap.text(val + offset, idx, f"{val:+.4f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color=color)
                else:
                    ax_shap.text(val - offset, idx, f"{val:+.4f}", va="center", ha="right", fontsize=8.5, fontweight="bold", color=color)

            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(feat_names, fontsize=8.5, fontweight="600", color="#0F172A")
            ax_shap.tick_params(colors="#475569")
            ax_shap.invert_yaxis()
            ax_shap.axvline(0, color="#64748B", linewidth=1.0, linestyle="--")
            ax_shap.set_xlabel(f"SHAP Attribution towards '{diag['predicted_fault']}'", fontsize=8.5, color="#334155", fontweight="600")
            ax_shap.grid(axis="x", linestyle=":", alpha=0.6, color="#E2E8F0")
            for spine in ax_shap.spines.values():
                spine.set_color("#CBD5E1")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

            # Explicit Visual Legend with vivid color swatches
            st.markdown("""
            <div style="display: flex; gap: 20px; font-size: 0.85rem; margin-top: 4px; padding: 7px 14px; background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px;">
                <div><span style="display:inline-block; width:12px; height:12px; background-color:#DC2626; border-radius:2px; vertical-align:middle; margin-right:6px;"></span><b style="color:#991B1B;">Red Bar:</b> Pushes toward fault diagnosis (Hazardous factor)</div>
                <div><span style="display:inline-block; width:12px; height:12px; background-color:#059669; border-radius:2px; vertical-align:middle; margin-right:6px;"></span><b style="color:#065F46;">Green Bar:</b> Safe stabilizing factor (Mitigating influence)</div>
            </div>
            """, unsafe_allow_html=True)

    # Row 5: Prescriptive Maintenance Protocol
    if "prescriptive_plan" in diag and diag["prescriptive_plan"]:
        plan = diag["prescriptive_plan"]
        st.markdown("---")
        st.subheader(f"Prescriptive Maintenance & Root-Cause Avoidance Protocol: {diag['predicted_fault']}")

        plan_c1, plan_c2 = st.columns(2)
        with plan_c1:
            st.markdown("""
            <div class="protocol-card" style="border-top: 3px solid #DC2626;">
                <div style="font-size: 1.05rem; font-weight: bold; color: #991B1B; margin-bottom: 0.6rem;">Immediate Operator Action Protocol</div>
            """, unsafe_allow_html=True)
            for act in plan["immediate_actions"]:
                st.markdown(f"<div style='color: #1E293B; font-size: 0.92rem; margin-bottom: 0.5rem;'>• {act}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with plan_c2:
            st.markdown("""
            <div class="protocol-card" style="border-top: 3px solid #059669;">
                <div style="font-size: 1.05rem; font-weight: bold; color: #065F46; margin-bottom: 0.6rem;">How to Avoid This Fault in the Future</div>
            """, unsafe_allow_html=True)
            for rec in plan["how_to_avoid"]:
                st.markdown(f"<div style='color: #1E293B; font-size: 0.92rem; margin-bottom: 0.5rem;'>• {rec}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("System could not initialize predictor models. Please check outputs directory.")
