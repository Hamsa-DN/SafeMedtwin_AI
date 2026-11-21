"""
SafeMedTwin — Streamlit Dashboard
====================================
Interactive frontend: disease risk prediction, SHAP explanations, Digital
Twin what-if simulation, MedGuard safety flags, and LLM vs SLM comparison.

Run:  streamlit run frontend/app.py
Requires the backend running at BACKEND_URL (default http://localhost:8000).
"""
import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from body_twin import generate_body_twin_html, organ_data_from_predictions

BACKEND_URL = os.environ.get("SAFEMEDTWIN_BACKEND", "http://localhost:8000")

st.set_page_config(page_title="SafeMedTwin", page_icon="🩺", layout="wide")

DISEASE_LABELS = {
    "diabetes": "Diabetes", "cvd": "Cardiovascular Disease", "ckd": "Chronic Kidney Disease",
    "liver": "Liver Disease", "hypertension": "Hypertension",
}

RISK_COLORS = {"low": "#2E8B57", "moderate": "#D9A200", "high": "#C0392B"}


def api_get(path):
    r = requests.get(f"{BACKEND_URL}{path}", timeout=15)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=30)
    if r.status_code >= 400:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    return r.json()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 SafeMedTwin")
st.sidebar.caption("Hallucination-Aware Medical AI with Small Language Models")

try:
    health = api_get("/health")
    st.sidebar.success("Backend connected")
except Exception:
    st.sidebar.error(f"Cannot reach backend at {BACKEND_URL}. Start it with:\n\nuvicorn backend.main:app --port 8000")
    st.stop()

patient_id = st.sidebar.text_input("Patient ID", value="patient_001")
disease = st.sidebar.selectbox("Disease", list(DISEASE_LABELS.keys()), format_func=lambda d: DISEASE_LABELS[d])

schema = api_get(f"/diseases/{disease}/features")
feature_cols = schema["features"]
model_metrics = schema["metrics"]

st.sidebar.markdown("---")
st.sidebar.caption(f"Model AUC: **{model_metrics['roc_auc']:.3f}**  ·  Trained on {model_metrics['n_train']} patients")

# ------------------------------------------------------------------- input
st.title(f"{DISEASE_LABELS[disease]} Risk Assessment")

# default plausible values per feature, keyed on common medical ranges
DEFAULTS = {
    "age": 45, "bmi": 27.0, "glucose": 110.0, "blood_pressure": 78.0, "insulin": 80.0,
    "skin_thickness": 23.0, "pregnancies": 1, "family_history": 0, "physical_activity_hrs_wk": 4.0,
    "sex_male": 1, "cholesterol": 200.0, "resting_bp": 125.0, "max_heart_rate": 150.0, "smoker": 0,
    "diabetes_flag": 0, "exercise_angina": 0, "blood_urea": 40.0, "serum_creatinine": 1.0,
    "hemoglobin": 13.5, "albumin_grade": 0, "hypertension": 0, "blood_glucose": 120.0,
    "total_bilirubin": 0.8, "alk_phosphotase": 200.0, "sgpt_alt": 30.0, "sgot_ast": 30.0,
    "albumin": 3.5, "alcohol_use": 0, "hepatitis_history": 0, "sodium_intake_mg": 3200.0,
    "salt_sensitivity": 0, "stress_level": 5.0,
}
BINARY_FEATURES = {"family_history", "sex_male", "smoker", "diabetes_flag", "exercise_angina",
                    "hypertension", "alcohol_use", "hepatitis_history", "salt_sensitivity"}

with st.form("patient_form"):
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(feature_cols):
        col = cols[i % 3]
        label = feat.replace("_", " ").title()
        default = DEFAULTS.get(feat, 0)
        if feat in BINARY_FEATURES:
            values[feat] = col.selectbox(label, [0, 1], index=int(default), key=feat)
        elif feat in ("pregnancies", "albumin_grade", "age"):
            values[feat] = col.number_input(label, min_value=0, value=int(default), step=1, key=feat)
        else:
            values[feat] = col.number_input(label, value=float(default), step=0.5, key=feat)
    submitted = st.form_submit_button("Run Prediction", type="primary")

if submitted:
    prediction = api_post("/predict", {
        "disease": disease, "patient_id": patient_id, "features": values,
    })
    st.session_state["last_prediction"] = prediction
    st.session_state["last_baseline"] = values
    st.session_state.setdefault("disease_results", {})[disease] = prediction

if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]
    baseline = st.session_state["last_baseline"]
    band = result["risk_band"]
    color = RISK_COLORS[band]

    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=result["probability"] * 100,
            number={"suffix": "%"},
            title={"text": f"{band.upper()} RISK"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 33], "color": "#EAF5EE"},
                             {"range": [33, 66], "color": "#FCF3D9"},
                             {"range": [66, 100], "color": "#FBEAE8"}]},
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Explanation (SHAP)")
        st.write(result["explanation"])
        contrib_df = pd.DataFrame(result["top_contributions"])
        fig2 = px.bar(
            contrib_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
            color="shap_value", color_continuous_scale=["#2E8B57", "#C0392B"],
            labels={"shap_value": "SHAP contribution", "feature": ""},
        )
        fig2.update_layout(height=280, showlegend=False, coloraxis_showscale=False,
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width='stretch')

    if result["safety_warnings"]:
        st.warning("MedGuard flags:\n" + "\n".join(f"- {w}" for w in result["safety_warnings"]))
    else:
        st.info("✅ MedGuard: no safety flags on this prediction.")

    # ------------------------------------------------ 3D Digital Twin body
    st.markdown("---")
    st.subheader("🫀 3D Digital Twin — Affected Body Systems")
    disease_results = st.session_state.get("disease_results", {})
    unassessed = [DISEASE_LABELS[d] for d in DISEASE_LABELS if d not in disease_results]
    if unassessed:
        st.caption(
            "Organs are colored by this patient's latest assessed risk for each disease. "
            f"Not yet assessed (shown gray): {', '.join(unassessed)} — run a prediction for "
            "that disease from the sidebar to light it up."
        )
    organ_data = organ_data_from_predictions(disease_results)
    st.iframe(generate_body_twin_html(organ_data), height=540)

    # -------------------------------------------------------- Digital Twin
    st.markdown("---")
    st.subheader("🧬 What-If Simulation")
    top_feature = result["top_contributions"][0]["feature"]
    sim_cols = st.columns(3)
    with sim_cols[0]:
        sweep_feature = st.selectbox("Feature to vary", feature_cols,
                                      index=feature_cols.index(top_feature) if top_feature in feature_cols else 0)
    with sim_cols[1]:
        current_val = float(baseline.get(sweep_feature, 0))
        lo = st.number_input("Range low", value=max(0.0, current_val * 0.5))
    with sim_cols[2]:
        hi = st.number_input("Range high", value=current_val * 1.5 if current_val > 0 else 10.0)

    if st.button("Run sweep"):
        values_range = list(pd.Series(range(0, 11)).apply(lambda i: lo + (hi - lo) * i / 10))
        sweep_result = api_post("/simulate/sweep", {
            "disease": disease, "baseline": baseline, "feature": sweep_feature, "values": values_range,
        })
        sweep_df = pd.DataFrame(sweep_result)
        fig3 = px.line(sweep_df, x=sweep_feature, y="probability", markers=True,
                        title=f"Risk sensitivity to {sweep_feature.replace('_', ' ')}")
        fig3.add_hline(y=baseline_prob if (baseline_prob := result["probability"]) else 0,
                        line_dash="dot", annotation_text="current baseline")
        fig3.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig3, width='stretch')

    # -------------------------------------------------- LLM vs SLM compare
    st.markdown("---")
    st.subheader("🤖 LLM vs SLM — Recommendation Comparison")
    if st.button("Generate & compare recommendations"):
        comp = api_post("/compare/llm-slm", {
            "disease": disease, "risk_band": band, "top_factor": top_feature,
            "contributions_summary": result["explanation"],
        })
        cc1, cc2 = st.columns(2)
        for col, key, label in [(cc1, "slm", "SLM (local, offline)"), (cc2, "llm", "LLM (cloud API)")]:
            r = comp[key]
            with col:
                st.markdown(f"**{label}**")
                st.write(r["text"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{r['latency_ms']:.0f} ms")
                m2.metric("Accuracy", f"{r['accuracy_score']:.0%}")
                m3.metric("Completeness", f"{r['completeness_score']:.0%}")
                if r["hallucination_flags"]:
                    st.error("⚠️ " + "; ".join(r["hallucination_flags"]))
                else:
                    st.success("✅ No hallucination flags")

    # ------------------------------------------------------------ history
    st.markdown("---")
    st.subheader("📈 Patient History")
    history = api_get(f"/twin/{patient_id}/history?disease={disease}")
    if history:
        hist_df = pd.DataFrame(history)
        fig4 = px.line(hist_df, x="timestamp", y="probability", markers=True)
        fig4.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig4, width='stretch')
    else:
        st.caption("No prior recorded predictions for this patient yet.")
