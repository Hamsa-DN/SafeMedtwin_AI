"""
SafeMedTwin — Unified Web App
================================
Single-process Streamlit app (no separate FastAPI server needed) — this is
the deployable entry point for Streamlit Community Cloud or any single-
service host. It imports the prediction/explainability/safety/twin/LLM
modules directly rather than over HTTP.

Run locally:   streamlit run streamlit_app.py
Deploy:        push this repo to GitHub, then create a new app at
                https://share.streamlit.io pointing at streamlit_app.py

Login-gated: each user has their own account (SQLite-backed, see
auth/auth.py) and only ever sees their own predictions, chat history, and
3D twin state.
"""
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Load .env for local (non-Docker) runs, so teammates get the same
# GEMINI_API_KEY / ANTHROPIC_API_KEY setup whether they run this directly
# or via `docker compose up` (which uses its own env_file mechanism).
# Silently does nothing if .env doesn't exist or python-dotenv isn't
# installed — never breaks the app, just skips convenience.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.explainability import explain_prediction, explanation_to_sentence
from models.distilbert_inference import classify_patient, is_available as distilbert_available
from safety.medguard import validate_prediction, risk_band as compute_risk_band
from safety.alert_engine import evaluate as evaluate_alerts
from digital_twin.simulator import simulate_scenario, multi_scenario_sweep
from llm_slm.compare import compare as compare_llm_slm, consistency_test
from llm_slm.chatbot import chat as chatbot_reply
from diet.recommendations import get_rule_based_plan, get_personalized_plan
from vision.report_reader import scan_report_image
from vision.pdf_reader import scan_report_pdf
from vision.pdf_reader import scan_report_pdf
from auth import auth
from frontend.body_twin import generate_body_twin_html, organ_data_from_predictions

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DISEASES = ["diabetes", "cvd", "ckd", "liver", "hypertension"]
DISEASE_LABELS = {
    "diabetes": "Diabetes", "cvd": "Cardiovascular Disease", "ckd": "Chronic Kidney Disease",
    "liver": "Liver Disease", "hypertension": "Hypertension",
}
RISK_COLORS = {"low": "#2E8B57", "moderate": "#D9A200", "high": "#C0392B"}
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

st.set_page_config(page_title="SafeMedTwin", page_icon="🩺", layout="wide")


@st.cache_resource
def load_bundle(disease: str) -> dict:
    path = os.path.join(MODEL_DIR, f"{disease}_model.joblib")
    if not os.path.exists(path):
        st.error(
            f"Model file for '{disease}' not found at {path}.\n\n"
            "This project ships without pre-trained models (binary files "
            "don't survive file transfer reliably) — you need to train "
            "them once, locally, before running the app:\n\n"
            "```\npython data/generate_datasets.py\npython models/train_models.py\n```\n\n"
            "Then restart this app."
        )
        st.stop()
    try:
        bundle = joblib.load(path)
    except Exception as e:
        st.error(
            f"Could not load the model file for '{disease}' ({e}).\n\n"
            "This usually means the file got corrupted, or was trained with "
            "a different XGBoost version than the one installed now. Fix: "
            "delete it and retrain locally —\n\n"
            f"```\ndel models\\{disease}_model.joblib\npython models/train_models.py\n```"
        )
        st.stop()
    # Defensive: explicitly (re)attach feature names to the booster after
    # unpickling. XGBoost's internal feature-name metadata doesn't always
    # survive a joblib round-trip cleanly across library versions, which
    # can otherwise cause "data did not contain feature names" errors at
    # predict time even though we always pass a properly-columned
    # DataFrame. This makes prediction robust regardless of exact version.
    bundle["model"].get_booster().feature_names = bundle["feature_cols"]
    return bundle


# ============================================================ AUTH GATE ===
if "user_id" not in st.session_state:
    st.title("🩺 SafeMedTwin")
    st.caption("Hallucination-Aware Medical AI with Small Language Models — please log in to continue.")
    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            do_login = st.form_submit_button("Log in", type="primary")
        if do_login:
            ok, uid, msg = auth.authenticate(u, p)
            if ok:
                st.session_state["user_id"] = uid
                st.session_state["username"] = u.strip().lower()
                st.session_state.pop("login_u", None)
                st.session_state.pop("login_p", None)
                st.rerun()
            else:
                st.error(msg)

    with tab_register:
        with st.form("register_form"):
            u2 = st.text_input("Choose a username", key="reg_u")
            p2 = st.text_input("Choose a password (min 6 characters)", type="password", key="reg_p")
            do_register = st.form_submit_button("Create account", type="primary")
        if do_register:
            ok, msg = auth.register_user(u2, p2)
            if ok:
                st.success(msg + " You can log in now.")
            else:
                st.error(msg)

    st.info("Your data (assessments, chat history) is private to your account — other users can't see it.")
    st.stop()

user_id = st.session_state["user_id"]
username = st.session_state["username"]

# =========================================================== SIDEBAR ======
st.sidebar.title("🩺 SafeMedTwin")
st.sidebar.caption(f"Logged in as **{username}**")
if st.sidebar.button("Log out", key="btn_logout"):
    st.session_state.pop("user_id", None)
    st.session_state.pop("username", None)
    st.rerun()

st.sidebar.markdown("---")
disease = st.sidebar.selectbox("Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="sb_disease")
bundle = load_bundle(disease)
feature_cols = bundle["feature_cols"]
st.sidebar.caption(f"Model AUC: **{bundle['metrics']['roc_auc']:.3f}** · trained on {bundle['metrics']['n_train']} patients")

user_history = auth.get_latest_prediction_per_disease(user_id)

# =============================================================== TABS =====
tab_assess, tab_twin, tab_diet, tab_chat, tab_scan, tab_history = st.tabs(
    ["🩺 Risk Assessment", "🫀 3D Twin & What-If", "🥗 Diet & Lifestyle", "🤖 Chatbot",
     "📄 Report Scanner", "📈 History"]
)

# ------------------------------------------------------- Risk Assessment --
with tab_assess:
    st.header(f"{DISEASE_LABELS[disease]} Risk Assessment")

    prefill = st.session_state.get("scan_prefill", {})
    with st.form("patient_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(feature_cols):
            col = cols[i % 3]
            label = feat.replace("_", " ").title()
            default = prefill.get(feat, DEFAULTS.get(feat, 0))
            if feat in BINARY_FEATURES:
                values[feat] = col.selectbox(label, [0, 1], index=int(default), key=f"a_{feat}")
            elif feat in ("pregnancies", "albumin_grade", "age"):
                values[feat] = col.number_input(label, min_value=0, value=int(default), step=1, key=f"a_{feat}")
            else:
                values[feat] = col.number_input(label, value=float(default), step=0.5, key=f"a_{feat}")
        submitted = st.form_submit_button("Run Prediction", type="primary")

    if submitted:
        row = pd.DataFrame([values])[feature_cols]
        probability = float(bundle["model"].predict_proba(row)[0, 1])
        band = compute_risk_band(probability)
        safety = validate_prediction(disease, values, probability)
        contributions = explain_prediction(bundle, row, top_k=5)
        explanation = explanation_to_sentence(disease, contributions)

        previous_band = user_history.get(disease, {}).get("risk_band")
        alerts = evaluate_alerts(disease, values, band, previous_band=previous_band)

        auth.save_prediction(user_id, disease, probability, band, values)
        user_history[disease] = {"probability": probability, "risk_band": band, "features": values}

        st.session_state["last_result"] = {
            "probability": probability, "risk_band": band, "safety": safety,
            "contributions": contributions, "explanation": explanation, "alerts": alerts,
            "baseline": values, "disease": disease,
        }
        st.session_state.pop("scan_prefill", None)

    result = st.session_state.get("last_result")
    if result and result["disease"] == disease:
        band = result["risk_band"]
        color = RISK_COLORS[band]
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"},
                title={"text": f"{band.upper()} RISK"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 33], "color": "#EAF5EE"}, {"range": [33, 66], "color": "#FCF3D9"},
                                 {"range": [66, 100], "color": "#FBEAE8"}]},
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.subheader("Explanation (SHAP)")
            st.write(result["explanation"])
            contrib_df = pd.DataFrame(result["contributions"])
            fig2 = px.bar(contrib_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
                           color="shap_value", color_continuous_scale=["#2E8B57", "#C0392B"],
                           labels={"shap_value": "SHAP contribution", "feature": ""})
            fig2.update_layout(height=280, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, width='stretch')

        if result["safety"].warnings:
            st.warning("MedGuard flags:\n" + "\n".join(f"- {w}" for w in result["safety"].warnings))
        else:
            st.info("✅ MedGuard: no safety flags on this prediction.")

        if result["alerts"]:
            for a in result["alerts"]:
                (st.error if a.severity == "critical" else st.warning)(f"🔔 {a.message}")

    elif not result:
        st.caption("Fill in the form above and run a prediction to see results.")

# ---------------------------------------------------------------- 3D Twin --
with tab_twin:
    st.header("🫀 3D Digital Twin & What-If Simulation")
    unassessed = [DISEASE_LABELS[d] for d in DISEASES if d not in user_history]
    if unassessed:
        st.caption(f"Not yet assessed (shown gray): {', '.join(unassessed)}")
    organ_data = organ_data_from_predictions(user_history)
    st.iframe(generate_body_twin_html(organ_data), height=540)

    st.markdown("---")
    st.subheader("What-If Simulation")
    if disease not in user_history:
        st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable simulation.")
    else:
        baseline = user_history[disease]["features"]
        sweep_feature = st.selectbox("Feature to vary", feature_cols, key="sweep_feat")
        current_val = float(baseline.get(sweep_feature, 0))
        c1, c2 = st.columns(2)
        lo = c1.number_input("Range low", value=max(0.0, current_val * 0.5), key="sweep_lo")
        hi = c2.number_input("Range high", value=current_val * 1.5 if current_val > 0 else 10.0, key="sweep_hi")
        if st.button("Run sweep", key="btn_sweep"):
            try:
                values_range = [lo + (hi - lo) * i / 10 for i in range(11)]
                sweep_df = pd.DataFrame(multi_scenario_sweep(bundle, baseline, sweep_feature, values_range))
                fig3 = px.line(sweep_df, x=sweep_feature, y="probability", markers=True,
                                title=f"Risk sensitivity to {sweep_feature.replace('_', ' ')}")
                fig3.update_yaxes(range=[0, 1], tickformat=".0%")
                st.plotly_chart(fig3, width='stretch')
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # ---------------------------------------------- LLM vs SLM comparison --
    st.markdown("---")
    st.subheader("🤖 LLM vs SLM Recommendation Comparison")
    if disease not in user_history:
        st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable this comparison.")
    else:
        band = user_history[disease]["risk_band"]
        row = pd.DataFrame([user_history[disease]["features"]])[feature_cols]
        top_factor = explain_prediction(bundle, row, top_k=1)[0]["feature"]

        cbtn1, cbtn2 = st.columns(2)
        if cbtn1.button("Generate & compare", key="btn_compare"):
            comp = compare_llm_slm(disease, band, top_factor)
            cc1, cc2 = st.columns(2)
            for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
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

        if cbtn2.button("Run consistency test (3 runs each)", key="btn_consistency"):
            with st.spinner("Running each model 3 times..."):
                ctest = consistency_test(disease, band, top_factor, n_runs=3)
            cc1, cc2 = st.columns(2)
            for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
                r = ctest[key]
                with col:
                    st.markdown(f"**{label} consistency score: {r['consistency_score']:.0%}**")
                    st.caption("1.0 = identical outputs every run; lower = more variance run-to-run")
                    for i, out in enumerate(r["outputs"], 1):
                        with st.expander(f"Run {i}"):
                            st.write(out)

    # -------------------------------------------- DistilBERT classifier --
    st.markdown("---")
    st.subheader("🔤 DistilBERT Text Classifier (complementary to the SLM)")
    st.caption(
        "TinyLlama (the SLM above) *generates* recommendations — a different job from "
        "classification. DistilBERT here is fine-tuned to classify risk band directly "
        "from a text description of the patient, showing what a small encoder model is "
        "actually built for: fast, cheap classification rather than free text generation."
    )
    if not distilbert_available():
        st.info(
            "No fine-tuned DistilBERT classifier found yet. Train one (takes a few minutes "
            "on CPU, faster with a GPU):\n\n"
            "```\npip install -r requirements-slm.txt\n"
            "python models/train_distilbert_classifier.py\n```"
        )
    elif disease not in user_history:
        st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to classify this patient.")
    else:
        if st.button("Classify with DistilBERT", key="btn_distilbert"):
            row = pd.DataFrame([user_history[disease]["features"]])[feature_cols].iloc[0]
            try:
                result = classify_patient(disease, row)
                st.code(result["text"], language=None)
                c1, c2, c3 = st.columns(3)
                c1.metric("DistilBERT says", result["label"])
                c2.metric("Confidence", f"{result['confidence']:.0%}")
                c3.metric("Latency", f"{result['latency_ms']:.1f} ms")
                xgb_band = user_history[disease]["risk_band"]
                if result["label"] == xgb_band:
                    st.success(f"✅ Agrees with the XGBoost model's prediction ({xgb_band})")
                else:
                    st.warning(f"⚠️ Disagrees with the XGBoost model ({xgb_band}) — worth noting "
                               f"in your report as a genuine model-comparison finding, not a bug.")
                with st.expander("Full probability distribution"):
                    st.json(result["probabilities"])
            except Exception as e:
                st.error(f"Classification failed: {e}")

# ------------------------------------------------------- Diet & Lifestyle --
with tab_diet:
    st.header("🥗 Diet & Lifestyle Recommendations")
    if disease not in user_history:
        st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first for a personalized plan.")
        base_plan = get_rule_based_plan(disease, "unknown")
        st.write(base_plan["framing"])
    else:
        band = user_history[disease]["risk_band"]
        top_factors = [c["feature"] for c in explain_prediction(
            bundle, pd.DataFrame([user_history[disease]["features"]])[feature_cols], top_k=3
        )]
        if st.button("Generate personalized plan", key="btn_diet"):
            plan = get_personalized_plan(disease, band, top_factors)
            st.session_state["diet_plan"] = plan
        base_plan = st.session_state.get("diet_plan") or get_rule_based_plan(disease, band)

    st.info(base_plan["framing"])
    if base_plan.get("personalized_note"):
        st.markdown(f"**For you:** {base_plan['personalized_note']}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Diet")
        for item in base_plan["diet"]:
            st.markdown(f"- {item}")
    with c2:
        st.subheader("Lifestyle")
        for item in base_plan["lifestyle"]:
            st.markdown(f"- {item}")
    st.caption("General educational guidance only — not a substitute for advice from your physician or a registered dietitian.")

# ------------------------------------------------------------------ Chat --
with tab_chat:
    st.header("🤖 SafeMedTwin Assistant")
    st.caption("Ask about your assessments — this assistant can see your latest results but never diagnoses.")

    for msg in auth.get_chat_history(user_id):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask a question about your health data...")
    if user_msg:
        with st.chat_message("user"):
            st.write(user_msg)
        auth.save_chat_message(user_id, "user", user_msg)

        history_for_api = [{"role": m["role"], "content": m["content"]} for m in auth.get_chat_history(user_id, limit=10)]
        reply = chatbot_reply(user_msg, user_history, conversation_history=history_for_api[:-1])
        with st.chat_message("assistant"):
            st.write(reply)
        auth.save_chat_message(user_id, "assistant", reply)

# ------------------------------------------------------- Report Scanner --
with tab_scan:
    st.header("📄 Medical Report Scanner")
    st.caption("Upload a photo/screenshot OR a PDF of a lab report. Extracted values can prefill "
               "the assessment form above — nothing is auto-submitted.")

    uploaded = st.file_uploader("Upload report (image or PDF)", type=["png", "jpg", "jpeg", "pdf"], key="report_upload")
    if uploaded and st.button("Scan report", key="btn_scan"):
        is_pdf = uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")
        with st.spinner("Reading report..."):
            result = scan_report_pdf(uploaded.getvalue()) if is_pdf else scan_report_image(uploaded.getvalue())

        if "error" in result:
            st.error(result["error"])
            if result.get("raw"):
                with st.expander("Show raw model response (for debugging)"):
                    st.code(result["raw"])
        elif not result.get("legible", True):
            st.warning(result.get("summary", "Could not read this report."))
        else:
            st.success(f"Report read successfully ({result.get('read_method', 'image')}).")
            st.write(result["summary"])
            if result.get("safety_flags"):
                st.caption("⚠️ " + "; ".join(result["safety_flags"]))
            extracted = result.get("extracted_values", {})
            if extracted:
                st.subheader("Extracted values")
                st.json(extracted)
                if st.button("Use these values to prefill the assessment form", key="btn_prefill"):
                    st.session_state["scan_prefill"] = {k: v for k, v in extracted.items() if k in feature_cols}
                    st.success("Prefilled — go to the Risk Assessment tab and run a prediction.")

# ------------------------------------------------------------- History --
with tab_history:
    st.header("📈 Your Assessment History")
    all_predictions = auth.get_user_predictions(user_id)
    if not all_predictions:
        st.caption("No assessments recorded yet — run a prediction in the Risk Assessment tab first.")
    else:
        hist_disease = st.selectbox(
            "Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="history_disease"
        )
        disease_rows = [r for r in all_predictions if r["disease"] == hist_disease]
        if not disease_rows:
            st.caption(f"No {DISEASE_LABELS[hist_disease]} assessments yet for your account.")
        else:
            hist_df = pd.DataFrame(disease_rows)[["created_at", "probability", "risk_band"]]
            hist_df["created_at"] = pd.to_datetime(hist_df["created_at"])
            fig5 = px.line(hist_df, x="created_at", y="probability", markers=True,
                            title=f"{DISEASE_LABELS[hist_disease]} risk over time")
            fig5.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig5, width='stretch')

            st.subheader("All assessments")
            display_df = hist_df.rename(columns={
                "created_at": "Date", "probability": "Risk %", "risk_band": "Risk Band"
            })
            display_df["Risk %"] = (display_df["Risk %"] * 100).round(1)
            st.dataframe(display_df.sort_values("Date", ascending=False), width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Across all diseases")
        summary_rows = []
        for d in DISEASES:
            d_rows = [r for r in all_predictions if r["disease"] == d]
            if d_rows:
                latest = max(d_rows, key=lambda r: r["created_at"])
                summary_rows.append({
                    "Disease": DISEASE_LABELS[d], "Latest Risk %": round(latest["probability"] * 100, 1),
                    "Risk Band": latest["risk_band"], "Assessments": len(d_rows),
                    "Last Assessed": latest["created_at"],
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
