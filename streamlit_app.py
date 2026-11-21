# # """
# # SafeMedTwin — Unified Web App
# # ================================
# # Single-process Streamlit app (no separate FastAPI server needed) — this is
# # the deployable entry point for Streamlit Community Cloud or any single-
# # service host. It imports the prediction/explainability/safety/twin/LLM
# # modules directly rather than over HTTP.

# # Run locally:   streamlit run streamlit_app.py
# # Deploy:        push this repo to GitHub, then create a new app at
# #                 https://share.streamlit.io pointing at streamlit_app.py

# # Login-gated: each user has their own account (SQLite-backed, see
# # auth/auth.py) and only ever sees their own predictions, chat history, and
# # 3D twin state.
# # """
# # import os
# # import joblib
# # import pandas as pd
# # import streamlit as st
# # import plotly.graph_objects as go
# # import plotly.express as px

# # # Load .env for local (non-Docker) runs, so teammates get the same
# # # GEMINI_API_KEY / ANTHROPIC_API_KEY setup whether they run this directly
# # # or via `docker compose up` (which uses its own env_file mechanism).
# # # Silently does nothing if .env doesn't exist or python-dotenv isn't
# # # installed — never breaks the app, just skips convenience.
# # try:
# #     from dotenv import load_dotenv
# #     load_dotenv()
# # except ImportError:
# #     pass

# # from models.explainability import explain_prediction, explanation_to_sentence
# # from models.distilbert_inference import classify_patient, is_available as distilbert_available
# # from safety.medguard import validate_prediction, risk_band as compute_risk_band
# # from safety.alert_engine import evaluate as evaluate_alerts
# # from digital_twin.simulator import simulate_scenario, multi_scenario_sweep
# # from llm_slm.compare import compare as compare_llm_slm, consistency_test
# # from llm_slm.chatbot import chat as chatbot_reply
# # from diet.recommendations import get_rule_based_plan, get_personalized_plan
# # from vision.report_reader import scan_report_image
# # from vision.pdf_reader import scan_report_pdf
# # from vision.pdf_reader import scan_report_pdf
# # from auth import auth
# # from frontend.body_twin import generate_body_twin_html, organ_data_from_predictions

# # MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# # DISEASES = ["diabetes", "cvd", "ckd", "liver", "hypertension"]
# # DISEASE_LABELS = {
# #     "diabetes": "Diabetes", "cvd": "Cardiovascular Disease", "ckd": "Chronic Kidney Disease",
# #     "liver": "Liver Disease", "hypertension": "Hypertension",
# # }
# # RISK_COLORS = {"low": "#2E8B57", "moderate": "#D9A200", "high": "#C0392B"}
# # DEFAULTS = {
# #     "age": 45, "bmi": 27.0, "glucose": 110.0, "blood_pressure": 78.0, "insulin": 80.0,
# #     "skin_thickness": 23.0, "pregnancies": 1, "family_history": 0, "physical_activity_hrs_wk": 4.0,
# #     "sex_male": 1, "cholesterol": 200.0, "resting_bp": 125.0, "max_heart_rate": 150.0, "smoker": 0,
# #     "diabetes_flag": 0, "exercise_angina": 0, "blood_urea": 40.0, "serum_creatinine": 1.0,
# #     "hemoglobin": 13.5, "albumin_grade": 0, "hypertension": 0, "blood_glucose": 120.0,
# #     "total_bilirubin": 0.8, "alk_phosphotase": 200.0, "sgpt_alt": 30.0, "sgot_ast": 30.0,
# #     "albumin": 3.5, "alcohol_use": 0, "hepatitis_history": 0, "sodium_intake_mg": 3200.0,
# #     "salt_sensitivity": 0, "stress_level": 5.0,
# # }
# # BINARY_FEATURES = {"family_history", "sex_male", "smoker", "diabetes_flag", "exercise_angina",
# #                     "hypertension", "alcohol_use", "hepatitis_history", "salt_sensitivity"}

# # st.set_page_config(page_title="SafeMedTwin", page_icon="🩺", layout="wide")


# # @st.cache_resource
# # def load_bundle(disease: str) -> dict:
# #     path = os.path.join(MODEL_DIR, f"{disease}_model.joblib")
# #     if not os.path.exists(path):
# #         st.error(
# #             f"Model file for '{disease}' not found at {path}.\n\n"
# #             "This project ships without pre-trained models (binary files "
# #             "don't survive file transfer reliably) — you need to train "
# #             "them once, locally, before running the app:\n\n"
# #             "```\npython data/generate_datasets.py\npython models/train_models.py\n```\n\n"
# #             "Then restart this app."
# #         )
# #         st.stop()
# #     try:
# #         bundle = joblib.load(path)
# #     except Exception as e:
# #         st.error(
# #             f"Could not load the model file for '{disease}' ({e}).\n\n"
# #             "This usually means the file got corrupted, or was trained with "
# #             "a different XGBoost version than the one installed now. Fix: "
# #             "delete it and retrain locally —\n\n"
# #             f"```\ndel models\\{disease}_model.joblib\npython models/train_models.py\n```"
# #         )
# #         st.stop()
# #     # Defensive: explicitly (re)attach feature names to the booster after
# #     # unpickling. XGBoost's internal feature-name metadata doesn't always
# #     # survive a joblib round-trip cleanly across library versions, which
# #     # can otherwise cause "data did not contain feature names" errors at
# #     # predict time even though we always pass a properly-columned
# #     # DataFrame. This makes prediction robust regardless of exact version.
# #     bundle["model"].get_booster().feature_names = bundle["feature_cols"]
# #     return bundle


# # # ============================================================ AUTH GATE ===
# # # if "user_id" not in st.session_state:
# # #     st.title("🩺 SafeMedTwin")
# # #     st.caption("Hallucination-Aware Medical AI with Small Language Models — please log in to continue.")
# # #     tab_login, tab_register = st.tabs(["Log in", "Create account"])

# # #     with tab_login:
# # #         with st.form("login_form"):
# # #             u = st.text_input("Username", key="login_u")
# # #             p = st.text_input("Password", type="password", key="login_p")
# # #             do_login = st.form_submit_button("Log in", type="primary")
# # #         if do_login:
# # #             ok, uid, msg = auth.authenticate(u, p)
# # #             if ok:
# # #                 st.session_state["user_id"] = uid
# # #                 st.session_state["username"] = u.strip().lower()
# # #                 st.session_state.pop("login_u", None)
# # #                 st.session_state.pop("login_p", None)
# # #                 st.rerun()
# # #             else:
# # #                 st.error(msg)

# # #     with tab_register:
# # #         with st.form("register_form"):
# # #             u2 = st.text_input("Choose a username", key="reg_u")
# # #             p2 = st.text_input("Choose a password (min 6 characters)", type="password", key="reg_p")
# # #             do_register = st.form_submit_button("Create account", type="primary")
# # #         if do_register:
# # #             ok, msg = auth.register_user(u2, p2)
# # #             if ok:
# # #                 st.success(msg + " You can log in now.")
# # #             else:
# # #                 st.error(msg)

# # #     st.info("Your data (assessments, chat history) is private to your account — other users can't see it.")
# # #     st.stop()


# # # ============================================================ AUTH GATE ===
# # if "user_id" not in st.session_state:

# #     # Welcome page styling
# #     st.markdown("""
# #     <style>
# #     .stApp {
# #         background: linear-gradient(135deg, #f4f8fc 0%, #edf7f7 100%);
# #     }

# #     .hero-card {
# #         background: linear-gradient(145deg, #142b4a, #176b78);
# #         padding: 42px 34px;
# #         border-radius: 22px;
# #         color: white;
# #         min-height: 460px;
# #         box-shadow: 0 12px 35px rgba(20, 43, 74, 0.15);
# #     }

# #     .hero-card h1 {
# #         color: white;
# #         font-size: 43px;
# #         font-weight: 750;
# #         margin-bottom: 8px;
# #     }

# #     .hero-card .tagline {
# #         color: #9ee6dc;
# #         font-size: 19px;
# #         font-weight: 600;
# #         line-height: 1.5;
# #         margin-bottom: 22px;
# #     }

# #     .hero-card .description {
# #         color: #e1edf4;
# #         font-size: 15px;
# #         line-height: 1.8;
# #     }

    

# #     .feature {
# #         background: rgba(255,255,255,0.07);
# #         border: 1px solid rgba(255,255,255,0.13);
# #         border-radius: 10px;
# #         padding: 13px 16px;
# #         margin-top: 11px;
# #         color: #ffffff;
# #         font-size: 14px;
# #         font-weight: 650;
# #         line-height: 1.5;
# #     }

# #     .feature div {
# #         color: #c6dce5;
# #         font-size: 12px;
# #         font-weight: 400;
# #         margin-top: 4px;
# #         line-height: 1.5;
# #     }

# #     .login-heading {
# #         color: #183653;
# #         font-size: 29px;
# #         font-weight: 700;
# #         margin-bottom: 5px;
# #     }

# #     .login-subtitle {
# #         color: #718096;
# #         font-size: 14px;
# #         margin-bottom: 24px;
# #     }

# #     [data-testid="stForm"] {
# #         background: white;
# #         padding: 24px;
# #         border: 1px solid #e1e8ef;
# #         border-radius: 16px;
# #         box-shadow: 0 8px 25px rgba(30, 60, 90, 0.06);
# #     }

# #     [data-testid="stTextInput"] input {
# #         border-radius: 9px;
# #     }

# #     [data-testid="stFormSubmitButton"] button {
# #         background: #138b88;
# #         color: white;
# #         border-radius: 9px;
# #         border: none;
# #         font-weight: 600;
# #         padding: 10px 22px;
# #     }

# #     [data-testid="stFormSubmitButton"] button:hover {
# #         background: #106f70;
# #         color: white;
# #         border: none;
# #     }

# #     .privacy-note {
# #         text-align: center;
# #         color: #718096;
# #         font-size: 13px;
# #         margin-top: 24px;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)

# #     # Two-column welcome page
# #     left, right = st.columns([1.1, 0.9], gap="large")

# #     # with left:
# #     #     st.markdown("""
# #     #     <div class="hero-card">
# #     #         <div style="font-size:48px; margin-bottom:12px;">✚</div>
# #     #         <h1>SafeMedTwin</h1>
# #     #         <div class="tagline">
# #     #             Your Health. Your Digital Twin.<br>
# #     #             Smarter Decisions.
# #     #         </div>

# #     #         <div class="description">
# #     #             An AI-powered medical decision-support platform
# #     #             designed to understand patient health, predict
# #     #             disease risks and provide safety-aware insights.
# #     #         </div>

# #     #         <div class="feature">
# #     #             ◈ Patient Digital Twin & Health Monitoring
# #     #         </div>
# #     #         <div class="feature">
# #     #             ◈ Multi-Disease Risk Prediction
# #     #         </div>
# #     #         <div class="feature">
# #     #             ◈ AI-Powered Medical Recommendations
# #     #         </div>
# #     #         <div class="feature">
# #     #             ◈ MedGuard Safety Validation
# #     #         </div>
# #     #     </div>
# #     #     """, unsafe_allow_html=True)


# #     # with left:
# #     #     st.markdown("""
# #     #     <div class="hero-card">
# #     #         <div style="font-size:48px; margin-bottom:12px;">&#10010;</div>
# #     #         <h1>SafeMedTwin</h1>
# #     #         <div class="tagline">
# #     #             Your Health. Your Digital Twin.<br>
# #     #             Smarter Decisions.
# #     #         </div>
# #     #         <div class="description">
# #     #             An AI-powered medical decision-support platform
# #     #             designed to understand patient health, predict
# #     #             disease risks and provide safety-aware insights.
# #     #         </div>
# #     #         <div class="feature">
# #     #             &#9672; Patient Digital Twin &amp; Health Monitoring
# #     #         </div>
# #     #         <div class="feature">
# #     #             &#9672; Multi-Disease Risk Prediction
# #     #         </div>
# #     #         <div class="feature">
# #     #             &#9672; AI-Powered Medical Recommendations
# #     #         </div>
# #     #         <div class="feature">
# #     #             &#9672; MedGuard Safety Validation
# #     #         </div>
# #     #     </div>
# #     #     """, unsafe_allow_html=True)
    
# #     with left:
# #         st.markdown("""
# #         <div class="hero-card">
# #             <div style="font-size:42px; margin-bottom:10px;">&#10010;</div>
# #             <h1>SafeMedTwin</h1>
# #             <div class="tagline">
# #                 Intelligent Health Monitoring.<br>
# #                 Safer Medical Insights.
# #             </div>
# #             <div class="description">
# #                 A patient-centric medical decision-support platform
# #                 that creates a digital representation of individual
# #                 health profiles, evaluates potential disease risks
# #                 using machine learning, and generates AI-assisted
# #                 insights through an integrated safety-validation layer.
# #             </div>
# #             <div class="feature">
# #                 &#9672; Patient Digital Twin
# #                 <div>Consolidated health profiles and health monitoring</div>
# #             </div>
# #             <div class="feature">
# #                 &#9672; Multi-Disease Risk Assessment
# #                 <div>Machine-learning-based evaluation of potential health risks</div>
# #             </div>
# #             <div class="feature">
# #                 &#9672; AI-Assisted Recommendations
# #                 <div>Context-aware insights generated using language models</div>
# #             </div>
# #             <div class="feature">
# #                 &#9672; MedGuard Safety Layer
# #                 <div>Safety validation of AI-generated recommendations</div>
# #             </div>
# #             <div style="margin-top:22px; font-size:13px; color:#d6e9ef; line-height:1.6;">
# #                 Designed to support informed healthcare decisions,
# #                 not replace professional medical judgement.
# #             </div>
# #         </div>
# #         """, unsafe_allow_html=True)


# #     with right:
# #         st.markdown("""
# #         <div class="login-heading">Welcome !</div>
# #         <div class="login-subtitle">
# #             Sign in to access your personal health dashboard.
# #         </div>
# #         """, unsafe_allow_html=True)

# #         tab_login, tab_register = st.tabs(
# #             ["Log in", "Create account"]
# #         )

# #         with tab_login:
# #             with st.form("login_form"):
# #                 u = st.text_input("Username", key="login_u")
# #                 p = st.text_input(
# #                     "Password",
# #                     type="password",
# #                     key="login_p"
# #                 )
# #                 do_login = st.form_submit_button(
# #                     "Log in",
# #                     type="primary",
# #                     use_container_width=True
# #                 )

# #             if do_login:
# #                 ok, uid, msg = auth.authenticate(u, p)
# #                 if ok:
# #                     st.session_state["user_id"] = uid
# #                     st.session_state["username"] = u.strip().lower()
# #                     st.session_state.pop("login_u", None)
# #                     st.session_state.pop("login_p", None)
# #                     st.rerun()
# #                 else:
# #                     st.error(msg)

# #         with tab_register:
# #             with st.form("register_form"):
# #                 u2 = st.text_input(
# #                     "Choose a username",
# #                     key="reg_u"
# #                 )
# #                 p2 = st.text_input(
# #                     "Choose a password (min 6 characters)",
# #                     type="password",
# #                     key="reg_p"
# #                 )
# #                 do_register = st.form_submit_button(
# #                     "Create account",
# #                     type="primary",
# #                     use_container_width=True
# #                 )

# #             if do_register:
# #                 ok, msg = auth.register_user(u2, p2)
# #                 if ok:
# #                     st.success(msg + " You can log in now.")
# #                 else:
# #                     st.error(msg)

# #         st.markdown("""
# #         <div class="privacy-note">
# #             🔒 Your health assessments and chat history
# #             remain private to your account.
# #         </div>
# #         """, unsafe_allow_html=True)

# #     st.stop()


# # user_id = st.session_state["user_id"]
# # username = st.session_state["username"]

# # # =========================================================== SIDEBAR ======
# # st.sidebar.title("🩺 SafeMedTwin")
# # st.sidebar.caption(f"Logged in as **{username}**")
# # if st.sidebar.button("Log out", key="btn_logout"):
# #     st.session_state.pop("user_id", None)
# #     st.session_state.pop("username", None)
# #     st.rerun()

# # st.sidebar.markdown("---")
# # disease = st.sidebar.selectbox("Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="sb_disease")
# # bundle = load_bundle(disease)
# # feature_cols = bundle["feature_cols"]
# # st.sidebar.caption(f"Model AUC: **{bundle['metrics']['roc_auc']:.3f}** · trained on {bundle['metrics']['n_train']} patients")

# # user_history = auth.get_latest_prediction_per_disease(user_id)

# # # =============================================================== TABS =====
# # tab_assess, tab_twin, tab_diet, tab_chat, tab_scan, tab_history = st.tabs(
# #     ["🩺 Risk Assessment", "🫀 3D Twin & What-If", "🥗 Diet & Lifestyle", "🤖 Chatbot",
# #      "📄 Report Scanner", "📈 History"]
# # )

# # # ------------------------------------------------------- Risk Assessment --
# # with tab_assess:
# #     st.header(f"{DISEASE_LABELS[disease]} Risk Assessment")

# #     prefill = st.session_state.get("scan_prefill", {})
# #     with st.form("patient_form"):
# #         cols = st.columns(3)
# #         values = {}
# #         for i, feat in enumerate(feature_cols):
# #             col = cols[i % 3]
# #             label = feat.replace("_", " ").title()
# #             default = prefill.get(feat, DEFAULTS.get(feat, 0))
# #             if feat in BINARY_FEATURES:
# #                 values[feat] = col.selectbox(label, [0, 1], index=int(default), key=f"a_{feat}")
# #             elif feat in ("pregnancies", "albumin_grade", "age"):
# #                 values[feat] = col.number_input(label, min_value=0, value=int(default), step=1, key=f"a_{feat}")
# #             else:
# #                 values[feat] = col.number_input(label, value=float(default), step=0.5, key=f"a_{feat}")
# #         submitted = st.form_submit_button("Run Prediction", type="primary")

# #     if submitted:
# #         row = pd.DataFrame([values])[feature_cols]
# #         probability = float(bundle["model"].predict_proba(row)[0, 1])
# #         band = compute_risk_band(probability)
# #         safety = validate_prediction(disease, values, probability)
# #         contributions = explain_prediction(bundle, row, top_k=5)
# #         explanation = explanation_to_sentence(disease, contributions)

# #         previous_band = user_history.get(disease, {}).get("risk_band")
# #         alerts = evaluate_alerts(disease, values, band, previous_band=previous_band)

# #         auth.save_prediction(user_id, disease, probability, band, values)
# #         user_history[disease] = {"probability": probability, "risk_band": band, "features": values}

# #         st.session_state["last_result"] = {
# #             "probability": probability, "risk_band": band, "safety": safety,
# #             "contributions": contributions, "explanation": explanation, "alerts": alerts,
# #             "baseline": values, "disease": disease,
# #         }
# #         st.session_state.pop("scan_prefill", None)

# #     result = st.session_state.get("last_result")
# #     if result and result["disease"] == disease:
# #         band = result["risk_band"]
# #         color = RISK_COLORS[band]
# #         c1, c2 = st.columns([1, 2])
# #         with c1:
# #             fig = go.Figure(go.Indicator(
# #                 mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"},
# #                 title={"text": f"{band.upper()} RISK"},
# #                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
# #                        "steps": [{"range": [0, 33], "color": "#EAF5EE"}, {"range": [33, 66], "color": "#FCF3D9"},
# #                                  {"range": [66, 100], "color": "#FBEAE8"}]},
# #             ))
# #             fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
# #             st.plotly_chart(fig, width='stretch')
# #         with c2:
# #             st.subheader("Explanation (SHAP)")
# #             st.write(result["explanation"])
# #             contrib_df = pd.DataFrame(result["contributions"])
# #             fig2 = px.bar(contrib_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
# #                            color="shap_value", color_continuous_scale=["#2E8B57", "#C0392B"],
# #                            labels={"shap_value": "SHAP contribution", "feature": ""})
# #             fig2.update_layout(height=280, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
# #             st.plotly_chart(fig2, width='stretch')

# #         if result["safety"].warnings:
# #             st.warning("MedGuard flags:\n" + "\n".join(f"- {w}" for w in result["safety"].warnings))
# #         else:
# #             st.info("✅ MedGuard: no safety flags on this prediction.")

# #         if result["alerts"]:
# #             for a in result["alerts"]:
# #                 (st.error if a.severity == "critical" else st.warning)(f"🔔 {a.message}")

# #     elif not result:
# #         st.caption("Fill in the form above and run a prediction to see results.")

# # # ---------------------------------------------------------------- 3D Twin --
# # with tab_twin:
# #     st.header("🫀 3D Digital Twin & What-If Simulation")
# #     unassessed = [DISEASE_LABELS[d] for d in DISEASES if d not in user_history]
# #     if unassessed:
# #         st.caption(f"Not yet assessed (shown gray): {', '.join(unassessed)}")
# #     organ_data = organ_data_from_predictions(user_history)
# #     st.iframe(generate_body_twin_html(organ_data), height=540)

# #     st.markdown("---")
# #     st.subheader("What-If Simulation")
# #     if disease not in user_history:
# #         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable simulation.")
# #     else:
# #         baseline = user_history[disease]["features"]
# #         sweep_feature = st.selectbox("Feature to vary", feature_cols, key="sweep_feat")
# #         current_val = float(baseline.get(sweep_feature, 0))
# #         c1, c2 = st.columns(2)
# #         lo = c1.number_input("Range low", value=max(0.0, current_val * 0.5), key="sweep_lo")
# #         hi = c2.number_input("Range high", value=current_val * 1.5 if current_val > 0 else 10.0, key="sweep_hi")
# #         if st.button("Run sweep", key="btn_sweep"):
# #             try:
# #                 values_range = [lo + (hi - lo) * i / 10 for i in range(11)]
# #                 sweep_df = pd.DataFrame(multi_scenario_sweep(bundle, baseline, sweep_feature, values_range))
# #                 fig3 = px.line(sweep_df, x=sweep_feature, y="probability", markers=True,
# #                                 title=f"Risk sensitivity to {sweep_feature.replace('_', ' ')}")
# #                 fig3.update_yaxes(range=[0, 1], tickformat=".0%")
# #                 st.plotly_chart(fig3, width='stretch')
# #             except Exception as e:
# #                 st.error(f"Simulation failed: {e}")

# #     # ---------------------------------------------- LLM vs SLM comparison --
# #     st.markdown("---")
# #     st.subheader("🤖 LLM vs SLM Recommendation Comparison")
# #     if disease not in user_history:
# #         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable this comparison.")
# #     else:
# #         band = user_history[disease]["risk_band"]
# #         row = pd.DataFrame([user_history[disease]["features"]])[feature_cols]
# #         top_factor = explain_prediction(bundle, row, top_k=1)[0]["feature"]

# #         cbtn1, cbtn2 = st.columns(2)
# #         if cbtn1.button("Generate & compare", key="btn_compare"):
# #             comp = compare_llm_slm(disease, band, top_factor)
# #             cc1, cc2 = st.columns(2)
# #             for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
# #                 r = comp[key]
# #                 with col:
# #                     st.markdown(f"**{label}**")
# #                     st.write(r["text"])
# #                     m1, m2, m3 = st.columns(3)
# #                     m1.metric("Latency", f"{r['latency_ms']:.0f} ms")
# #                     m2.metric("Accuracy", f"{r['accuracy_score']:.0%}")
# #                     m3.metric("Completeness", f"{r['completeness_score']:.0%}")
# #                     if r["hallucination_flags"]:
# #                         st.error("⚠️ " + "; ".join(r["hallucination_flags"]))

# #         if cbtn2.button("Run consistency test (3 runs each)", key="btn_consistency"):
# #             with st.spinner("Running each model 3 times..."):
# #                 ctest = consistency_test(disease, band, top_factor, n_runs=3)
# #             cc1, cc2 = st.columns(2)
# #             for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
# #                 r = ctest[key]
# #                 with col:
# #                     st.markdown(f"**{label} consistency score: {r['consistency_score']:.0%}**")
# #                     st.caption("1.0 = identical outputs every run; lower = more variance run-to-run")
# #                     for i, out in enumerate(r["outputs"], 1):
# #                         with st.expander(f"Run {i}"):
# #                             st.write(out)

# #     # -------------------------------------------- DistilBERT classifier --
# #     st.markdown("---")
# #     st.subheader("🔤 DistilBERT Text Classifier (complementary to the SLM)")
# #     st.caption(
# #         "TinyLlama (the SLM above) *generates* recommendations — a different job from "
# #         "classification. DistilBERT here is fine-tuned to classify risk band directly "
# #         "from a text description of the patient, showing what a small encoder model is "
# #         "actually built for: fast, cheap classification rather than free text generation."
# #     )
# #     if not distilbert_available():
# #         st.info(
# #             "No fine-tuned DistilBERT classifier found yet. Train one (takes a few minutes "
# #             "on CPU, faster with a GPU):\n\n"
# #             "```\npip install -r requirements-slm.txt\n"
# #             "python models/train_distilbert_classifier.py\n```"
# #         )
# #     elif disease not in user_history:
# #         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to classify this patient.")
# #     else:
# #         if st.button("Classify with DistilBERT", key="btn_distilbert"):
# #             row = pd.DataFrame([user_history[disease]["features"]])[feature_cols].iloc[0]
# #             try:
# #                 result = classify_patient(disease, row)
# #                 st.code(result["text"], language=None)
# #                 c1, c2, c3 = st.columns(3)
# #                 c1.metric("DistilBERT says", result["label"])
# #                 c2.metric("Confidence", f"{result['confidence']:.0%}")
# #                 c3.metric("Latency", f"{result['latency_ms']:.1f} ms")
# #                 xgb_band = user_history[disease]["risk_band"]
# #                 if result["label"] == xgb_band:
# #                     st.success(f"✅ Agrees with the XGBoost model's prediction ({xgb_band})")
# #                 else:
# #                     st.warning(f"⚠️ Disagrees with the XGBoost model ({xgb_band}) — worth noting "
# #                                f"in your report as a genuine model-comparison finding, not a bug.")
# #                 with st.expander("Full probability distribution"):
# #                     st.json(result["probabilities"])
# #             except Exception as e:
# #                 st.error(f"Classification failed: {e}")

# # # ------------------------------------------------------- Diet & Lifestyle --
# # with tab_diet:
# #     st.header("🥗 Diet & Lifestyle Recommendations")
# #     if disease not in user_history:
# #         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first for a personalized plan.")
# #         base_plan = get_rule_based_plan(disease, "unknown")
# #         st.write(base_plan["framing"])
# #     else:
# #         band = user_history[disease]["risk_band"]
# #         top_factors = [c["feature"] for c in explain_prediction(
# #             bundle, pd.DataFrame([user_history[disease]["features"]])[feature_cols], top_k=3
# #         )]
# #         if st.button("Generate personalized plan", key="btn_diet"):
# #             plan = get_personalized_plan(disease, band, top_factors)
# #             st.session_state["diet_plan"] = plan
# #         base_plan = st.session_state.get("diet_plan") or get_rule_based_plan(disease, band)

# #     st.info(base_plan["framing"])
# #     if base_plan.get("personalized_note"):
# #         st.markdown(f"**For you:** {base_plan['personalized_note']}")
# #     c1, c2 = st.columns(2)
# #     with c1:
# #         st.subheader("Diet")
# #         for item in base_plan["diet"]:
# #             st.markdown(f"- {item}")
# #     with c2:
# #         st.subheader("Lifestyle")
# #         for item in base_plan["lifestyle"]:
# #             st.markdown(f"- {item}")
# #     st.caption("General educational guidance only — not a substitute for advice from your physician or a registered dietitian.")

# # # ------------------------------------------------------------------ Chat --
# # with tab_chat:
# #     st.header("🤖 SafeMedTwin Assistant")
# #     st.caption("Ask about your assessments — this assistant can see your latest results but never diagnoses.")

# #     for msg in auth.get_chat_history(user_id):
# #         with st.chat_message(msg["role"]):
# #             st.write(msg["content"])

# #     user_msg = st.chat_input("Ask a question about your health data...")
# #     if user_msg:
# #         with st.chat_message("user"):
# #             st.write(user_msg)
# #         auth.save_chat_message(user_id, "user", user_msg)

# #         history_for_api = [{"role": m["role"], "content": m["content"]} for m in auth.get_chat_history(user_id, limit=10)]
# #         reply = chatbot_reply(user_msg, user_history, conversation_history=history_for_api[:-1])
# #         with st.chat_message("assistant"):
# #             st.write(reply)
# #         auth.save_chat_message(user_id, "assistant", reply)

# # # ------------------------------------------------------- Report Scanner --
# # with tab_scan:
# #     st.header("📄 Medical Report Scanner")
# #     st.caption("Upload a photo/screenshot OR a PDF of a lab report. Extracted values can prefill "
# #                "the assessment form above — nothing is auto-submitted.")

# #     uploaded = st.file_uploader("Upload report (image or PDF)", type=["png", "jpg", "jpeg", "pdf"], key="report_upload")
# #     if uploaded and st.button("Scan report", key="btn_scan"):
# #         is_pdf = uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")
# #         with st.spinner("Reading report..."):
# #             result = scan_report_pdf(uploaded.getvalue()) if is_pdf else scan_report_image(uploaded.getvalue())

# #         if "error" in result:
# #             st.error(result["error"])
# #             if result.get("raw"):
# #                 with st.expander("Show raw model response (for debugging)"):
# #                     st.code(result["raw"])
# #         elif not result.get("legible", True):
# #             st.warning(result.get("summary", "Could not read this report."))
# #         else:
# #             st.success(f"Report read successfully ({result.get('read_method', 'image')}).")
# #             st.write(result["summary"])
# #             if result.get("safety_flags"):
# #                 st.caption("⚠️ " + "; ".join(result["safety_flags"]))
# #             extracted = result.get("extracted_values", {})
# #             if extracted:
# #                 st.subheader("Extracted values")
# #                 st.json(extracted)
# #                 if st.button("Use these values to prefill the assessment form", key="btn_prefill"):
# #                     st.session_state["scan_prefill"] = {k: v for k, v in extracted.items() if k in feature_cols}
# #                     st.success("Prefilled — go to the Risk Assessment tab and run a prediction.")

# # # ------------------------------------------------------------- History --
# # with tab_history:
# #     st.header("📈 Your Assessment History")
# #     all_predictions = auth.get_user_predictions(user_id)
# #     if not all_predictions:
# #         st.caption("No assessments recorded yet — run a prediction in the Risk Assessment tab first.")
# #     else:
# #         hist_disease = st.selectbox(
# #             "Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="history_disease"
# #         )
# #         disease_rows = [r for r in all_predictions if r["disease"] == hist_disease]
# #         if not disease_rows:
# #             st.caption(f"No {DISEASE_LABELS[hist_disease]} assessments yet for your account.")
# #         else:
# #             hist_df = pd.DataFrame(disease_rows)[["created_at", "probability", "risk_band"]]
# #             hist_df["created_at"] = pd.to_datetime(hist_df["created_at"])
# #             fig5 = px.line(hist_df, x="created_at", y="probability", markers=True,
# #                             title=f"{DISEASE_LABELS[hist_disease]} risk over time")
# #             fig5.update_yaxes(range=[0, 1], tickformat=".0%")
# #             st.plotly_chart(fig5, width='stretch')

# #             st.subheader("All assessments")
# #             display_df = hist_df.rename(columns={
# #                 "created_at": "Date", "probability": "Risk %", "risk_band": "Risk Band"
# #             })
# #             display_df["Risk %"] = (display_df["Risk %"] * 100).round(1)
# #             st.dataframe(display_df.sort_values("Date", ascending=False), width='stretch', hide_index=True)

# #         st.markdown("---")
# #         st.subheader("Across all diseases")
# #         summary_rows = []
# #         for d in DISEASES:
# #             d_rows = [r for r in all_predictions if r["disease"] == d]
# #             if d_rows:
# #                 latest = max(d_rows, key=lambda r: r["created_at"])
# #                 summary_rows.append({
# #                     "Disease": DISEASE_LABELS[d], "Latest Risk %": round(latest["probability"] * 100, 1),
# #                     "Risk Band": latest["risk_band"], "Assessments": len(d_rows),
# #                     "Last Assessed": latest["created_at"],
# #                 })
# #         if summary_rows:
# #             st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)




# """
# SafeMedTwin — Unified Web App
# ================================
# Single-process Streamlit app (no separate FastAPI server needed) — this is
# the deployable entry point for Streamlit Community Cloud or any single-
# service host. It imports the prediction/explainability/safety/twin/LLM
# modules directly rather than over HTTP.

# Run locally:   streamlit run streamlit_app.py
# Deploy:        push this repo to GitHub, then create a new app at
#                 https://share.streamlit.io pointing at streamlit_app.py

# Login-gated: each user has their own account (SQLite-backed, see
# auth/auth.py) and only ever sees their own predictions, chat history, and
# 3D twin state.
# """
# import os
# import joblib
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px

# # Load .env for local (non-Docker) runs, so teammates get the same
# # GEMINI_API_KEY / ANTHROPIC_API_KEY setup whether they run this directly
# # or via `docker compose up` (which uses its own env_file mechanism).
# # Silently does nothing if .env doesn't exist or python-dotenv isn't
# # installed — never breaks the app, just skips convenience.
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# from models.explainability import explain_prediction, explanation_to_sentence
# from models.distilbert_inference import classify_patient, is_available as distilbert_available
# from safety.medguard import validate_prediction, risk_band as compute_risk_band
# from safety.alert_engine import evaluate as evaluate_alerts
# from digital_twin.simulator import simulate_scenario, multi_scenario_sweep
# from llm_slm.compare import compare as compare_llm_slm, consistency_test
# from llm_slm.chatbot import chat as chatbot_reply
# from diet.recommendations import get_rule_based_plan, get_personalized_plan
# from vision.report_reader import scan_report_image
# from vision.pdf_reader import scan_report_pdf
# from vision.pdf_reader import scan_report_pdf
# from auth import auth
# from frontend.body_twin import generate_body_twin_html, organ_data_from_predictions

# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# DISEASES = ["diabetes", "cvd", "ckd", "liver", "hypertension"]
# DISEASE_LABELS = {
#     "diabetes": "Diabetes", "cvd": "Cardiovascular Disease", "ckd": "Chronic Kidney Disease",
#     "liver": "Liver Disease", "hypertension": "Hypertension",
# }
# RISK_COLORS = {"low": "#2E8B57", "moderate": "#D9A200", "high": "#C0392B"}
# DEFAULTS = {
#     "age": 45, "bmi": 27.0, "glucose": 110.0, "blood_pressure": 78.0, "insulin": 80.0,
#     "skin_thickness": 23.0, "pregnancies": 1, "family_history": 0, "physical_activity_hrs_wk": 4.0,
#     "sex_male": 1, "cholesterol": 200.0, "resting_bp": 125.0, "max_heart_rate": 150.0, "smoker": 0,
#     "diabetes_flag": 0, "exercise_angina": 0, "blood_urea": 40.0, "serum_creatinine": 1.0,
#     "hemoglobin": 13.5, "albumin_grade": 0, "hypertension": 0, "blood_glucose": 120.0,
#     "total_bilirubin": 0.8, "alk_phosphotase": 200.0, "sgpt_alt": 30.0, "sgot_ast": 30.0,
#     "albumin": 3.5, "alcohol_use": 0, "hepatitis_history": 0, "sodium_intake_mg": 3200.0,
#     "salt_sensitivity": 0, "stress_level": 5.0,
# }
# BINARY_FEATURES = {"family_history", "sex_male", "smoker", "diabetes_flag", "exercise_angina",
#                     "hypertension", "alcohol_use", "hepatitis_history", "salt_sensitivity"}

# st.set_page_config(page_title="SafeMedTwin", page_icon="🩺", layout="wide")


# @st.cache_resource
# def load_bundle(disease: str) -> dict:
#     path = os.path.join(MODEL_DIR, f"{disease}_model.joblib")
#     if not os.path.exists(path):
#         st.error(
#             f"Model file for '{disease}' not found at {path}.\n\n"
#             "This project ships without pre-trained models (binary files "
#             "don't survive file transfer reliably) — you need to train "
#             "them once, locally, before running the app:\n\n"
#             "```\npython data/generate_datasets.py\npython models/train_models.py\n```\n\n"
#             "Then restart this app."
#         )
#         st.stop()
#     try:
#         bundle = joblib.load(path)
#     except Exception as e:
#         st.error(
#             f"Could not load the model file for '{disease}' ({e}).\n\n"
#             "This usually means the file got corrupted, or was trained with "
#             "a different XGBoost version than the one installed now. Fix: "
#             "delete it and retrain locally —\n\n"
#             f"```\ndel models\\{disease}_model.joblib\npython models/train_models.py\n```"
#         )
#         st.stop()
#     # Defensive: explicitly (re)attach feature names to the booster after
#     # unpickling. XGBoost's internal feature-name metadata doesn't always
#     # survive a joblib round-trip cleanly across library versions, which
#     # can otherwise cause "data did not contain feature names" errors at
#     # predict time even though we always pass a properly-columned
#     # DataFrame. This makes prediction robust regardless of exact version.
#     bundle["model"].get_booster().feature_names = bundle["feature_cols"]
#     return bundle


# # ============================================================ AUTH GATE ===
# if "user_id" not in st.session_state:

#     # Clean, premium clinical welcome page. Authentication remains unchanged.
#     st.markdown("""
#     <style>
#     [data-testid="stAppViewContainer"] {
#         background:
#             radial-gradient(ellipse at 4% 8%, rgba(74, 190, 207, .18), transparent 32%),
#             radial-gradient(ellipse at 96% 92%, rgba(20, 145, 166, .12), transparent 30%),
#             linear-gradient(135deg, #edf7fa 0%, #f8fbfc 52%, #eaf4f8 100%);
#     }
#     [data-testid="stHeader"] { background: transparent; }
#     .block-container { max-width: 1480px; padding-top: 2.4rem; padding-bottom: 2rem; }

#     /* Restrained ECG detail; no repeating icons or wallpaper */
#     [data-testid="stAppViewContainer"]:before {
#         content: ""; position: fixed; pointer-events: none; inset: auto 0 0 auto;
#         width: 48vw; height: 190px; opacity: .16;
#         background-repeat: no-repeat; background-position: right bottom; background-size: contain;
#         background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 180'%3E%3Cpath d='M0 112h250l22-2 22-40 28 88 27-55 20 9h531' fill='none' stroke='%234ba9bd' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
#     }

#     .med-hero {
#         position: relative; isolation: isolate; overflow: hidden;
#         min-height: 675px; padding: 42px 38px 30px; border-radius: 26px;
#         color: #f4fbff;
#         background: linear-gradient(145deg, #062b48 0%, #073f5b 58%, #086d78 100%);
#         border: 1px solid rgba(75, 220, 222, .58);
#         box-shadow: 0 22px 60px rgba(9, 53, 78, .18), inset 0 1px 0 rgba(255,255,255,.08);
#     }
#     .med-hero:before {
#         content: ""; position: absolute; z-index: -1; width: 430px; height: 430px;
#         top: -245px; right: -160px; border-radius: 50%;
#         border: 1px solid rgba(92,226,231,.19);
#         box-shadow: 0 0 0 28px rgba(92,226,231,.035), 0 0 0 62px rgba(92,226,231,.025);
#     }
#     .med-hero:after {
#         content: ""; position: absolute; z-index: -1; width: 420px; height: 220px;
#         right: -130px; bottom: -130px;
#         background: radial-gradient(ellipse, rgba(15,194,184,.22), transparent 68%);
#     }
#     .brand-row { display:flex; align-items:center; gap:15px; margin-bottom:45px; }
#     .brand-mark { width:55px; height:55px; flex:0 0 55px; }
#     .brand-name { color:#f7fdff; font-size:36px; line-height:1.1; font-weight:780; letter-spacing:-1px; }
#     .brand-name span { color:#55ded6; }
#     .hero-kicker { color:#73e5df; text-transform:uppercase; letter-spacing:2.1px; font-size:11px; font-weight:750; margin-bottom:12px; }
#     .hero-tagline { color:#ffffff; font-size:29px; line-height:1.22; letter-spacing:-.5px; font-weight:750; margin:0 0 15px; max-width:590px; }
#     .hero-tagline span { color:#64e1d9; }
#     .hero-rule { width:76px; height:3px; border-radius:4px; background:linear-gradient(90deg,#27ded2,rgba(39,222,210,0)); margin:0 0 20px; }
#     .hero-description { max-width:590px; color:#d7eaf1; font-size:14px; line-height:1.75; margin:0 0 28px; }

#     .capability-heading { color:#a8dce2; font-size:10px; font-weight:750; letter-spacing:1.7px; text-transform:uppercase; margin:0 0 11px; }
#     .feature-list { display:grid; gap:10px; }
#     .med-feature { display:flex; align-items:center; gap:14px; min-height:67px; padding:10px 14px; border-radius:13px;
#         background:linear-gradient(100deg,rgba(75,185,208,.16),rgba(255,255,255,.055));
#         border:1px solid rgba(163,224,234,.17); }
#     .med-feature-icon { width:43px; height:43px; flex:0 0 43px; display:grid; place-items:center; border-radius:12px;
#         color:#69e9e1; background:rgba(27,188,194,.13); border:1px solid rgba(87,224,222,.25); }
#     .med-feature-icon svg { width:24px; height:24px; }
#     .med-feature-title { color:#f5fcff; font-size:13.5px; font-weight:720; line-height:1.3; }
#     .med-feature-desc { color:#c3dce5; font-size:11.5px; line-height:1.4; margin-top:4px; }
#     .hero-footer { display:flex; align-items:center; gap:13px; margin-top:24px; padding-top:17px;
#         border-top:1px solid rgba(177,226,233,.16); color:#cce5eb; font-size:11.5px; line-height:1.5; }
#     .footer-mark { width:28px; height:28px; flex:0 0 28px; display:grid; place-items:center; border-radius:50%;
#         color:#62e5dc; border:1px solid rgba(98,229,220,.34); font-size:15px; }

#     .login-shell { padding:22px 8px 0 12px; }
#     .login-welcome { color:#0b304d; font-size:36px; font-weight:780; letter-spacing:-.9px; margin:18px 0 6px; }
#     .login-subtitle { color:#688198; font-size:14px; line-height:1.6; margin-bottom:25px; }
#     [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:26px; border-bottom:1px solid rgba(68,117,145,.20); }
#     [data-testid="stTabs"] button[role="tab"] { color:#31536e; font-weight:620; padding:10px 2px; }
#     [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#078e96; }
#     [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#13b8b0; height:3px; }
#     [data-testid="stTabs"] [data-baseweb="tab-border"] { background:transparent; }
#     [data-testid="stForm"] { background:rgba(255,255,255,.94); padding:26px 24px 23px; border:1px solid #dce9ef; border-radius:18px;
#         box-shadow:0 16px 42px rgba(31,76,102,.09); }
#     [data-testid="stTextInput"] label { color:#193e59; font-size:13px; font-weight:650; }
#     [data-testid="stTextInput"] input { background:#f0f6fa; border:1px solid #d9e6ed; border-radius:10px; min-height:47px; color:#183d59; }
#     [data-testid="stTextInput"] input:focus { border-color:#1aaeb0; box-shadow:0 0 0 2px rgba(26,174,176,.12); }
#     [data-testid="stFormSubmitButton"] button { min-height:48px; border-radius:10px; color:white; border:0; font-weight:700;
#         background:linear-gradient(100deg,#087f91,#12aaa5); box-shadow:0 6px 15px rgba(9,137,148,.17); }
#     [data-testid="stFormSubmitButton"] button:hover { color:white; border:0; background:linear-gradient(100deg,#066d7e,#0b928f); }
#     .privacy-note { display:flex; align-items:center; justify-content:center; gap:7px; color:#71899d; text-align:center;
#         font-size:12px; line-height:1.55; margin-top:23px; }
#     .privacy-symbol { color:#168b91; font-size:14px; }

#     @media (max-width: 900px) {
#         .block-container { padding:1.1rem 1rem; }
#         .med-hero { min-height:auto; padding:29px 23px 24px; }
#         .brand-row { margin-bottom:30px; }
#         .brand-name { font-size:30px; }
#         .hero-tagline { font-size:25px; }
#         .login-shell { padding:12px 0 0; }
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     left, right = st.columns([1.06, 0.94], gap="large")

#     with left:
#         st.markdown("""
#         <section class="med-hero">
#           <div class="brand-row">
#             <svg class="brand-mark" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
#               <rect x="17" y="2" width="30" height="60" rx="10" fill="#20C8C7"/>
#               <rect x="2" y="17" width="60" height="30" rx="10" fill="#20C8C7"/>
#               <path d="M5 33h13l5-11 7 23 7-17 5 5h17" stroke="#06334E" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
#             </svg>
#             <div class="brand-name">SafeMed<span>Twin</span></div>
#           </div>
#           <div class="hero-kicker">Intelligent healthcare technology</div>
#           <h1 class="hero-tagline">Understand health.<br><span>Anticipate risk.</span> Support better decisions.</h1>
#           <div class="hero-rule"></div>
#           <p class="hero-description">A patient-centric medical decision-support platform that builds an individual health profile, evaluates potential disease risks using machine learning, and provides AI-assisted insights through an integrated safety-validation layer.</p>
#           <div class="capability-heading">Platform capabilities</div>
#           <div class="feature-list">
#             <div class="med-feature"><div class="med-feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="7" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></svg></div><div><div class="med-feature-title">Patient Digital Twin</div><div class="med-feature-desc">A consolidated representation of individual health profiles.</div></div></div>
#             <div class="med-feature"><div class="med-feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/><path d="M12 8v8m-4-4h8"/></svg></div><div><div class="med-feature-title">Multi-Disease Risk Assessment</div><div class="med-feature-desc">Machine-learning evaluation across multiple health conditions.</div></div></div>
#             <div class="med-feature"><div class="med-feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3a7 7 0 0 0-7 7v1a4 4 0 0 0 0 8h2l2-3V8l-2-2m7-3a7 7 0 0 1 7 7v1a4 4 0 0 1 0 8h-2l-2-3V8l2-2M9 9h6m-6 4h6m-3-8v14"/></svg></div><div><div class="med-feature-title">AI-Assisted Recommendations</div><div class="med-feature-desc">Context-aware insights generated using language models.</div></div></div>
#             <div class="med-feature"><div class="med-feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/><path d="m8.5 12 2.3 2.3 4.8-5"/></svg></div><div><div class="med-feature-title">MedGuard Safety Layer</div><div class="med-feature-desc">Safety validation of AI-generated recommendations.</div></div></div>
#           </div>
#           <div class="hero-footer"><span class="footer-mark">+</span><span>Designed to support informed healthcare decisions —<br>not replace professional medical judgement.</span></div>
#         </section>
#         """, unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="login-shell"><div class="login-welcome">Welcome!</div><div class="login-subtitle">Sign in to access your personal health dashboard.</div></div>', unsafe_allow_html=True)

#         tab_login, tab_register = st.tabs(["Log in", "Create account"])

#         with tab_login:
#             with st.form("login_form"):
#                 u = st.text_input("Username", key="login_u", placeholder="Enter your username")
#                 p = st.text_input("Password", type="password", key="login_p", placeholder="Enter your password")
#                 do_login = st.form_submit_button("Log in", type="primary", use_container_width=True)
#             if do_login:
#                 ok, uid, msg = auth.authenticate(u, p)
#                 if ok:
#                     st.session_state["user_id"] = uid
#                     st.session_state["username"] = u.strip().lower()
#                     st.session_state.pop("login_u", None)
#                     st.session_state.pop("login_p", None)
#                     st.rerun()
#                 else:
#                     st.error(msg)

#         with tab_register:
#             with st.form("register_form"):
#                 u2 = st.text_input("Choose a username", key="reg_u", placeholder="Create a username")
#                 p2 = st.text_input("Choose a password (min 6 characters)", type="password", key="reg_p", placeholder="Create a password")
#                 do_register = st.form_submit_button("Create account", type="primary", use_container_width=True)
#             if do_register:
#                 ok, msg = auth.register_user(u2, p2)
#                 if ok:
#                     st.success(msg + " You can log in now.")
#                 else:
#                     st.error(msg)

#         st.markdown('<div class="privacy-note"><span class="privacy-symbol">▣</span><span>Your health assessments and chat history remain private to your account.</span></div>', unsafe_allow_html=True)

#     st.stop()

# user_id = st.session_state["user_id"]
# username = st.session_state["username"]

# # =========================================================== SIDEBAR ======
# st.sidebar.title("🩺 SafeMedTwin")
# st.sidebar.caption(f"Logged in as **{username}**")
# if st.sidebar.button("Log out", key="btn_logout"):
#     st.session_state.pop("user_id", None)
#     st.session_state.pop("username", None)
#     st.rerun()

# st.sidebar.markdown("---")
# disease = st.sidebar.selectbox("Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="sb_disease")
# bundle = load_bundle(disease)
# feature_cols = bundle["feature_cols"]
# st.sidebar.caption(f"Model AUC: **{bundle['metrics']['roc_auc']:.3f}** · trained on {bundle['metrics']['n_train']} patients")

# user_history = auth.get_latest_prediction_per_disease(user_id)

# # =============================================================== TABS =====
# tab_assess, tab_twin, tab_diet, tab_chat, tab_scan, tab_history = st.tabs(
#     ["🩺 Risk Assessment", "🫀 3D Twin & What-If", "🥗 Diet & Lifestyle", "🤖 Chatbot",
#      "📄 Report Scanner", "📈 History"]
# )

# # ------------------------------------------------------- Risk Assessment --
# with tab_assess:
#     st.header(f"{DISEASE_LABELS[disease]} Risk Assessment")

#     prefill = st.session_state.get("scan_prefill", {})
#     with st.form("patient_form"):
#         cols = st.columns(3)
#         values = {}
#         for i, feat in enumerate(feature_cols):
#             col = cols[i % 3]
#             label = feat.replace("_", " ").title()
#             default = prefill.get(feat, DEFAULTS.get(feat, 0))
#             if feat in BINARY_FEATURES:
#                 values[feat] = col.selectbox(label, [0, 1], index=int(default), key=f"a_{feat}")
#             elif feat in ("pregnancies", "albumin_grade", "age"):
#                 values[feat] = col.number_input(label, min_value=0, value=int(default), step=1, key=f"a_{feat}")
#             else:
#                 values[feat] = col.number_input(label, value=float(default), step=0.5, key=f"a_{feat}")
#         submitted = st.form_submit_button("Run Prediction", type="primary")

#     if submitted:
#         row = pd.DataFrame([values])[feature_cols]
#         probability = float(bundle["model"].predict_proba(row)[0, 1])
#         band = compute_risk_band(probability)
#         safety = validate_prediction(disease, values, probability)
#         contributions = explain_prediction(bundle, row, top_k=5)
#         explanation = explanation_to_sentence(disease, contributions)

#         previous_band = user_history.get(disease, {}).get("risk_band")
#         alerts = evaluate_alerts(disease, values, band, previous_band=previous_band)

#         auth.save_prediction(user_id, disease, probability, band, values)
#         user_history[disease] = {"probability": probability, "risk_band": band, "features": values}

#         st.session_state["last_result"] = {
#             "probability": probability, "risk_band": band, "safety": safety,
#             "contributions": contributions, "explanation": explanation, "alerts": alerts,
#             "baseline": values, "disease": disease,
#         }
#         st.session_state.pop("scan_prefill", None)

#     result = st.session_state.get("last_result")
#     if result and result["disease"] == disease:
#         band = result["risk_band"]
#         color = RISK_COLORS[band]
#         c1, c2 = st.columns([1, 2])
#         with c1:
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"},
#                 title={"text": f"{band.upper()} RISK"},
#                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
#                        "steps": [{"range": [0, 33], "color": "#EAF5EE"}, {"range": [33, 66], "color": "#FCF3D9"},
#                                  {"range": [66, 100], "color": "#FBEAE8"}]},
#             ))
#             fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
#             st.plotly_chart(fig, width='stretch')
#         with c2:
#             st.subheader("Explanation (SHAP)")
#             st.write(result["explanation"])
#             contrib_df = pd.DataFrame(result["contributions"])
#             fig2 = px.bar(contrib_df.sort_values("shap_value"), x="shap_value", y="feature", orientation="h",
#                            color="shap_value", color_continuous_scale=["#2E8B57", "#C0392B"],
#                            labels={"shap_value": "SHAP contribution", "feature": ""})
#             fig2.update_layout(height=280, showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
#             st.plotly_chart(fig2, width='stretch')

#         if result["safety"].warnings:
#             st.warning("MedGuard flags:\n" + "\n".join(f"- {w}" for w in result["safety"].warnings))
#         else:
#             st.info("✅ MedGuard: no safety flags on this prediction.")

#         if result["alerts"]:
#             for a in result["alerts"]:
#                 (st.error if a.severity == "critical" else st.warning)(f"🔔 {a.message}")

#     elif not result:
#         st.caption("Fill in the form above and run a prediction to see results.")

# # ---------------------------------------------------------------- 3D Twin --
# with tab_twin:
#     st.header("🫀 3D Digital Twin & What-If Simulation")
#     unassessed = [DISEASE_LABELS[d] for d in DISEASES if d not in user_history]
#     if unassessed:
#         st.caption(f"Not yet assessed (shown gray): {', '.join(unassessed)}")
#     organ_data = organ_data_from_predictions(user_history)
#     st.iframe(generate_body_twin_html(organ_data), height=540)

#     st.markdown("---")
#     st.subheader("What-If Simulation")
#     if disease not in user_history:
#         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable simulation.")
#     else:
#         baseline = user_history[disease]["features"]
#         sweep_feature = st.selectbox("Feature to vary", feature_cols, key="sweep_feat")
#         current_val = float(baseline.get(sweep_feature, 0))
#         c1, c2 = st.columns(2)
#         lo = c1.number_input("Range low", value=max(0.0, current_val * 0.5), key="sweep_lo")
#         hi = c2.number_input("Range high", value=current_val * 1.5 if current_val > 0 else 10.0, key="sweep_hi")
#         if st.button("Run sweep", key="btn_sweep"):
#             try:
#                 values_range = [lo + (hi - lo) * i / 10 for i in range(11)]
#                 sweep_df = pd.DataFrame(multi_scenario_sweep(bundle, baseline, sweep_feature, values_range))
#                 fig3 = px.line(sweep_df, x=sweep_feature, y="probability", markers=True,
#                                 title=f"Risk sensitivity to {sweep_feature.replace('_', ' ')}")
#                 fig3.update_yaxes(range=[0, 1], tickformat=".0%")
#                 st.plotly_chart(fig3, width='stretch')
#             except Exception as e:
#                 st.error(f"Simulation failed: {e}")

#     # ---------------------------------------------- LLM vs SLM comparison --
#     st.markdown("---")
#     st.subheader("🤖 LLM vs SLM Recommendation Comparison")
#     if disease not in user_history:
#         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to enable this comparison.")
#     else:
#         band = user_history[disease]["risk_band"]
#         row = pd.DataFrame([user_history[disease]["features"]])[feature_cols]
#         top_factor = explain_prediction(bundle, row, top_k=1)[0]["feature"]

#         cbtn1, cbtn2 = st.columns(2)
#         if cbtn1.button("Generate & compare", key="btn_compare"):
#             comp = compare_llm_slm(disease, band, top_factor)
#             cc1, cc2 = st.columns(2)
#             for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
#                 r = comp[key]
#                 with col:
#                     st.markdown(f"**{label}**")
#                     st.write(r["text"])
#                     m1, m2, m3 = st.columns(3)
#                     m1.metric("Latency", f"{r['latency_ms']:.0f} ms")
#                     m2.metric("Accuracy", f"{r['accuracy_score']:.0%}")
#                     m3.metric("Completeness", f"{r['completeness_score']:.0%}")
#                     if r["hallucination_flags"]:
#                         st.error("⚠️ " + "; ".join(r["hallucination_flags"]))

#         if cbtn2.button("Run consistency test (3 runs each)", key="btn_consistency"):
#             with st.spinner("Running each model 3 times..."):
#                 ctest = consistency_test(disease, band, top_factor, n_runs=3)
#             cc1, cc2 = st.columns(2)
#             for col, key, label in [(cc1, "slm", "SLM"), (cc2, "llm", "LLM")]:
#                 r = ctest[key]
#                 with col:
#                     st.markdown(f"**{label} consistency score: {r['consistency_score']:.0%}**")
#                     st.caption("1.0 = identical outputs every run; lower = more variance run-to-run")
#                     for i, out in enumerate(r["outputs"], 1):
#                         with st.expander(f"Run {i}"):
#                             st.write(out)

#     # -------------------------------------------- DistilBERT classifier --
#     st.markdown("---")
#     st.subheader("🔤 DistilBERT Text Classifier (complementary to the SLM)")
#     st.caption(
#         "TinyLlama (the SLM above) *generates* recommendations — a different job from "
#         "classification. DistilBERT here is fine-tuned to classify risk band directly "
#         "from a text description of the patient, showing what a small encoder model is "
#         "actually built for: fast, cheap classification rather than free text generation."
#     )
#     if not distilbert_available():
#         st.info(
#             "No fine-tuned DistilBERT classifier found yet. Train one (takes a few minutes "
#             "on CPU, faster with a GPU):\n\n"
#             "```\npip install -r requirements-slm.txt\n"
#             "python models/train_distilbert_classifier.py\n```"
#         )
#     elif disease not in user_history:
#         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first to classify this patient.")
#     else:
#         if st.button("Classify with DistilBERT", key="btn_distilbert"):
#             row = pd.DataFrame([user_history[disease]["features"]])[feature_cols].iloc[0]
#             try:
#                 result = classify_patient(disease, row)
#                 st.code(result["text"], language=None)
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric("DistilBERT says", result["label"])
#                 c2.metric("Confidence", f"{result['confidence']:.0%}")
#                 c3.metric("Latency", f"{result['latency_ms']:.1f} ms")
#                 xgb_band = user_history[disease]["risk_band"]
#                 if result["label"] == xgb_band:
#                     st.success(f"✅ Agrees with the XGBoost model's prediction ({xgb_band})")
#                 else:
#                     st.warning(f"⚠️ Disagrees with the XGBoost model ({xgb_band}) — worth noting "
#                                f"in your report as a genuine model-comparison finding, not a bug.")
#                 with st.expander("Full probability distribution"):
#                     st.json(result["probabilities"])
#             except Exception as e:
#                 st.error(f"Classification failed: {e}")

# # ------------------------------------------------------- Diet & Lifestyle --
# with tab_diet:
#     st.header("🥗 Diet & Lifestyle Recommendations")
#     if disease not in user_history:
#         st.caption(f"Run a {DISEASE_LABELS[disease]} prediction first for a personalized plan.")
#         base_plan = get_rule_based_plan(disease, "unknown")
#         st.write(base_plan["framing"])
#     else:
#         band = user_history[disease]["risk_band"]
#         top_factors = [c["feature"] for c in explain_prediction(
#             bundle, pd.DataFrame([user_history[disease]["features"]])[feature_cols], top_k=3
#         )]
#         if st.button("Generate personalized plan", key="btn_diet"):
#             plan = get_personalized_plan(disease, band, top_factors)
#             st.session_state["diet_plan"] = plan
#         base_plan = st.session_state.get("diet_plan") or get_rule_based_plan(disease, band)

#     st.info(base_plan["framing"])
#     if base_plan.get("personalized_note"):
#         st.markdown(f"**For you:** {base_plan['personalized_note']}")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.subheader("Diet")
#         for item in base_plan["diet"]:
#             st.markdown(f"- {item}")
#     with c2:
#         st.subheader("Lifestyle")
#         for item in base_plan["lifestyle"]:
#             st.markdown(f"- {item}")
#     st.caption("General educational guidance only — not a substitute for advice from your physician or a registered dietitian.")

# # ------------------------------------------------------------------ Chat --
# with tab_chat:
#     st.header("🤖 SafeMedTwin Assistant")
#     st.caption("Ask about your assessments — this assistant can see your latest results but never diagnoses.")

#     for msg in auth.get_chat_history(user_id):
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])

#     user_msg = st.chat_input("Ask a question about your health data...")
#     if user_msg:
#         with st.chat_message("user"):
#             st.write(user_msg)
#         auth.save_chat_message(user_id, "user", user_msg)

#         history_for_api = [{"role": m["role"], "content": m["content"]} for m in auth.get_chat_history(user_id, limit=10)]
#         reply = chatbot_reply(user_msg, user_history, conversation_history=history_for_api[:-1])
#         with st.chat_message("assistant"):
#             st.write(reply)
#         auth.save_chat_message(user_id, "assistant", reply)

# # ------------------------------------------------------- Report Scanner --
# with tab_scan:
#     st.header("📄 Medical Report Scanner")
#     st.caption("Upload a photo/screenshot OR a PDF of a lab report. Extracted values can prefill "
#                "the assessment form above — nothing is auto-submitted.")

#     uploaded = st.file_uploader("Upload report (image or PDF)", type=["png", "jpg", "jpeg", "pdf"], key="report_upload")
#     if uploaded and st.button("Scan report", key="btn_scan"):
#         is_pdf = uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")
#         with st.spinner("Reading report..."):
#             result = scan_report_pdf(uploaded.getvalue()) if is_pdf else scan_report_image(uploaded.getvalue())

#         if "error" in result:
#             st.error(result["error"])
#             if result.get("raw"):
#                 with st.expander("Show raw model response (for debugging)"):
#                     st.code(result["raw"])
#         elif not result.get("legible", True):
#             st.warning(result.get("summary", "Could not read this report."))
#         else:
#             st.success(f"Report read successfully ({result.get('read_method', 'image')}).")
#             st.write(result["summary"])
#             if result.get("safety_flags"):
#                 st.caption("⚠️ " + "; ".join(result["safety_flags"]))
#             extracted = result.get("extracted_values", {})
#             if extracted:
#                 st.subheader("Extracted values")
#                 st.json(extracted)
#                 if st.button("Use these values to prefill the assessment form", key="btn_prefill"):
#                     st.session_state["scan_prefill"] = {k: v for k, v in extracted.items() if k in feature_cols}
#                     st.success("Prefilled — go to the Risk Assessment tab and run a prediction.")

# # ------------------------------------------------------------- History --
# with tab_history:
#     st.header("📈 Your Assessment History")
#     all_predictions = auth.get_user_predictions(user_id)
#     if not all_predictions:
#         st.caption("No assessments recorded yet — run a prediction in the Risk Assessment tab first.")
#     else:
#         hist_disease = st.selectbox(
#             "Disease", DISEASES, format_func=lambda d: DISEASE_LABELS[d], key="history_disease"
#         )
#         disease_rows = [r for r in all_predictions if r["disease"] == hist_disease]
#         if not disease_rows:
#             st.caption(f"No {DISEASE_LABELS[hist_disease]} assessments yet for your account.")
#         else:
#             hist_df = pd.DataFrame(disease_rows)[["created_at", "probability", "risk_band"]]
#             hist_df["created_at"] = pd.to_datetime(hist_df["created_at"])
#             fig5 = px.line(hist_df, x="created_at", y="probability", markers=True,
#                             title=f"{DISEASE_LABELS[hist_disease]} risk over time")
#             fig5.update_yaxes(range=[0, 1], tickformat=".0%")
#             st.plotly_chart(fig5, width='stretch')

#             st.subheader("All assessments")
#             display_df = hist_df.rename(columns={
#                 "created_at": "Date", "probability": "Risk %", "risk_band": "Risk Band"
#             })
#             display_df["Risk %"] = (display_df["Risk %"] * 100).round(1)
#             st.dataframe(display_df.sort_values("Date", ascending=False), width='stretch', hide_index=True)

#         st.markdown("---")
#         st.subheader("Across all diseases")
#         summary_rows = []
#         for d in DISEASES:
#             d_rows = [r for r in all_predictions if r["disease"] == d]
#             if d_rows:
#                 latest = max(d_rows, key=lambda r: r["created_at"])
#                 summary_rows.append({
#                     "Disease": DISEASE_LABELS[d], "Latest Risk %": round(latest["probability"] * 100, 1),
#                     "Risk Band": latest["risk_band"], "Assessments": len(d_rows),
#                     "Last Assessed": latest["created_at"],
#                 })
#         if summary_rows:
#             st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)




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
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 4% 10%, rgba(35,181,194,.13), transparent 26%),
          radial-gradient(circle at 96% 90%, rgba(75,132,207,.12), transparent 28%),
          linear-gradient(135deg,#f4fbfd 0%,#ffffff 52%,#f1f7fc 100%);
    }
    [data-testid="stHeader"] {background:transparent;}
    .block-container {max-width:1380px;padding-top:2.3rem;padding-bottom:2.5rem;}
    .top-brand {display:flex;align-items:center;gap:12px;margin:0 0 34px 2px;}
    .top-brand-mark {width:43px;height:43px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#087f91,#18b8b0);box-shadow:0 7px 18px rgba(10,143,153,.2);}
    .top-brand-mark svg {width:26px;height:26px;}
    .top-brand-name {font-size:27px;font-weight:800;letter-spacing:-.8px;color:#123b59;line-height:1;}
    .top-brand-name span {color:#079b9d;}
    .top-brand-note {margin-left:auto;color:#6e8799;font-size:11px;font-weight:750;letter-spacing:1.5px;text-transform:uppercase;}

    .intro-panel {padding:25px 28px 23px 4px;}
    .eyebrow {display:inline-flex;align-items:center;gap:8px;color:#078c95;font-size:11px;font-weight:800;letter-spacing:1.7px;text-transform:uppercase;margin-bottom:17px;}
    .eyebrow:before {content:"";width:25px;height:2px;background:#19b8b0;border-radius:3px;}
    .intro-title {font-size:43px;line-height:1.13;letter-spacing:-1.6px;font-weight:820;color:#123652;margin:0 0 17px;max-width:650px;}
    .intro-title span {color:#07999e;}
    .intro-copy {font-size:15px;line-height:1.85;color:#587187;max-width:650px;margin:0 0 27px;}
    .section-label {font-size:10px;letter-spacing:1.8px;font-weight:800;color:#7b95a6;text-transform:uppercase;margin:0 0 13px;}
    .cap-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:690px;}
    .cap-item {display:flex;align-items:flex-start;gap:12px;padding:15px 14px;border:1px solid #e1edf2;border-radius:14px;background:rgba(255,255,255,.82);box-shadow:0 5px 16px rgba(25,70,95,.035);min-height:91px;}
    .cap-icon {width:37px;height:37px;flex:0 0 37px;display:grid;place-items:center;border-radius:11px;background:#e4f7f6;color:#078f94;}
    .cap-item:nth-child(2) .cap-icon {background:#eaf1ff;color:#477bc4;}
    .cap-item:nth-child(3) .cap-icon {background:#eaf5fc;color:#3986ad;}
    .cap-item:nth-child(4) .cap-icon {background:#e8f7f0;color:#278b69;}
    .cap-icon svg {width:20px;height:20px;}
    .cap-title {font-size:13px;font-weight:760;color:#183e59;line-height:1.35;margin:1px 0 4px;}
    .cap-desc {font-size:11px;line-height:1.5;color:#71899a;}
    .clinical-note {display:flex;align-items:center;gap:10px;margin-top:22px;color:#6e8798;font-size:11.5px;line-height:1.5;}
    .clinical-note span {width:25px;height:25px;display:grid;place-items:center;border-radius:50%;background:#e4f6f5;color:#078e94;font-weight:800;flex:0 0 25px;}

    .auth-wrap {padding:13px 0 0 18px;}
    .auth-kicker {color:#07979b;font-size:10px;font-weight:800;letter-spacing:1.8px;text-transform:uppercase;margin-bottom:10px;}
    .auth-title {font-size:35px;font-weight:810;letter-spacing:-1px;color:#123652;margin:0 0 7px;}
    .auth-subtitle {font-size:13px;color:#72899b;line-height:1.65;margin-bottom:23px;}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {gap:25px;border-bottom:1px solid #dce8ee;}
    [data-testid="stTabs"] button[role="tab"] {color:#567086;font-weight:650;padding:10px 2px;}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:#078f96;}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:linear-gradient(90deg,#0ba6a6,#4b91d0);height:3px;}
    [data-testid="stTabs"] [data-baseweb="tab-border"] {background:transparent;}
    [data-testid="stForm"] {background:rgba(255,255,255,.96);padding:25px 23px 22px;border:1px solid #dfebf0;border-radius:17px;box-shadow:0 16px 42px rgba(25,70,95,.09);}
    [data-testid="stTextInput"] label {color:#24465f;font-size:12px;font-weight:680;}
    [data-testid="stTextInput"] input {background:#f4f8fb;border:1px solid #dce8ef;border-radius:9px;min-height:46px;color:#173d58;}
    [data-testid="stTextInput"] input:focus {border-color:#17aeb0;box-shadow:0 0 0 2px rgba(23,174,176,.12);}
    [data-testid="stFormSubmitButton"] button {min-height:47px;border-radius:9px;color:white;border:0;font-weight:740;background:linear-gradient(100deg,#087f91,#0d9fa5 58%,#20b6a6);box-shadow:0 7px 16px rgba(9,137,148,.17);}
    [data-testid="stFormSubmitButton"] button:hover {color:white;border:0;background:linear-gradient(100deg,#066f80,#078d98,#139e92);}
    .privacy-note {display:flex;justify-content:center;align-items:center;gap:7px;color:#7890a0;text-align:center;font-size:11px;line-height:1.6;margin-top:21px;}
    .privacy-note span {color:#07969b;}
    @media(max-width:900px){.block-container{padding:1.2rem 1rem;}.top-brand-note{display:none;}.intro-panel{padding:8px 0 20px;}.intro-title{font-size:34px;}.cap-grid{grid-template-columns:1fr;}.auth-wrap{padding:8px 0 0;}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="top-brand">
      <div class="top-brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-5 4 10 2-5h6"/><path d="M12 2a10 10 0 1 0 10 10" opacity=".0"/></svg></div>
      <div class="top-brand-name">SafeMed<span>Twin</span></div>
      <div class="top-brand-note">Intelligent healthcare · Safety-aware AI</div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown("""
        <section class="intro-panel">
          <div class="eyebrow">Healthcare intelligence, made clearer</div>
          <h1 class="intro-title">Understand your health.<br><span>Make informed decisions.</span></h1>
          <p class="intro-copy">SafeMedTwin brings health information, machine-learning risk assessment and AI-assisted guidance into one patient-focused platform. It helps make complex health insights easier to understand, with an integrated safety layer designed to review AI-generated recommendations.</p>
          <div class="section-label">What the platform brings together</div>
          <div class="cap-grid">
            <div class="cap-item"><div class="cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="7" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/></svg></div><div><div class="cap-title">Patient Health Profile</div><div class="cap-desc">A consolidated view of personal health information and assessments.</div></div></div>
            <div class="cap-item"><div class="cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/><path d="M12 8v8m-4-4h8"/></svg></div><div><div class="cap-title">Risk Assessment</div><div class="cap-desc">Machine-learning evaluation across multiple health conditions.</div></div></div>
            <div class="cap-item"><div class="cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19V5m0 14h16M8 15v-4m4 4V7m4 8V9m4 6V4"/></svg></div><div><div class="cap-title">AI-Assisted Health Insights</div><div class="cap-desc">Context-aware information to support better-informed conversations.</div></div></div>
            <div class="cap-item"><div class="cap-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/><path d="m8.5 12 2.3 2.3 4.8-5"/></svg></div><div><div class="cap-title">MedGuard Safety Layer</div><div class="cap-desc">Safety validation for AI-generated recommendations and alerts.</div></div></div>
          </div>
          <div class="clinical-note"><span>✓</span>Designed to support informed healthcare decisions — not replace professional medical judgement.</div>
        </section>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-wrap"><div class="auth-kicker">Secure personal access</div><div class="auth-title">Welcome!</div><div class="auth-subtitle">Sign in to continue to your personal health dashboard.</div></div>', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["Log in", "Create account"])
        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Username", key="login_u", placeholder="Enter your username")
                p = st.text_input("Password", type="password", key="login_p", placeholder="Enter your password")
                do_login = st.form_submit_button("Log in", type="primary", use_container_width=True)
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
                u2 = st.text_input("Choose a username", key="reg_u", placeholder="Create a username")
                p2 = st.text_input("Choose a password (min 6 characters)", type="password", key="reg_p", placeholder="Create a password")
                do_register = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if do_register:
                ok, msg = auth.register_user(u2, p2)
                if ok:
                    st.success(msg + " You can log in now.")
                else:
                    st.error(msg)
        st.markdown('<div class="privacy-note"><span>▣</span> Your health assessments and chat history remain private to your account.</div>', unsafe_allow_html=True)
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
                    accuracy = r.get("accuracy_score")

                    m2.metric("Accuracy",f"{accuracy:.0%}" if accuracy is not None else "N/A"
)
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
