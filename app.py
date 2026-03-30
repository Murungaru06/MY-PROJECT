"""
app.py  ─  Student Performance Predictor
Streamlit app that loads a pre-trained ANN and predicts student risk level.

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── Background gradient ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}
.card h3 { margin-top: 0; color: #c9b8ff; font-weight: 600; }

/* ── Result cards ── */
.result-at-risk {
    background: linear-gradient(135deg, rgba(231,76,60,0.25), rgba(192,57,43,0.15));
    border: 2px solid #e74c3c;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.result-average {
    background: linear-gradient(135deg, rgba(243,156,18,0.25), rgba(230,126,34,0.15));
    border: 2px solid #f39c12;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, rgba(39,174,96,0.25), rgba(46,204,113,0.15));
    border: 2px solid #27ae60;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
}
.result-title { font-size: 1.1rem; color: #ccc; margin-bottom: 8px; font-weight: 500; }
.result-label { font-size: 2.4rem; font-weight: 700; margin: 10px 0; }
.result-emoji { font-size: 3rem; }

/* ── Probability bar ── */
.prob-bar-wrap { margin: 6px 0; }
.prob-label { font-size: 0.85rem; color: #ccc; margin-bottom: 3px; }

/* ── Section headers ── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(167,139,250,0.3);
}

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124,58,237,0.6);
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-val { font-size: 1.7rem; font-weight: 700; color: #a78bfa; }
.metric-lbl { font-size: 0.78rem; color: #aaa; margin-top: 4px; }

/* ── Tips ── */
.tip-box {
    background: rgba(99,102,241,0.15);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #d4d4ff;
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model & artefacts ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    required = [
        'student_performance_ann.keras',
        'scaler.pkl',
        'label_encoders.pkl',
        'feature_names.pkl',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, missing

    from tensorflow.keras.models import load_model
    model          = load_model('student_performance_ann.keras')
    scaler         = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names  = joblib.load('feature_names.pkl')
    return model, scaler, label_encoders, feature_names, []


model, scaler, label_encoders, feature_names, missing_files = load_artifacts()


# ── Helper: build feature vector ─────────────────────────────────
def build_features(inputs: dict) -> np.ndarray:
    """
    inputs: dict of raw (un-encoded) user values keyed by column name.
    Returns a 1-D float32 array ready for the model.
    """
    row = {}
    for col in feature_names:
        val = inputs[col]
        if col in label_encoders:
            val = label_encoders[col].transform([str(val)])[0]
        row[col] = float(val)
    return np.array([list(row.values())], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR  ─  navigation + about
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔮 Predict", "ℹ️ About Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#888; line-height:1.6'>
    <b>Model:</b> ANN (Deep Learning)<br>
    <b>Layers:</b> 256 → 128 → 64 → 32 → 3<br>
    <b>Dataset:</b> UCI Student Performance<br>
    <b>Target:</b> Final Grade (G3)<br>
    <b>Classes:</b> At Risk · Average · High Performer
    </div>
    """, unsafe_allow_html=True)

    if missing_files:
        st.error(f"Missing files:\n" + "\n".join(f"• {f}" for f in missing_files))
        st.info("Run `python train_model.py` first to generate model files.")


# ═══════════════════════════════════════════════════════════════════
#  MISSING FILES GUARD
# ═══════════════════════════════════════════════════════════════════
if missing_files:
    st.markdown("# 🎓 Student Performance Predictor")
    st.error("⚠️  Model files not found. Please train the model first.")
    st.code("python train_model.py", language="bash")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════
if "🔮 Predict" in page:

    st.markdown("# 🎓 Student Performance Predictor")
    st.markdown(
        "<p style='color:#aaa;font-size:1rem;margin-top:-10px;'>"
        "Fill in the student profile below and click <b>Predict</b> to assess their risk level.</p>",
        unsafe_allow_html=True
    )

    # ── Two-column layout ─────────────────────────────────────────
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:

        # ── SECTION 1: School & Demographics ──────────────────────
        st.markdown('<div class="section-header">📋 School & Demographics</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            school  = st.selectbox("School", ["GP", "MS"],
                                   help="GP = Gabriel Pereira · MS = Mousinho da Silveira")
            sex     = st.selectbox("Sex", ["F", "M"])
            age     = st.slider("Age", 15, 22, 17)
            address = st.selectbox("Address", ["U", "R"],
                                   help="U = Urban · R = Rural")
        with c2:
            famsize = st.selectbox("Family Size", ["GT3", "LE3"],
                                   help="GT3 = >3 members · LE3 = ≤3 members")
            Pstatus = st.selectbox("Parents' Status", ["T", "A"],
                                   help="T = Together · A = Apart")
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])
            reason   = st.selectbox("School Choice Reason",
                                    ["course", "home", "reputation", "other"])

        # ── SECTION 2: Education ──────────────────────────────────
        st.markdown('<div class="section-header">🎒 Education & Family</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            Medu = st.select_slider("Mother's Education",
                                    options=[0, 1, 2, 3, 4],
                                    value=2,
                                    format_func=lambda x: ["None","Primary","5th-9th","Secondary","Higher"][x])
            Fedu = st.select_slider("Father's Education",
                                    options=[0, 1, 2, 3, 4],
                                    value=2,
                                    format_func=lambda x: ["None","Primary","5th-9th","Secondary","Higher"][x])
            Mjob = st.selectbox("Mother's Job",
                                ["at_home", "health", "other", "services", "teacher"])
        with c4:
            Fjob       = st.selectbox("Father's Job",
                                      ["at_home", "health", "other", "services", "teacher"])
            traveltime = st.select_slider("Travel Time (to school)",
                                          options=[1, 2, 3, 4],
                                          format_func=lambda x: ["<15 min","15-30 min","30-60 min",">1 hr"][x-1])
            studytime  = st.select_slider("Weekly Study Time",
                                          options=[1, 2, 3, 4],
                                          format_func=lambda x: ["<2 hrs","2-5 hrs","5-10 hrs",">10 hrs"][x-1])

        # ── SECTION 3: Support ────────────────────────────────────
        st.markdown('<div class="section-header">🏠 Support & Activities</div>',
                    unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            schoolsup  = st.radio("School Support",  ["yes", "no"], horizontal=True)
            famsup     = st.radio("Family Support",   ["yes", "no"], horizontal=True)
            paid       = st.radio("Extra Paid Classes", ["yes", "no"], horizontal=True)
            activities = st.radio("Extracurricular",  ["yes", "no"], horizontal=True)
        with c6:
            nursery  = st.radio("Attended Nursery",   ["yes", "no"], horizontal=True)
            higher   = st.radio("Wants Higher Edu",   ["yes", "no"], horizontal=True)
            internet = st.radio("Internet at Home",   ["yes", "no"], horizontal=True)
            romantic = st.radio("In a Relationship",  ["yes", "no"], horizontal=True)

        # ── SECTION 4: Social & Health ────────────────────────────
        st.markdown('<div class="section-header">🌡️ Social, Health & Grades</div>',
                    unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7:
            famrel   = st.slider("Family Relationship Quality", 1, 5, 3,
                                 help="1 = Very Bad · 5 = Excellent")
            freetime = st.slider("Free Time After School",      1, 5, 3)
            goout    = st.slider("Going Out with Friends",       1, 5, 3)
            health   = st.slider("Current Health Status",        1, 5, 3)
        with c8:
            Dalc     = st.slider("Workday Alcohol Consumption",  1, 5, 1,
                                 help="1 = Very Low · 5 = Very High")
            Walc     = st.slider("Weekend Alcohol Consumption",  1, 5, 1)
            failures = st.slider("Past Class Failures",          0, 3, 0)
            absences = st.slider("Number of Absences",           0, 93, 4)

        # ── SECTION 5: Mid-term grades ────────────────────────────
        st.markdown('<div class="section-header">📊 Mid-Term Grades (0 – 20)</div>',
                    unsafe_allow_html=True)
        cg1, cg2 = st.columns(2)
        with cg1:
            G1 = st.slider("G1 — First Period Grade",  0, 20, 10)
        with cg2:
            G2 = st.slider("G2 — Second Period Grade", 0, 20, 10)

        # ── Predict button ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Performance", use_container_width=True)

    # ── Result panel ──────────────────────────────────────────────
    with col_result:
        st.markdown("<br>", unsafe_allow_html=True)

        if predict_clicked:
            inputs = dict(
                school=school, sex=sex, age=age, address=address,
                famsize=famsize, Pstatus=Pstatus, Medu=Medu, Fedu=Fedu,
                Mjob=Mjob, Fjob=Fjob, reason=reason, guardian=guardian,
                traveltime=traveltime, studytime=studytime, failures=failures,
                schoolsup=schoolsup, famsup=famsup, paid=paid,
                activities=activities, nursery=nursery, higher=higher,
                internet=internet, romantic=romantic, famrel=famrel,
                freetime=freetime, goout=goout, Dalc=Dalc, Walc=Walc,
                health=health, absences=absences, G1=G1, G2=G2
            )

            X_input = build_features(inputs)
            X_scaled = scaler.transform(X_input)

            with st.spinner("Analysing student profile …"):
                time.sleep(0.5)   # small UX delay for feel
                probs = model.predict(X_scaled, verbose=0)[0]

            pred_class = int(np.argmax(probs))
            CLASS_INFO = {
                0: ("🔴 At Risk",        "result-at-risk", "#e74c3c",
                    "This student is at risk of failing. Immediate academic intervention is recommended."),
                1: ("🟡 Average",         "result-average", "#f39c12",
                    "This student is performing at an average level. Consistent support can help them excel."),
                2: ("🟢 High Performer",  "result-high",    "#27ae60",
                    "This student is a high performer! Keep up the motivation and enrichment activities."),
            }
            label, css_class, color, advice = CLASS_INFO[pred_class]
            confidence = float(probs[pred_class]) * 100

            # ── Risk badge ────────────────────────────────────────
            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-title">Predicted Risk Level</div>
                <div class="result-label" style="color:{color}">{label}</div>
                <div style="color:#ccc;font-size:0.95rem;margin-top:8px">
                    Confidence: <b style="color:{color}">{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability bars ──────────────────────────────────
            st.markdown('<div class="card"><h3>📊 Class Probabilities</h3>', unsafe_allow_html=True)
            class_labels = ["🔴 At Risk", "🟡 Average", "🟢 High Performer"]
            bar_colors   = ["#e74c3c", "#f39c12", "#27ae60"]
            for i, (lbl, clr) in enumerate(zip(class_labels, bar_colors)):
                pct = float(probs[i]) * 100
                st.markdown(f'<div class="prob-label">{lbl} — {pct:.1f}%</div>',
                            unsafe_allow_html=True)
                st.progress(float(probs[i]))
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Advice ────────────────────────────────────────────
            st.markdown(f"""
            <div class="tip-box">
                💡 <b>Insight:</b> {advice}
            </div>
            """, unsafe_allow_html=True)

            # ── Key factors ───────────────────────────────────────
            st.markdown('<div class="card"><h3>🔑 Key Input Summary</h3>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{G1}</div>
                    <div class="metric-lbl">G1 Grade</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{G2}</div>
                    <div class="metric-lbl">G2 Grade</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{failures}</div>
                    <div class="metric-lbl">Past Failures</div>
                </div>""", unsafe_allow_html=True)

            m4, m5, m6 = st.columns(3)
            with m4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{absences}</div>
                    <div class="metric-lbl">Absences</div>
                </div>""", unsafe_allow_html=True)
            with m5:
                study_labels = ["<2 hrs","2-5 hrs","5-10 hrs",">10 hrs"]
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="font-size:1.1rem">{study_labels[studytime-1]}</div>
                    <div class="metric-lbl">Study Time</div>
                </div>""", unsafe_allow_html=True)
            with m6:
                higher_emoji = "✅" if higher == "yes" else "❌"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{higher_emoji}</div>
                    <div class="metric-lbl">Aims for Higher Edu</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Personalised tips ──────────────────────────────────
            st.markdown('<div class="card"><h3>📌 Recommendations</h3>', unsafe_allow_html=True)
            tips = []
            if G1 < 10 or G2 < 10:
                tips.append("📚 Mid-term grades are low — consider a tutor or study group.")
            if failures > 0:
                tips.append(f"⚠️ {failures} past failure(s) — targeted remediation needed.")
            if absences > 10:
                tips.append("🏫 High absence rate — investigate attendance barriers.")
            if studytime <= 1:
                tips.append("⏱️ Study time is below 2 hrs/week — encourage a study schedule.")
            if Walc >= 4 or Dalc >= 3:
                tips.append("🚨 High alcohol consumption reported — counselling may help.")
            if internet == "no":
                tips.append("🌐 No internet at home — consider school resource access.")
            if higher == "no":
                tips.append("🎯 Student doesn't aspire to higher education — motivational support may help.")
            if not tips:
                tips.append("🌟 Profile looks strong! Maintain current habits and keep setting goals.")

            for tip in tips:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # ── Placeholder state ──────────────────────────────────
            st.markdown("""
            <div class="card" style="text-align:center;padding:50px 24px;">
                <div style="font-size:4rem;margin-bottom:16px;">🎓</div>
                <h3 style="color:#c9b8ff;font-size:1.3rem;">Ready to Predict</h3>
                <p style="color:#999;font-size:0.95rem;line-height:1.6">
                    Fill in the student profile on the left and click
                    <b style="color:#a78bfa">🔮 Predict Performance</b>
                    to see the AI prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h3>🧠 How It Works</h3>
                <p style="color:#bbb;font-size:0.9rem;line-height:1.7">
                    This app uses a trained <b>Artificial Neural Network (ANN)</b> with
                    4 hidden layers (256 → 128 → 64 → 32 neurons), BatchNormalization,
                    and Dropout regularisation.<br><br>
                    The model classifies each student into one of three risk categories
                    based on 31 academic, social, and demographic features.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h3>📊 Risk Classes</h3>
                <div class="tip-box" style="border-color:#e74c3c;background:rgba(231,76,60,0.1)">
                    🔴 <b>At Risk</b> — Final grade &lt; 10
                </div>
                <div class="tip-box" style="border-color:#f39c12;background:rgba(243,156,18,0.1)">
                    🟡 <b>Average</b> — Final grade 10–13
                </div>
                <div class="tip-box" style="border-color:#27ae60;background:rgba(39,174,96,0.1)">
                    🟢 <b>High Performer</b> — Final grade ≥ 14
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ABOUT MODEL
# ═══════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown("# ℹ️ About the Model")

    st.markdown("""
    <div class="card">
        <h3>🏗️ Architecture</h3>
        <p style="color:#bbb;line-height:1.8">
        The predictor uses a <b>deep Artificial Neural Network</b> trained on the
        UCI Student Performance dataset.<br><br>
        <b>Input:</b> 31 features (academic, social, demographic)<br>
        <b>Layer 1:</b> Dense(256, ReLU) + BatchNorm + Dropout(0.3)<br>
        <b>Layer 2:</b> Dense(128, ReLU) + BatchNorm + Dropout(0.3)<br>
        <b>Layer 3:</b> Dense(64, ReLU)  + BatchNorm + Dropout(0.2)<br>
        <b>Layer 4:</b> Dense(32, ReLU)  + Dropout(0.2)<br>
        <b>Output:</b> Dense(3, Softmax) → At Risk / Average / High Performer
        </p>
    </div>

    <div class="card">
        <h3>⚙️ Training Setup</h3>
        <p style="color:#bbb;line-height:1.8">
        <b>Optimiser:</b> Adam (lr = 0.001)<br>
        <b>Loss:</b> Categorical Crossentropy<br>
        <b>Epochs:</b> Up to 100 (EarlyStopping, patience=20)<br>
        <b>Scheduler:</b> ReduceLROnPlateau (factor=0.5, patience=8)<br>
        <b>Split:</b> 70% train · 15% validation · 15% test
        </p>
    </div>

    <div class="card">
        <h3>🔑 Most Important Features</h3>
        <p style="color:#bbb;line-height:1.8">
        Based on Random Forest feature importance analysis:<br><br>
        1. G2 — Second Period Grade (strongest predictor)<br>
        2. G1 — First Period Grade<br>
        3. failures — Past class failures<br>
        4. absences — Number of absences<br>
        5. Medu — Mother's education level<br>
        6. studytime — Weekly study time<br>
        7. age — Student age<br>
        8. goout — Social activity level
        </p>
    </div>

    <div class="card">
        <h3>🎯 Target Classes</h3>
        <div class="tip-box" style="border-color:#e74c3c;background:rgba(231,76,60,0.1)">
            🔴 <b>At Risk (0)</b> — G3 &lt; 10. Student needs urgent academic support.
        </div>
        <div class="tip-box" style="border-color:#f39c12;background:rgba(243,156,18,0.1)">
            🟡 <b>Average (1)</b> — G3 between 10–13. Student is progressing but has room to grow.
        </div>
        <div class="tip-box" style="border-color:#27ae60;background:rgba(39,174,96,0.1)">
            🟢 <b>High Performer (2)</b> — G3 ≥ 14. Student is excelling academically.
        </div>
    </div>
    """, unsafe_allow_html=True)
