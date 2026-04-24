import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Student Risk AI",
    page_icon="🎓",
    layout="wide"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(to right, #0f172a, #1e293b);
        color: white;
    }

    /* Title styling */
    h1 {
        text-align: center;
        color: #ffffff;
        font-weight: 700;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 20px;
    }

    /* Card style */
    .card {
        background-color: rgba(255, 255, 255, 0.06);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* Button */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
    }

    /* Dataframe styling */
    .dataframe {
        background-color: white;
        color: black;
    }

    </style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("<h1>🎓 Student Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict student performance and academic risk using AI</div>", unsafe_allow_html=True)

st.divider()

# LOAD ARTIFACTS

@st.cache_resource
def load_artifacts():
    return joblib.load("artifacts/student_risk_artifacts.joblib")

artifacts = load_artifacts()
reg = artifacts["reg"]
clf = artifacts["clf_pipe"]
threshold = artifacts["threshold"]
base_features = artifacts["base_features"]


# ============================
# CSV UPLOAD SECTION
# ============================
st.markdown("## Upload Student Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Data")
    st.dataframe(data.head())

    if st.button("Run Prediction on Dataset"):
        try:
            with st.spinner("Predicting..."):

                # Example feature selection (we will refine later)
                X = data[base_features]

                # Predict score
                predicted_scores = reg.predict(X)

                # Risk classification
                risk = ["Risk" if s < threshold else "Safe" for s in predicted_scores]

                # Add results
                data["Predicted_Score"] = predicted_scores
                data["Risk"] = risk

            st.success("Prediction completed!")
            st.dataframe(data)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# MANUAL INPUT SECTION 

st.markdown("## Predict Single Student")

with st.form("manual_input_form"):
    study_hours = st.number_input("Weekly Self Study Hours", value=5.0)
    attendance  = st.number_input("Attendance Percentage", value=80.0)
    participation = st.number_input("Class Participation", value=5.0)
    submitted = st.form_submit_button("Predict Student Risk")

if submitted:
    try:
        with st.spinner("Analyzing student performance..."):
            input_data = pd.DataFrame([{
                "weekly_self_study_hours": study_hours,
                "attendance_percentage": attendance,
                "class_participation": participation
            }])[base_features]

            pred_score = reg.predict(input_data)[0]
            X_aug = input_data.copy()
            X_aug["predicted_score"] = pred_score

            prob = clf.predict_proba(X_aug)[:, 1][0]
            risk = "At Risk" if prob >= threshold else "Safe"

        st.success("Prediction completed!")
        st.write(f"Predicted Score: **{pred_score:.2f}**")
        st.write(f"At Risk Probability: **{prob:.3f}** (threshold={threshold})")
        st.write(f"Risk Level: **{risk}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("""
<div class='card'>

<h3>Understanding Your Result</h3>

<p>Your <b>predicted score</b> estimates your academic performance, while your <b>risk probability</b> shows your likelihood of facing academic challenges.</p>

<ul>
<li><b>Safe:</b> You are currently performing well, but continued effort is important.</li>
<li><b>At Risk:</b> You may need to improve your study habits to avoid poor performance.</li>
</ul>

<p><i>If your probability is close to the threshold, you are in a borderline zone and should take action early.</i></p>

<p><b>How to improve:</b></p>
<ul>
<li>Increase your study hours</li>
<li>Improve class attendance</li>
<li>Participate more actively in class</li>
</ul>

<p>Small improvements in these areas can significantly reduce your risk.</p>

</div>
""", unsafe_allow_html=True)
