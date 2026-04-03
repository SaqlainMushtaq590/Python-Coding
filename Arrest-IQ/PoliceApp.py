import streamlit as st
import pandas as pd
import pickle
import os

# --- Load Model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "police_arrest_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# --- Page Config ---
st.set_page_config(page_title="ArrestIQ — Police Stop Predictor", page_icon="🚔")

# --- Header ---
st.title("🚔 ArrestIQ — Police Stop Outcome Predictor")
st.markdown("""
Welcome to **ArrestIQ**! This app uses a Machine Learning model trained on **65,000+ real police stop records** 
to predict whether a traffic stop is likely to result in an arrest.

Simply fill in the stop details on the left sidebar and click **Analyze Stop** to get the prediction.
""")

st.divider()

# --- Sidebar Inputs ---
st.sidebar.title("🗂️ Enter Stop Details")
st.sidebar.markdown("Fill in the details of the traffic stop below:")

age = st.sidebar.slider("🎂 Driver's Age", 16, 90, 25, help="Select the age of the driver")

gender = st.sidebar.radio(
    "👤 Driver's Gender",
    ["Male", "Female"],
    help="Select the gender of the driver"
)

race = st.sidebar.selectbox(
    "🌍 Driver's Race",
    ["White", "Black", "Hispanic", "Other"],
    help="Select the race of the driver as recorded in the stop"
)

search = st.sidebar.selectbox(
    "🔍 Was the Driver Searched?",
    ["Yes", "No"],
    help="Was a physical search conducted on the driver or vehicle during this stop?"
)

drugs = st.sidebar.selectbox(
    "💊 Was this a Drug-Related Stop?",
    ["Yes", "No"],
    help="Was the reason for this stop related to suspected drug activity?"
)

st.sidebar.divider()
st.sidebar.info("📌 All predictions are based on historical data patterns and are for educational purposes only.")

# --- Input Dictionary ---
input_dict = {
    "driver_age": age,
    "search_conducted": 1 if search == "Yes" else 0,
    "drugs_related_stop": 1 if drugs == "Yes" else 0,
    "driver_gender_M": 1 if gender == "Male" else 0,
    "driver_race_Black": 1 if race == "Black" else 0,
    "driver_race_Hispanic": 1 if race == "Hispanic" else 0,
    "driver_race_Other": 1 if race == "Other" else 0,
    "driver_race_White": 1 if race == "White" else 0
}

input_df = pd.DataFrame([input_dict])

# --- Analyze Button ---
if st.button("🔍 Analyze This Stop", use_container_width=True):

    probabilities = model.predict_proba(input_df)[0]
    prob_arrest = probabilities[1] * 100
    prob_no_arrest = probabilities[0] * 100

    st.divider()

    # --- Combined Condition Logic ---
    if search == "Yes" and drugs == "Yes":
        risk_level = "🔴 HIGH"
        arrest_likely = True
        reason = (
            "Both a search was conducted AND the stop was drug-related. "
            "These are the two strongest indicators of arrest in the dataset. "
            "When both are present together, the likelihood of arrest increases significantly."
        )

    elif search == "Yes" and drugs == "No":
        risk_level = "🟠 MODERATE"
        arrest_likely = prob_arrest > 15
        reason = (
            "A search was conducted during this stop. "
            "Officers typically search a driver when they have reasonable suspicion. "
            "The final outcome depends on what was found during the search."
        )

    elif search == "No" and drugs == "Yes":
        risk_level = "🟡 LOW-MODERATE"
        arrest_likely = prob_arrest > 25
        reason = (
            "This stop was drug-related, but no search was conducted. "
            "The officer may have suspected drug activity but did not find enough evidence to proceed with a search. "
            "Arrest is possible but not confirmed at this stage."
        )

    else:
        risk_level = "🟢 LOW"
        arrest_likely = prob_arrest > 50
        reason = (
            "No search was conducted and there was no drug involvement in this stop. "
            "This appears to be a routine traffic stop, such as a speeding or equipment violation. "
            "The likelihood of arrest in such cases is generally very low."
        )

    # --- Prediction Result ---
    if arrest_likely:
        st.error("## 🚨 Prediction: Arrest Is Likely")
        st.progress(probabilities[1], text=f"Arrest Probability: {prob_arrest:.0f}%")
        st.markdown(f"The model predicts with **{prob_arrest:.0f}% confidence** that this stop will result in an **arrest**.")
    else:
        st.success("## ✅ Prediction: No Arrest Expected")
        st.progress(probabilities[0], text=f"No Arrest Probability: {prob_no_arrest:.0f}%")
        st.markdown(f"The model predicts with **{prob_no_arrest:.0f}% confidence** that this stop will **not** result in an arrest.")

    # --- Risk Level ---
    st.markdown(f"### Risk Level: {risk_level}")

    # --- Reason ---
    st.info(f"💡 **Why this prediction?**\n\n{reason}")

    st.divider()

    # --- Stop Condition Summary Table ---
    st.markdown("### 📋 Stop Condition Summary")
    st.markdown("Here is a breakdown of each factor entered and its level of impact on the prediction:")

    summary = {
        "Factor": [
            "Search Conducted",
            "Drug-Related Stop",
            "Driver's Age",
            "Driver's Gender",
            "Driver's Race"
        ],
        "Value Entered": [search, drugs, f"{age} years old", gender, race],
        "Impact on Prediction": [
            "🔴 High Impact — Search is the strongest arrest indicator" if search == "Yes" else "🟢 Low Impact — No search means lower arrest risk",
            "🔴 High Impact — Drug involvement increases arrest likelihood" if drugs == "Yes" else "🟢 Low Impact — No drug involvement detected",
            "🟡 Moderate Impact — Younger drivers show slightly higher risk" if age < 25 else "🟢 Low Impact — Age is within normal range",
            "🟡 Moderate Impact" if gender == "Male" else "🟢 Low Impact",
            "🟡 Moderate Impact — Based on historical data patterns" if race in ["Black", "Hispanic"] else "🟢 Low Impact"
        ]
    }

    st.table(pd.DataFrame(summary))

    # --- Probability Breakdown ---
    with st.expander("📊 View Full Probability Breakdown"):
        st.markdown(f"- **Probability of Arrest:** {prob_arrest:.2f}%")
        st.markdown(f"- **Probability of No Arrest:** {prob_no_arrest:.2f}%")
        st.divider()
        st.markdown("#### How the Risk Thresholds Work:")
        st.markdown("""
        | Search | Drugs | Risk Level | Arrest is Predicted When |
        |--------|-------|------------|--------------------------|
        | Yes | Yes | 🔴 HIGH | Always |
        | Yes | No | 🟠 MODERATE | Model confidence > 15% |
        | No | Yes | 🟡 LOW-MODERATE | Model confidence > 25% |
        | No | No | 🟢 LOW | Model confidence > 50% |
        """)

    st.divider()
    st.caption("⚠️ Disclaimer: This tool is built for educational purposes using historical police stop data. Predictions should not be used for real law enforcement decisions.")