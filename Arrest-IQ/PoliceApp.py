
import streamlit as st
import pandas as pd
import pickle

with open("police_arrest_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Arrest Predictor", page_icon="🚔")
st.title("🚔 Police Stop Outcome Predictor")
st.markdown("This app predicts the likelihood of arrest based on traffic stop details.")

st.sidebar.header("Traffic Stop Details")
age = st.sidebar.slider("Driver Age", 16, 90, 25)
gender = st.sidebar.radio("Gender", ["M", "F"])
race = st.sidebar.selectbox("Race", ["White", "Black", "Hispanic", "Other"])
search = st.sidebar.selectbox("Search Conducted", ["Yes", "No"])
drugs = st.sidebar.selectbox("Drugs Related Stop", ["Yes", "No"])

input_dict = {
    "driver_age": age,
    "search_conducted": 1 if search == "Yes" else 0,
    "drugs_related_stop": 1 if drugs == "Yes" else 0,
    "driver_gender_M": 1 if gender == "M" else 0,
    "driver_race_Black": 1 if race == "Black" else 0,
    "driver_race_Hispanic": 1 if race == "Hispanic" else 0,
    "driver_race_Other": 1 if race == "Other" else 0,
    "driver_race_White": 1 if race == "White" else 0
}

input_df = pd.DataFrame([input_dict])

if st.button("🔍 Analyze Stop"):

    probabilities = model.predict_proba(input_df)[0]
    prob_arrest = probabilities[1] * 100
    prob_no_arrest = probabilities[0] * 100

    # --- Combined Condition Logic ---
    if search == "Yes" and drugs == "Yes":
        # Strongest signal - both search and drugs present
        risk_level = "HIGH"
        arrest_likely = True
        reason = "Search was conducted AND stop is drugs-related — strongest arrest indicators both present."

    elif search == "Yes" and drugs == "No":
        # Moderate - search happened but no drugs
        risk_level = "MODERATE"

        arrest_likely = prob_arrest > 15
        reason = "Search was conducted. Arrest depends on what was found during search."

    elif search == "No" and drugs == "Yes":
        # Lower - drugs suspected but no search yet
        risk_level = "LOW-MODERATE"
        arrest_likely = prob_arrest > 20
        reason = "Stop is drugs-related but no search was conducted. Officer suspects but hasn't confirmed."

    else:
        # No search, no drugs - lowest risk
        risk_level = "LOW"
        arrest_likely = prob_arrest > 50
        reason = "No search conducted and no drug involvement. Likely a routine traffic stop."

    st.divider()

    # --- Display Result ---
    if arrest_likely:
        st.error("### 🚨 Prediction: Arrest Likely")
        st.progress(probabilities[1])
        st.write(f"**Model Confidence:** {prob_arrest:.0f}% chance of arrest")
    else:
        st.success("### ✅ Prediction: No Arrest Likely")
        st.progress(probabilities[0])
        st.write(f"**Model Confidence:** {prob_no_arrest:.0f}% chance of no arrest")

    # --- Risk Badge ---
    st.markdown(f"**Risk Level:** `{risk_level}`")

    # --- Reason Explanation ---
    st.info(f"💡 **Why:** {reason}")

    st.divider()

    # --- Condition Summary Table ---
    st.markdown("#### 📋 Stop Condition Summary")
    summary = {
        "Factor": ["Search Conducted", "Drugs Related", "Driver Age", "Gender", "Race"],
        "Value": [search, drugs, age, gender, race],
        "Impact": [
            "🔴 High" if search == "Yes" else "🟢 Low",
            "🔴 High" if drugs == "Yes" else "🟢 Low",
            "🟡 Moderate" if age < 25 else "🟢 Low",
            "🟡 Moderate" if gender == "M" else "🟢 Low",
            "🟡 Moderate" if race in ["Black", "Hispanic"] else "🟢 Low"
        ]
    }
    st.table(pd.DataFrame(summary))

    # --- Probability Breakdown ---
    with st.expander("📊 See Full Probability Breakdown"):
        st.write(f"- Probability of Arrest: **{prob_arrest:.2f}%**")
        st.write(f"- Probability of No Arrest: **{prob_no_arrest:.2f}%**")
        st.markdown("""
        **Condition Logic Used:**
        | Search | Drugs | Risk Level | Arrest Threshold |
        |--------|-------|------------|-----------------|
        | Yes | Yes | HIGH | Always Arrest Likely |
        | Yes | No | MODERATE | > 15% |
        | No | Yes | LOW-MODERATE | > 20% |
        | No | No | LOW | > 30% |
        """)