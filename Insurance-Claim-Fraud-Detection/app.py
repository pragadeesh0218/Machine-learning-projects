import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("insurance_ridge_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("💰 Insurance Claim Amount Prediction")

st.write(
    "Predict the estimated insurance claim amount using "
    "a Ridge Regression model."
)

st.divider()


# --------------------------------------------------
# Random input generator
# --------------------------------------------------

def generate_random_input():

    return {
        "age": np.random.randint(20, 70),
        "months_as_customer": np.random.randint(1, 500),
        "policy_deductable": np.random.choice([500, 1000, 2000]),
        "policy_annual_premium": round(
            np.random.uniform(400, 1500), 2
        ),
        "number_of_vehicles_involved": np.random.randint(1, 4),
        "bodily_injuries": np.random.randint(0, 3),
        "witnesses": np.random.randint(0, 4),
        "incident_hour_of_the_day": np.random.randint(0, 24),

        "incident_type": np.random.choice([
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        ]),

        "collision_type": np.random.choice([
            "Rear Collision",
            "Side Collision",
            "Front Collision"
        ]),

        "incident_severity": np.random.choice([
            "Major Damage",
            "Minor Damage",
            "Total Loss",
            "Trivial Damage"
        ]),

        "property_damage": np.random.choice([
            "YES",
            "NO"
        ]),

        "police_report_available": np.random.choice([
            "YES",
            "NO"
        ])
    }


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "random_data" not in st.session_state:
    st.session_state.random_data = generate_random_input()


# --------------------------------------------------
# Random input button
# --------------------------------------------------

if st.button("🎲 Generate Random Input"):

    st.session_state.random_data = generate_random_input()


data = st.session_state.random_data


# --------------------------------------------------
# Input form
# --------------------------------------------------

st.subheader("Enter Insurance Details")

col1, col2, col3 = st.columns(3)


# Numerical inputs
with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=int(data["age"])
    )

    months_as_customer = st.number_input(
        "Months as Customer",
        min_value=0,
        max_value=1000,
        value=int(data["months_as_customer"])
    )

    policy_deductable = st.selectbox(
        "Policy Deductible",
        [500, 1000, 2000],
        index=[500, 1000, 2000].index(
            int(data["policy_deductable"])
        )
    )

    policy_annual_premium = st.number_input(
        "Policy Annual Premium",
        min_value=0.0,
        value=float(data["policy_annual_premium"]),
        step=10.0
    )


with col2:

    number_of_vehicles_involved = st.number_input(
        "Number of Vehicles Involved",
        min_value=1,
        max_value=10,
        value=int(data["number_of_vehicles_involved"])
    )

    bodily_injuries = st.number_input(
        "Bodily Injuries",
        min_value=0,
        max_value=10,
        value=int(data["bodily_injuries"])
    )

    witnesses = st.number_input(
        "Witnesses",
        min_value=0,
        max_value=10,
        value=int(data["witnesses"])
    )

    incident_hour_of_the_day = st.slider(
        "Incident Hour",
        min_value=0,
        max_value=23,
        value=int(data["incident_hour_of_the_day"])
    )


with col3:

    incident_type = st.selectbox(
        "Incident Type",
        [
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        ],
        index=[
            "Single Vehicle Collision",
            "Multi-vehicle Collision",
            "Vehicle Theft",
            "Parked Car"
        ].index(data["incident_type"])
    )

    collision_type = st.selectbox(
        "Collision Type",
        [
            "Rear Collision",
            "Side Collision",
            "Front Collision"
        ],
        index=[
            "Rear Collision",
            "Side Collision",
            "Front Collision"
        ].index(data["collision_type"])
    )

    incident_severity = st.selectbox(
        "Incident Severity",
        [
            "Major Damage",
            "Minor Damage",
            "Total Loss",
            "Trivial Damage"
        ],
        index=[
            "Major Damage",
            "Minor Damage",
            "Total Loss",
            "Trivial Damage"
        ].index(data["incident_severity"])
    )

    property_damage = st.selectbox(
        "Property Damage",
        ["YES", "NO"],
        index=["YES", "NO"].index(data["property_damage"])
    )

    police_report_available = st.selectbox(
        "Police Report Available",
        ["YES", "NO"],
        index=["YES", "NO"].index(
            data["police_report_available"]
        )
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Claim Amount", type="primary"):

    input_data = pd.DataFrame([{
        "age": age,
        "months_as_customer": months_as_customer,
        "policy_deductable": policy_deductable,
        "policy_annual_premium": policy_annual_premium,
        "number_of_vehicles_involved": number_of_vehicles_involved,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "incident_hour_of_the_day": incident_hour_of_the_day,

        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": incident_severity,
        "property_damage": property_damage,
        "police_report_available": police_report_available
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Estimated Insurance Claim Amount",
        value=f"{prediction:,.2f}"
    )

    st.info(
        "This is an ML-based estimated claim amount and "
        "should not be interpreted as the final approved claim."
    )