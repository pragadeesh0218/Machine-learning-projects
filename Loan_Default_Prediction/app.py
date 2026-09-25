import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_logistic_model.pkl")
preprocessor = joblib.load("loan_preprocessor.pkl")
threshold = joblib.load("loan_threshold.pkl")

st.title("Loan Default Prediction")

st.write(
    "Enter borrower information to estimate the probability of loan default."
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

loan_term = st.selectbox(
    "Loan Term",
    [36, 60]
)

interest_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=10.0
)

monthly_installment = st.number_input(
    "Monthly Installment",
    min_value=0.0,
    value=300.0
)

loan_sub_grade = st.selectbox(
    "Loan Sub Grade",
    [
        "A1", "A2", "A3", "A4", "A5",
        "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5",
        "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5",
        "F1", "F2", "F3", "F4", "F5",
        "G1", "G2", "G3", "G4", "G5"
    ]
)

employment_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

income_verification = st.selectbox(
    "Income Verification",
    ["Verified", "Source Verified", "Not Verified"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    [
        "credit_card",
        "debt_consolidation",
        "home_improvement",
        "major_purchase",
        "small_business",
        "car",
        "other"
    ]
)

dti = st.number_input(
    "Debt To Income Ratio",
    min_value=0.0,
    value=15.0
)

delinquencies = st.number_input(
    "Delinquencies Last 2 Years",
    min_value=0,
    value=0
)

credit_inquiries = st.number_input(
    "Credit Inquiries Last 6 Months",
    min_value=0,
    value=1
)

open_accounts = st.number_input(
    "Open Credit Accounts",
    min_value=0,
    value=8
)

public_records = st.number_input(
    "Public Records",
    min_value=0,
    value=0
)

revolving_balance = st.number_input(
    "Revolving Balance",
    min_value=0.0,
    value=5000.0
)

revolving_utilization = st.number_input(
    "Revolving Utilization (%)",
    min_value=0.0,
    value=40.0
)

total_accounts = st.number_input(
    "Total Credit Accounts",
    min_value=0,
    value=15
)

bankruptcies = st.number_input(
    "Public Record Bankruptcies",
    min_value=0.0,
    value=0.0
)

issue_year = st.number_input(
    "Issue Year",
    min_value=2007,
    max_value=2026,
    value=2015
)

issue_month = st.number_input(
    "Issue Month",
    min_value=1,
    max_value=12,
    value=1
)

credit_history_years = st.number_input(
    "Credit History Years",
    min_value=0.0,
    value=10.0
)

if st.button("Predict Loan Default"):

    input_data = pd.DataFrame([{
        "Loan Amount": loan_amount,
        "Loan Term": loan_term,
        "Interest Rate": interest_rate,
        "Monthly Installment": monthly_installment,
        "Loan Sub Grade": loan_sub_grade,
        "Employment Length": employment_length,
        "Home Ownership": home_ownership,
        "Annual Income": annual_income,
        "Income Verification": income_verification,
        "Loan Purpose": loan_purpose,
        "Debt To Income Ratio": dti,
        "Delinquencies Last 2 Years": delinquencies,
        "Credit Inquiries Last 6 Months": credit_inquiries,
        "Open Credit Accounts": open_accounts,
        "Public Records": public_records,
        "Revolving Balance": revolving_balance,
        "Revolving Utilization": revolving_utilization,
        "Total Credit Accounts": total_accounts,
        "Public Record Bankruptcies": bankruptcies,
        "Issue Year": issue_year,
        "Issue Month": issue_month,
        "Credit History Years": credit_history_years
    }])

    input_processed = preprocessor.transform(input_data)

    probability = model.predict_proba(
        input_processed
    )[0, 1]

    prediction = int(probability >= threshold)

    st.subheader("Prediction")

    st.write(
        f"Default Probability: **{probability:.2%}**"
    )

    if prediction == 1:
        st.error("Prediction: Loan Default")
    else:
        st.success("Prediction: Non-Default")