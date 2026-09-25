import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Federal Tax Liability Prediction")

long_term_capital_gain = st.number_input("Long Term Capital Gain", min_value=0.0)
rental_income = st.number_input("Rental Income", min_value=0.0)
total_itemized_deductions = st.number_input("Total Itemized Deductions", min_value=0.0)
interest_income = st.number_input("Interest Income", min_value=0.0)
dividend_income = st.number_input("Dividend Income", min_value=0.0)
real_estate_tax = st.number_input("Real Estate Tax", min_value=0.0)
wage_income = st.number_input("Wage Income", min_value=0.0)
business_income = st.number_input("Business Income", min_value=0.0)
spouse_age = st.number_input("Spouse Age", min_value=0, max_value=120)
pension_income = st.number_input("Pension Income", min_value=0.0)

if st.button("Predict Tax Liability"):

    new_data = np.array([[
        long_term_capital_gain,
        rental_income,
        total_itemized_deductions,
        interest_income,
        dividend_income,
        real_estate_tax,
        wage_income,
        business_income,
        spouse_age,
        pension_income
    ]])

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)
    prediction=max(0.0,prediction[0])

    st.success(f"Predicted Federal Tax Liability: ${prediction:.2f}")