#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import streamlit as st
import joblib
import numpy as np
# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Data Prediction", page_icon="")

st.title("Fraud Data Prediction")

st.write("Enter the feature values below to predict the data.")

# User Inputs
avg_amount_days = st.number_input("avg_amount_days", min_value=0.0, format="%.4f")
transaction_amount = st.number_input("transaction_amount", min_value=0.0, format="%.4f")
is_declined = st.number_input("is_declined", min_value=0.0, format="%.4f")
number_declines_days = st.number_input("number_declines_days", min_value=0.0, format="%.4f")
foreign_transaction = st.number_input("foreign_transaction", min_value=0.0, format="%.4f")
high_risk_countries = st.number_input("high_risk_countries", min_value=0.0, format="%.4f")
daily_chbk_avg_amt = st.number_input("daily_chbk_avg_amt", min_value=0.0, format="%.4f")
Sm_avg_chbk_amt = st.number_input("6m_avg_chbk_amt", min_value=0.0, format="%.4f")
Sm_chbk_freq = st.number_input("6m_chbk_freq", min_value=0.0, format="%.4f")

if st.button("Predict Diagnosis"):

    input_data = np.array([[
        avg_amount_days,
        transaction_amount,
        is_declined,
        number_declines_days,
        foreign_transaction,
        high_risk_countries,
        daily_chbk_avg_amt,
        Sm_avg_chbk_amt,
        Sm_chbk_freq,
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display Result
    if prediction[0] == "M" or prediction[0] == 1:
        st.error("🔴 fradulent: Yes, fradulent")
    else:
        st.success("🟢 fradulent: It is Safe")

