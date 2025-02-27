import streamlit as st
import pickle
import numpy as np

st.title("AI-Powered SME Loan Approval")

# Input fields
income = st.number_input("Annual Income ($)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

# Load the trained model
model = pickle.load(open("loan_model.pkl", "rb"))

if st.button("Check Loan Approval"):
    # Prepare data for prediction
    features = np.array([[income, loan_amount, credit_score]])
    
    # Make prediction
    prediction = model.predict(features)

    # Show result
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Denied ❌")



