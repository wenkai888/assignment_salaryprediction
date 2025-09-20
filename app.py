import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import load
import streamlit as st


# Load trained model
model = load("best_salary_model.joblib")

st.title("💼 Salary Prediction App")
st.write("Enter the details below to predict salary:")

# User inputs
education = st.selectbox(
    "Education", 
    ["High School", "Bachelor's", "Master's", "PhD"]
)
location = st.selectbox(
    "Location", 
    ["Urban", "Suburban", "Rural"]
)
job_title = st.selectbox(
    "Job Title", 
    ["Analyst", "Engineer", "Manager", "Director"]
)
gender = st.selectbox(
    "Gender", 
    ["Male", "Female"]
)
experience = st.number_input(
    "Years of Experience", min_value=0, max_value=50, value=5
)
age = st.number_input(
    "Age", min_value=18, max_value=64, value=25
)

# Predict button
if st.button("Predict Salary"):
    # Validation checks
    if experience > age - 18:
        st.error("⚠️ Years of experience cannot exceed working age (age - 18). Please adjust.")
    else:
        # Create DataFrame from inputs
        input_df = pd.DataFrame([{
            "Education": education,
            "Location": location,
            "Job_Title": job_title,
            "Gender": gender,
            "Experience": experience,
            "Age": age
        }])
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        # RMSE for range (replace with your model’s RMSE)
        RMSE = 10588.75
        lower = max(0, prediction - RMSE)  # no negative salary
        upper = prediction + RMSE
        
        # Show results
        st.success(f"💰 Predicted Salary: $ {prediction:,.2f}")
        st.info(f"📊 Estimated Salary Range: $ {lower:,.2f} – $ {upper:,.2f}")











