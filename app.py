import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Telecom Customer Churn Prediction",)

model = joblib.load("xgb_churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Telecom Customer Churn Prediction")
st.write("Enter customer information below to predict ",
         "whether the customer is likely to churn.")
st.divider()
st.subheader("Customer Information")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

internet_service = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"])
st.divider()

if st.button(" Predict Churn"):
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "Contract_One year": [1 if contract == "One year" else 0],
        "Contract_Two year": [1 if contract == "Two year" else 0],
        "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
        "InternetService_No": [1 if internet_service == "No" else 0]})
    
    input_data = pd.get_dummies(input_data,drop_first=True)
    input_data = input_data.reindex(columns=model_columns,fill_value=0)

    prediction = model.predict(input_data)
    prediction_probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader(" Prediction Result")
    if prediction[0] == 1:
        st.error(
            "The customer is likely to churn."
        )
    else:
        st.success(
            "The customer is not likely to churn."
        )
    st.metric(label="Churn Probability",value=f"{prediction_probability:.2%}")
    st.progress(float(prediction_probability))