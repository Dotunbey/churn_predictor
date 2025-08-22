import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open("model_C=1.0.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

dv, model = load_model()

st.title("📊 Customer Churn Prediction App")

st.markdown("Fill in the customer details below to check if they are likely to churn.")

# Input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["female", "male"])
    seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["yes", "no"])
    dependents = st.selectbox("Dependents", ["yes", "no"])
    phoneservice = st.selectbox("Phone Service", ["yes", "no"])
    multiplelines = st.selectbox("Multiple Lines", ["no", "yes", "no_phone_service"])
    internetservice = st.selectbox("Internet Service", ["dsl", "fiber_optic", "no"])
    onlinesecurity = st.selectbox("Online Security", ["yes", "no", "no_internet_service"])
    onlinebackup = st.selectbox("Online Backup", ["yes", "no", "no_internet_service"])
    deviceprotection = st.selectbox("Device Protection", ["yes", "no", "no_internet_service"])
    techsupport = st.selectbox("Tech Support", ["yes", "no", "no_internet_service"])
    streamingtv = st.selectbox("Streaming TV", ["yes", "no", "no_internet_service"])
    streamingmovies = st.selectbox("Streaming Movies", ["yes", "no", "no_internet_service"])
    contract = st.selectbox("Contract", ["month-to-month", "one_year", "two_year"])
    paperlessbilling = st.selectbox("Paperless Billing", ["yes", "no"])
    paymentmethod = st.selectbox("Payment Method", [
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)"
    ])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    customer = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "tenure": tenure,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
    }

    # Convert to dataframe and predict
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    st.subheader("🔎 Prediction Result")
    st.write(f"**Churn Probability: {y_pred:.2f}**")

    if y_pred >= 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")
