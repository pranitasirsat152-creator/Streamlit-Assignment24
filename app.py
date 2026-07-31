import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_classification_model.pkl")

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age", 1, 100, 40)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
bp = st.number_input("Resting Blood Pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 0, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
ecg = st.selectbox("Resting ECG", [0, 1, 2])
hr = st.number_input("Maximum Heart Rate", 50, 250, 150)
angina = st.selectbox("Exercise Angina", [0, 1])
oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
slope = st.selectbox("ST Slope", [0, 1, 2])

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Sex":[sex],
        "ChestPainType":[cp],
        "RestingBP":[bp],
        "Cholesterol":[chol],
        "FastingBS":[fbs],
        "RestingECG":[ecg],
        "MaxHR":[hr],
        "ExerciseAngina":[angina],
        "Oldpeak":[oldpeak],
        "ST_Slope":[slope]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")
