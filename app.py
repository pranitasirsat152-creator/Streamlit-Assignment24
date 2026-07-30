import joblib
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris

st.title("Assignment 24 - ML App")

cls_model = joblib.load("best_classifier.pkl")
reg_model = joblib.load("best_regressor.pkl")
scaler = joblib.load("scaler(1).pkl")

iris = load_iris()


problem = st.sidebar.radio("1. Choose Problem Type", ("Iris Classification", "Diabetes Regression"))

if problem == "Iris Classification":
    st.header("Iris Classification")
    inputs = []
    for i, name in enumerate(iris.feature_names):
        val = st.slider(name, float(iris.data[:,i].min()), float(iris.data[:,i].max()), float(iris.data[:,i].mean()))
        inputs.append(val)

    if st.button("Predict"):
        pred = cls_model.predict([inputs])
        st.success(f"Predicted: {iris.target_names[pred][0]}")

else:
    st.header("Diabetes Regression")
    inputs = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(10)]
    if st.button("Predict"):
        pred = reg_model.predict(scaler.transform([inputs]))
        st.success(f"Predicted Value: {pred[0]:.2f}")
