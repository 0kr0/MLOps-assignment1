import streamlit as st
import requests

st.title("🍷 Wine Class Predictor")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34)
density = st.number_input("Density", value=0.9978, format="%.5f")
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict"):
    data = {
        "fixed_acidity": fixed_acidity, 
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    try:
        response = requests.post("http://api:8000/predict", json=data)
        if response.status_code == 200:
            st.success(f"Predicted Wine Class: {response.json()['quality_prediction']}")  # **Фикс: Ключ 'quality_prediction' из твоего API**
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
