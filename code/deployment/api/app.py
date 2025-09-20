from fastapi import FastAPI
import joblib
import pandas as pd
import os
from pathlib import Path

app = FastAPI()

model = joblib.load("/models/wine_model.joblib")
scaler = joblib.load("/models/wine_scaler.joblib")

feature_order = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
@app.post("/predict")
def predict(data: dict):  # e.g., {"fixed_acidity": 7.4, "volatile_acidity": 0.7, ...} for all features
    # Convert input to DataFrame (ensure order matches training columns)

    api_to_model = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol"
    }

    # строим input_data с правильными именами
    input_data = {model_name: data.get(api_name, 0) for api_name, model_name in api_to_model.items()}
    input_df = pd.DataFrame([input_data])

    # scale + predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    return {"quality_prediction": int(prediction)}

