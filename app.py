import json
import os
from datetime import date

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bike Demand Prediction", page_icon="🚲", layout="centered")
st.title("🚲 Bike-Sharing Rental Demand Prediction")
st.write("Estimate hourly bike rental demand using calendar and weather information.")

MODEL_PATH = os.path.join("deployment_artifacts", "bike_demand_model.joblib")
FEATURES_PATH = os.path.join("deployment_artifacts", "feature_columns.json")
METRICS_PATH = os.path.join("deployment_artifacts", "model_metrics.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error(
        "Trained model files are missing. First run the Jupyter notebook through "
        "the 'Save the trained model for deployment' section. Then copy the "
        "generated deployment_artifacts folder beside this app.py file."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

with st.form("prediction_form"):
    selected_date = st.date_input("Date", value=date.today())
    hour = st.slider("Hour of day", 0, 23, 8)
    weekday = st.selectbox("Weekday (0 = Monday, 6 = Sunday)", list(range(7)), index=0)
    season = st.selectbox("Season", ["springer", "summer", "fall", "winter"])
    holiday = st.selectbox("Holiday", ["No", "Yes"])
    workingday = st.selectbox("Working day", ["Working Day", "No work"])
    weather = st.selectbox("Weather", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    temp = st.slider("Temperature (normalised)", 0.0, 1.0, 0.5)
    atemp = st.slider("Feels-like temperature (normalised)", 0.0, 1.0, 0.5)
    hum = st.slider("Humidity (normalised)", 0.0, 1.0, 0.5)
    windspeed = st.slider("Windspeed (normalised)", 0.0, 1.0, 0.2)
    submitted = st.form_submit_button("Predict bike demand")

if submitted:
    dt = pd.Timestamp(selected_date)
    input_values = {
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weathersit": weather,
        "hr": hour,
        "weekday": weekday,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "year_feature": dt.year,
        "month_feature": dt.month,
        "day_of_month": dt.day,
        "day_of_week_from_date": dt.dayofweek,
        "is_weekend": int(dt.dayofweek >= 5),
    }
    input_df = pd.DataFrame([{col: input_values.get(col) for col in feature_columns}])
    prediction = max(0, float(model.predict(input_df)[0]))
    st.success(f"Estimated hourly bike rentals: {prediction:.0f}")
    st.caption("This is an estimate from the trained model, not a guaranteed rental count.")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with st.expander("Model information"):
        st.write("Selected model:", metrics.get("selected_model", "Not recorded"))
        st.json(metrics.get("test_metrics", {}))
