import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Paths to model and scaler files
MODEL_PATH = 'model/rf_model.pkl'
SCALER_PATH = 'model/scaler.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler file not found! Please ensure they exist in the 'model/' folder.")
    st.stop()

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🚲 Bike Sharing Demand Predictor")
st.markdown("Predict the total number of bike rentals for a day based on weather and date features.")

# Input widgets
season = st.selectbox("Season", options=[1, 2, 3, 4], help="1: Spring, 2: Summer, 3: Fall, 4: Winter")
yr = st.selectbox("Year", options=[0, 1], help="0: 2011, 1: 2012")
mnth = st.slider("Month", 1, 12)
holiday = st.selectbox("Holiday", options=[0, 1], help="0: No, 1: Yes")
weekday = st.slider("Weekday", 0, 6, help="0=Sunday, 6=Saturday")
workingday = st.selectbox("Working Day", options=[0, 1], help="0: No, 1: Yes")
weathersit = st.selectbox(
    "Weather Situation", options=[1, 2, 3, 4],
    help="1: Clear, 2: Mist + Cloudy, 3: Light Rain/Snow, 4: Heavy Rain/Snow"
)
temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
atemp = st.slider("Feels Like Temperature (Normalized)", 0.0, 1.0, 0.5)
hum = st.slider("Humidity (Normalized)", 0.0, 1.0, 0.5)
windspeed = st.slider("Windspeed (Normalized)", 0.0, 1.0, 0.5)

if st.button("Predict Bike Count"):
    # Create DataFrame with feature names to avoid warnings
    input_df = pd.DataFrame(
        [[season, yr, mnth, holiday, weekday, workingday,
          weathersit, temp, atemp, hum, windspeed]],
        columns=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    )
    # Scale input
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Bike Count: {int(prediction)}")
