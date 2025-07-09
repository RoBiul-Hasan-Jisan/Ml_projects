import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")
st.title("🚗 Car Price Predictor")
st.markdown("Use the inputs below to enter car details and estimate the resale price.")

# Load model safely
MODEL_PATH = "model/car_price_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found! Please train the model first.")
    st.stop()

# Input fields in main area
present_price = st.number_input("Present Price (in Lakhs)", 0.0, 50.0, 5.0, step=0.5)
kms_driven = st.number_input("Kilometers Driven", 0, 500000, 25000, step=500)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.slider("Car Age (in years)", 0, 30, 5)

# Prepare input as DataFrame with column names
input_df = pd.DataFrame(
    [[present_price, kms_driven, owner, car_age]],
    columns=["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
)

# Predict button below inputs
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"💰 Estimated Resale Price: ₹ {predicted_price:.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Instruction text always visible
st.info("👈 Enter car details above and click Predict.")
