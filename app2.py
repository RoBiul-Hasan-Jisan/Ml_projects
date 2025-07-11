import streamlit as st
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")

# Load model and columns
model = joblib.load('model/car_price_model2.pkl')
with open('model/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Extract categorical options from model columns (drop_first applied during training)
fuel_types = [c.replace("fuel_", "") for c in model_columns if c.startswith("fuel_")]
seller_types = [c.replace("seller_type_", "") for c in model_columns if c.startswith("seller_type_")]
trans_types = [c.replace("transmission_", "") for c in model_columns if c.startswith("transmission_")]
brands = [c.replace("brand_", "") for c in model_columns if c.startswith("brand_")]

st.markdown("### Enter Car Details")

# Inputs
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000, value=25000, step=500)
car_age = st.slider("Car Age (Years)", min_value=0, max_value=30, value=5)
owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3, 4], index=1)

fuel = st.selectbox("Fuel Type", fuel_types)
seller_type = st.selectbox("Seller Type", seller_types)
transmission = st.selectbox("Transmission Type", trans_types)
brand = st.selectbox("Brand", brands)

# Prepare input vector of zeros for all features
input_data = np.zeros(len(model_columns))

for i, col in enumerate(model_columns):
    if col == 'car_age':
        input_data[i] = car_age
    elif col == 'owner':
        input_data[i] = owner
    elif col == 'km_driven':
        input_data[i] = np.log1p(km_driven)  # log transform as done in training
    elif col == f'fuel_{fuel}':
        input_data[i] = 1
    elif col == f'seller_type_{seller_type}':
        input_data[i] = 1
    elif col == f'transmission_{transmission}':
        input_data[i] = 1
    elif col == f'brand_{brand}':
        input_data[i] = 1

if st.button("💰 Predict Price"):
    prediction_log = model.predict([input_data])[0]
    prediction = np.expm1(prediction_log)  # inverse of log1p on selling_price
    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")
