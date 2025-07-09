import streamlit as st
import numpy as np
import joblib
import pickle

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Predictor")

# Load the model and feature columns
model = joblib.load('model/car_price_model2.pkl')
with open('model/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.markdown("###  Enter Car Details")

# Numeric inputs
km_driven = st.number_input(
    " Kilometers Driven", min_value=0, max_value=1000000, value=25000, step=500, key="km_driven"
)
car_age = st.slider(
    " Car Age (Years)", min_value=0, max_value=30, value=5, key="car_age"
)
owners = st.selectbox(
    " Number of Previous Owners", options=[0, 1, 2, 3, 4], index=1, key="owners"
)

st.markdown("---")

# Dropdowns for categorical features
fuel_types = [c.replace("fuel_", "") for c in model_columns if c.startswith("fuel_")]
seller_types = [c.replace("seller_type_", "") for c in model_columns if c.startswith("seller_type_")]
trans_types = [c.replace("transmission_", "") for c in model_columns if c.startswith("transmission_")]

fuel = st.selectbox(" Fuel Type", fuel_types, key="fuel")
seller_type = st.selectbox(" Seller Type", seller_types, key="seller_type")
transmission = st.selectbox(" Transmission Type", trans_types, key="transmission")

# Prepare input vector
input_data = np.zeros(len(model_columns))
for i, col in enumerate(model_columns):
    if col == 'km_driven':
        input_data[i] = km_driven
    elif col == 'car_age':
        input_data[i] = car_age
    elif col == 'owners':
        input_data[i] = owners
    elif col == f'fuel_{fuel}':
        input_data[i] = 1
    elif col == f'seller_type_{seller_type}':
        input_data[i] = 1
    elif col == f'transmission_{transmission}':
        input_data[i] = 1

if st.button("💰 Predict Price"):
    prediction = model.predict([input_data])[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")
