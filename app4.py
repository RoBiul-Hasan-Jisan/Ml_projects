import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Streamlit page config
st.set_page_config(page_title="Ice Cream Sales Polynomial Regression", layout="centered")
st.title("🍦 Ice Cream Sales Polynomial Regression")

# File paths
DATA_PATH = "data/Ice_cream selling data.csv"
MODEL_PATH = "model/polynomial_model.pkl"
POLY_PATH = "model/poly_features.pkl"

# Check dataset file
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at '{DATA_PATH}'. Please check the path and filename.")
    st.stop()

# Check model and transformer files
if not os.path.exists(MODEL_PATH) or not os.path.exists(POLY_PATH):
    st.error(f"Model or transformer files not found in 'model/' folder.")
    st.stop()

# Load dataset
data = pd.read_csv(DATA_PATH)
st.write("Dataset loaded successfully!")
st.dataframe(data.head())

feature_col = "Temperature (°C)"
target_col = "Ice Cream Sales (units)"

X = data[[feature_col]].values
y = data[target_col].values

# Load model and transformer with error handling
try:
    model = joblib.load(MODEL_PATH)
    poly = joblib.load(POLY_PATH)
    st.success("Model and polynomial transformer loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or transformer: {e}")
    st.stop()

# User input for temperature
temp_input = st.number_input(
    f"Enter {feature_col} to predict sales:",
    float(X.min()),
    float(X.max()),
    step=0.1
)

# Prediction
input_poly = poly.transform(np.array([[temp_input]]))
predicted_sales = model.predict(input_poly)[0]

st.markdown(f"### Predicted Ice Cream Sales at {temp_input}°C: **{predicted_sales:.2f} units**")

# Plot 1: Scatter + Polynomial fit curve
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, color='blue', label='Actual Data')

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

ax1.plot(X_range, y_range_pred, color='red', label='Polynomial Fit')
ax1.set_xlabel(feature_col)
ax1.set_ylabel(target_col)
ax1.set_title('Ice Cream Sales vs Temperature')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1, clear_figure=True)

# Calculate predictions on training data for residuals
y_pred = model.predict(poly.transform(X))
residuals = y - y_pred

# Plot 2: Histogram of Temperature
fig2, ax2 = plt.subplots()
sns.histplot(data[feature_col], kde=True, color='skyblue', ax=ax2)
ax2.set_title(f'Distribution of {feature_col}')
ax2.set_xlabel(feature_col)
ax2.set_ylabel('Frequency')
st.pyplot(fig2, clear_figure=True)

# Plot 3: Histogram of Ice Cream Sales
fig3, ax3 = plt.subplots()
sns.histplot(data[target_col], kde=True, color='salmon', ax=ax3)
ax3.set_title(f'Distribution of {target_col}')
ax3.set_xlabel(target_col)
ax3.set_ylabel('Frequency')
st.pyplot(fig3, clear_figure=True)

# Plot 4: Residual Plot
fig4, ax4 = plt.subplots()
ax4.scatter(X, residuals, color='purple')
ax4.axhline(y=0, color='black', linestyle='--')
ax4.set_title('Residual Plot')
ax4.set_xlabel(feature_col)
ax4.set_ylabel('Residuals (Actual - Predicted)')
ax4.grid(True)
st.pyplot(fig4, clear_figure=True)

# Plot 5: Predicted vs Actual Sales
fig5, ax5 = plt.subplots()
ax5.scatter(y, y_pred, color='green', alpha=0.7)
ax5.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax5.set_title('Predicted vs Actual Sales')
ax5.set_xlabel('Actual Sales')
ax5.set_ylabel('Predicted Sales')
ax5.grid(True)
st.pyplot(fig5, clear_figure=True)

# Plot 6: Residuals over samples
fig6, ax6 = plt.subplots()
ax6.plot(residuals, marker='o', linestyle='-', color='orange')
ax6.set_title('Residuals over Samples')
ax6.set_xlabel('Sample index')
ax6.set_ylabel('Residual')
ax6.grid(True)
st.pyplot(fig6, clear_figure=True)
