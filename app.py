import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Simple Linear Regression: Sales Prediction from TV Advertising")

# Load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv(dataset = pd.read_csv('../data/advertising.csv'))
    dataset = dataset.dropna()
    return dataset

data = load_data()

# Show raw data
if st.checkbox("Show raw data"):
    st.write(data.head(10))
    st.write(data.describe())

# Feature and target
X = data[['TV']].values
y = data['Sales'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test results
y_pred = model.predict(X_test)

# Show model performance
st.subheader("Model Performance on Test Set")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
st.write(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

# User input for prediction
st.subheader("Predict Sales from TV Advertising Spend")
tv_spend = st.slider("Select TV Advertising Spend", 
                     float(X.min()), float(X.max()), float(X.mean()))

# Predict sales for input
predicted_sales = model.predict(np.array([[tv_spend]]))[0]
st.write(f"Predicted Sales: **{predicted_sales:.2f}** units")

# Plot training data + regression line
st.subheader("Training Set Visualization")
fig1, ax1 = plt.subplots()
ax1.scatter(X_train, y_train, color='blue', label='Training data')
ax1.plot(X_train, model.predict(X_train), color='red', label='Regression line')
ax1.set_xlabel("TV Advertising Spend")
ax1.set_ylabel("Sales")
ax1.legend()
st.pyplot(fig1)

# Plot test data + regression line
st.subheader("Test Set Visualization")
fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test, color='green', label='Test data')
ax2.plot(X_train, model.predict(X_train), color='red', label='Regression line')
ax2.set_xlabel("TV Advertising Spend")
ax2.set_ylabel("Sales")
ax2.legend()
st.pyplot(fig2)
