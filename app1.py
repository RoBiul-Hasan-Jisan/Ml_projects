import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("House Price Prediction App with Dynamic Feature Selection")

@st.cache_data
def load_data():
    dataset = pd.read_csv(dataset = pd.read_csv('../data/HousingData.csv'))
    dataset = dataset.dropna()
    return dataset

data = load_data()

if st.checkbox("Show raw data"):
    st.write(data.head(10))
    st.write(data.describe())

# Select feature and target columns interactively
all_columns = data.columns.tolist()

feature_col = st.selectbox("Select Feature column (independent variable)", options=all_columns)
target_col = st.selectbox("Select Target column (dependent variable)", options=all_columns)

# Ensure feature and target are not the same
if feature_col == target_col:
    st.error("Feature and Target columns must be different!")
else:
    X = data[[feature_col]].values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("Model Performance on Test Set")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.3f}")

    st.subheader(f"Predict {target_col} from {feature_col}")

    min_val = float(X.min())
    max_val = float(X.max())
    mean_val = float(X.mean())

    input_val = st.slider(f"Select {feature_col}", min_val, max_val, mean_val)

    predicted_val = model.predict(np.array([[input_val]]))[0]
    st.write(f"Predicted {target_col}: **{predicted_val:.2f}**")

    st.subheader("Training Set Visualization")
    fig1, ax1 = plt.subplots()
    ax1.scatter(X_train, y_train, color='blue', label='Training data')
    ax1.plot(X_train, model.predict(X_train), color='red', label='Regression line')
    ax1.set_xlabel(feature_col)
    ax1.set_ylabel(target_col)
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Test Set Visualization")
    fig2, ax2 = plt.subplots()
    ax2.scatter(X_test, y_test, color='green', label='Test data')
    ax2.plot(X_train, model.predict(X_train), color='red', label='Regression line')
    ax2.set_xlabel(feature_col)
    ax2.set_ylabel(target_col)
    ax2.legend()
    st.pyplot(fig2)
