import streamlit as st
import pandas as pd
import pickle

# Load the saved SARIMAX model
with open("sarimax_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to get prediction for a given year and month
def get_prediction(year, month):
    try:
        # Create a DataFrame for the input date
        date_str = f"{year}-{month:02d}"  # Format the date as YYYY-MM
        date_index = pd.to_datetime([date_str])  # Convert to datetime index

        # Generate prediction
        forecast = model.get_prediction(start=date_index[0], end=date_index[0])
        prediction = forecast.predicted_mean.iloc[0]

        return prediction
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.title("SARIMAX Prediction App")

# User inputs
year = st.number_input("Enter the year (e.g., 2024):", min_value=1900, max_value=2100, step=1, format="%d")
month = st.number_input("Enter the month (1-12):", min_value=1, max_value=12, step=1, format="%d")

if st.button("Get Prediction"):
    prediction = get_prediction(year, month)
    st.write(f"The predicted value for {year}-{month:02d} is: {prediction}")
