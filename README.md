# Accidents_Prediction_AI_Challenge
# SARIMAX Accident Forecasting Application

This repository contains a forecasting application built using SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) to predict alcohol-related accidents over time in Munich. The main components are:

1. **Data Collection**: The script fetches historical accident data from an API.
2. **Data Preprocessing**: Cleans and processes the data to be suitable for forecasting.
3. **Modeling**: A SARIMAX model is trained to forecast future accident values.
4. **Visualization**: Visualizations of historical and forecasted data are generated.
5. **Evaluation**: The model's performance is evaluated using metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).
6. **Streamlit Application**: A simple web app to allow users to get predictions for any future year and month.

## Files in this Repository

### 1. `script.py`
This script performs the following:
- Fetches accident data from an open API.
- Processes and prepares the data for modeling.
- Trains a SARIMAX model on the data.
- Forecasts accident values for 2021.
- Evaluates the model's performance on test data.
- Saves the trained SARIMAX model to a file.

### 2. `app.py`
A simple Streamlit application to interact with the trained SARIMAX model. It allows users to input a specific year and month and get the corresponding forecasted value.

### 3. `sarimax_model.pkl`
This is the saved trained SARIMAX model from the `script.py`.

## Prerequisites

To run this application, you need to have the following Python packages mentioned in the requirements.txt
You can install using:

```bash
pip install -r requirements.txt
