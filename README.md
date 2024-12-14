# Accidents_Prediction_AI_Challenge
# SARIMAX Accident Forecasting Application
This repository features a forecasting application developed using SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) to predict alcohol-related accidents in Munich over time. The data is sourced from [Munich Open Data Portal](https://www.muenchen.de/open-data).
  

The main components are:
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
```

## How to Use the Application
### 1. Running the Forecasting Script
You can execute script.py to fetch data, preprocess it, train a SARIMAX model, and save it for future use. The model is saved as sarimax_model.pkl.

Run the script using:

bash
```
python script.py
```

This will:

- Fetch the historical accident data from the API.
- Process the data.
- Train the SARIMAX model on the data.
- Forecast future accident values for 2021.
- Evaluate the model's performance on test data.
- Save the trained model to a .pkl file for use in the Streamlit app.

### 2. Running the Streamlit App
To run the Streamlit app (app.py), first ensure that sarimax_model.pkl is available in the same directory. Then, run the following command:

bash
```
streamlit run app.py
```

This will launch a web interface where you can:

Input a year and month.
Get the forecasted value for that specific month and year based on the trained SARIMAX model.

### 3. Streamlit Interface

Title: The title of the application is "SARIMAX Prediction App".
Year Input: A field to input the year (e.g., 2024).
Month Input: A field to input the month (1-12).
Get Prediction Button: Once you enter the year and month, click this button to get the forecasted value for that month.🎉


Alternatively, have a look at the app here: [Streamlit App](https://accidentspredictionaichallenge-me5khw2rbrj2bfm2srhxa5.streamlit.app)


