import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle

# Function to fetch data from API
def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['result']['records']
        return pd.DataFrame(data)
    else:
        raise Exception("Failed to fetch data from API")

# Function to preprocess data
def preprocess_data(df):
    df.rename(columns={
        '_id': 'ID',
        'MONATSZAHL': 'Category',
        'AUSPRAEGUNG': 'Accident-type',
        'JAHR': 'Year',
        'MONAT': 'Month',
        'WERT': 'Value',
        'VORJAHRESWERT': 'Previous_Year_Value',
        'VERAEND_VORMONAT_PROZENT': 'Change_From_Previous_Month_Percent',
        'VERAEND_VORJAHRESMONAT_PROZENT': 'Change_From_Same_Month_Previous_Year_Percent',
        'ZWOELF_MONATE_MITTELWERT': '12_Month_Moving_Average'
    }, inplace=True)

    # Filter relevant columns
    df = df[['Category', 'Accident-type', 'Year', 'Month', 'Value']]

    # Handle missing and invalid values in the 'Month' column
    df['Month'] = df['Month'].replace('', np.nan)
    df = df.dropna(subset=['Month'])
    df = df[df['Month'].astype(str).str.isdigit()]

    # Extract Year and Month
    df['Year'] = df['Month'].astype(str).str[:4].astype(int)
    df['Month'] = df['Month'].astype(str).str[4:].astype(int)

    # Create a datetime index
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

    # Filter data
    df = df[df['Year'] <= 2020]
    df = df[(df['Category'] == 'Alkoholunfälle') & (df['Accident-type'] == 'insgesamt')]
    df.set_index('Date', inplace=True)
    df = df['Value'].resample('M').sum()

    return df

# Function to train SARIMAX model
def train_sarimax(train_data, order, seasonal_order):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    return results

# Function to forecast using SARIMAX
def forecast_sarimax(model, steps):
    forecast = model.get_forecast(steps=steps)
    forecast_df = forecast.summary_frame()
    return forecast_df

# Function to visualize data and forecast
def visualize_forecast(train_data, forecast_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label="Historical Data")
    plt.plot(forecast_df['mean'], label="Forecast", color='orange')
    plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='orange', alpha=0.2)
    plt.legend()
    plt.title(title)
    plt.show()

# Function to evaluate the model
def evaluate_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    return mse, mae

# Function to save the trained model
def save_model(model, filename):
    with open(filename, "wb") as model_file:
        pickle.dump(model, model_file)

# Main execution
def main():
    # API URL
    url_link = 'https://opendata.muenchen.de/api/3/action/datastore_search?resource_id=40094bd6-f82d-4979-949b-26c8dc00b9a7'

    # Fetch data
    raw_data = fetch_data(url_link)

    # Preprocess data
    df = preprocess_data(raw_data)

    # Split data into training set
    train = df[:'2020']

    # Train SARIMAX model
    sarimax_model = train_sarimax(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))

    # Forecast for 2021
    forecast_df = forecast_sarimax(sarimax_model, steps=12)

    # Visualize forecast
    visualize_forecast(train, forecast_df, "Accident Forecast for 'Alkoholunfälle'")

    # Evaluate model on 2020 data
    actual_data = df['2020':]
    predicted_data = forecast_df['mean']
    evaluate_model(actual_data, predicted_data)

    # Save the trained model
    save_model(sarimax_model, "sarimax_model.pkl")

if __name__ == "__main__":
    main()
