import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from math import sqrt
import altair as alt
import statsmodels.tools.tools as sm
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# Download data saham BMRI
df = yf.download("BMRI.JK", start="2019-12-01", end="2024-12-01")

# Memilih kolom Close dan mengisi nilai yang hilang
df_close = df['Close']
df_close = df_close.interpolate(method='linear')

# Membuat model ARIMA
model = ARIMA(df_close, order=(2, 1, 2))
model_fit = model.fit()

# Prediksi harga saham untuk 30 hari ke depan
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
last_date = df_close.index[-1]
forecast_index = pd.date_range(start=last_date + pd.DateOffset(1), periods=forecast_steps)
forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
forecast_df = forecast_df.set_index('Date')

# Streamlit layout
st.title("Prediksi Harga Saham BMRI.JK")
st.write("Menggunakan model ARIMA untuk memprediksi harga saham BMRI.JK selama 30 hari ke depan.")

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi Harga Saham BMRI.JK (30 Hari ke Depan)")
st.write(forecast_df)

# Pilihan untuk memilih tanggal tertentu
date_choice = st.date_input("Pilih tanggal prediksi", min_value=forecast_df.index.min(), max_value=forecast_df.index.max())

# Grafik prediksi vs aktual berdasarkan tanggal pilihan
if date_choice:
    st.subheader(f"Grafik Harga Saham BMRI.JK dan Prediksi pada Tanggal {date_choice}")
    
    # Filter data berdasarkan tanggal yang dipilih
    forecast_value = forecast_df.loc[date_choice, 'Forecast']
    actual_value = df_close[date_choice] if date_choice in df_close else None
    
    if actual_value is not None:
        st.write(f"Harga Aktual: {actual_value}")
    
    # Plot grafik
    plt.figure(figsize=(12, 6))
    plt.plot(df_close, label='Actual')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='orange')
    plt.axvline(x=date_choice, color='red', linestyle='--', label=f'Selected Date: {date_choice}')
    plt.title('Harga Saham BMRI.JK - Prediksi vs Aktual')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Penutupan')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Hitung dan tampilkan error prediksi
    actual_values = df_close.loc[forecast_df.index]
    mape = mean_absolute_percentage_error(actual_values, forecast_df['Forecast'])
    mse = mean_squared_error(actual_values, forecast_df['Forecast'])
    rmse = sqrt(mse)
