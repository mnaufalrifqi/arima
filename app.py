import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from math import sqrt
import altair as alt

# Membuat data untuk chart
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 30, 40, 50]
})

# Membuat chart
chart = alt.Chart(data).mark_line().encode(
    x='x',
    y='y'
)

# Menampilkan chart di Streamlit
st.altair_chart(chart, use_container_width=True)


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

# Menampilkan hasil prediksi dengan Streamlit
st.title('Prediksi Harga Saham BMRI.JK')

# Menampilkan tabel prediksi
st.write("### Tabel Prediksi Harga Saham BMRI.JK:")
st.dataframe(forecast_df)

# Menggunakan date_input untuk memilih tanggal
date_picker = st.date_input(
    "Pilih Tanggal untuk Melihat Prediksi",
    min_value=forecast_df.index.min(),
    max_value=forecast_df.index.max(),
    value=forecast_df.index[0]
)

# Menampilkan harga saham yang diprediksi pada tanggal yang dipilih
selected_prediction = forecast_df.loc[date_picker]
st.write(f"### Prediksi Harga Saham BMRI.JK pada {date_picker}:")
st.write(f"**Prediksi Harga:** {selected_prediction['Forecast']:.2f} IDR")

# Menampilkan grafik prediksi
st.write("### Grafik Prediksi Harga Saham:")
st.line_chart(forecast_df['Forecast'])

