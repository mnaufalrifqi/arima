import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# Download data saham BMRI.JK
df = yf.download("BMRI.JK", start="2019-12-01", end="2024-12-01")

# Data cleaning (interpolasi untuk nilai NaN)
df_close = df['Close']
df_close = df_close.interpolate(method='linear')

# Differencing untuk membuat data stasioner
df_diff = df_close.diff().dropna()

# Membangun model ARIMA
model = ARIMA(df_close, order=(2, 1, 2))
model_fit = model.fit()

# Prediksi 30 hari ke depan
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

# Menyiapkan tanggal untuk prediksi
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

# Tambahkan grafik untuk actual vs forecast menggunakan Altair
st.subheader("Grafik Harga Saham BMRI.JK (Prediksi vs Aktual)")

# Siapkan data untuk plotting menggunakan Altair
df_plot = pd.DataFrame({
    'Date': df_close.index,
    'Value': df_close.values,
    'Type': ['Actual'] * len(df_close)
})

forecast_plot = pd.DataFrame({
    'Date': forecast_df.index,
    'Value': forecast_df['Forecast'],
    'Type': ['Forecast'] * len(forecast_df)
})

# Gabungkan data actual dan forecast
df_combined = pd.concat([df_plot, forecast_plot])

# Grafik dengan Altair
chart = alt.Chart(df_combined).mark_line().encode(
    x='Date:T',
    y='Value:Q',
    color='Type:N'
).properties(
    title='BMRI.JK Stock Price Forecast (30 days)',
    width=800,
    height=400
)

# Menampilkan grafik Altair
st.altair_chart(chart)

# Pilihan untuk memilih tanggal tertentu
date_choice = st.date_input("Pilih tanggal prediksi", min_value=forecast_df.index.min().date(), max_value=forecast_df.index.max().date())

# Pastikan date_choice adalah dalam format yang sesuai dan berada dalam rentang prediksi
date_choice = pd.to_datetime(date_choice)  # memastikan format datetime yang benar

# Grafik prediksi vs aktual berdasarkan tanggal pilihan
if date_choice in forecast_df.index:
    st.subheader(f"Grafik Harga Saham BMRI.JK dan Prediksi pada Tanggal {date_choice.date()}")
    
    # Ambil nilai forecast untuk tanggal yang dipilih
    forecast_value = forecast_df.loc[date_choice, 'Forecast']
    st.write(f"Harga Prediksi pada {date_choice.date()}: {forecast_value}")
    
    # Plot grafik menggunakan Altair dengan garis vertikal di tanggal yang dipilih
    selected_plot = df_combined[df_combined['Date'] <= date_choice]
    selected_chart = alt.Chart(selected_plot).mark_line().encode(
        x='Date:T',
        y='Value:Q',
        color='Type:N'
    ).properties(
        title=f'Harga Saham BMRI.JK dan Prediksi pada {date_choice.date()}',
        width=800,
        height=400
    )

    # Menambahkan garis vertikal pada tanggal yang dipilih
    vertical_line = alt.Chart(pd.DataFrame({
        'Date': [date_choice],
        'Value': [forecast_value]
    })).mark_rule(color='red').encode(x='Date:T')

    # Gabungkan grafik dan garis vertikal
    final_chart = selected_chart + vertical_line
    st.altair_chart(final_chart)

    # Hitung dan tampilkan error prediksi
    actual_values = df_close.loc[forecast_df.index]
    mape = mean_absolute_percentage_error(actual_values, forecast_df['Forecast'])
    mse = mean_squared_error(actual_values, forecast_df['Forecast'])
    rmse = sqrt(mse)
    
    st.subheader("Error Prediksi")
    st.write(f"MAPE: {mape * 100:.2f}%")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
else:
    st.error(f"Tanggal {date_choice.date()} tidak ada dalam rentang prediksi.")
