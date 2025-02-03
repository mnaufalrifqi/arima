import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
import statsmodels.tools.tools as sm
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

warnings.filterwarnings("ignore")

df = yf.download("BMRI.JK", start="2019-12-01", end="2024-12-01")

df_info = df.info()

"""nih datanya cleaning"""

df_close = df['Close']
df_close = df_close.interpolate(method='linear')

df_close.index = pd.to_datetime(df_close.index)

"""datanya nyesuain dari adj close"""

st.title('Harga Penutupan Saham BMRI')
st.line_chart(df_close)

adf_test = adfuller(df_close)
adf_statistic = adf_test[0]
p_value = adf_test[1]
critical_values = adf_test[4]
st.write(f"ADF Statistic: {adf_statistic}")
st.write(f"P-value: {p_value}")
st.write("Critical Values:")
for key, value in critical_values.items():
    st.write(f"{key}: {value}")
if p_value < 0.05:
    st.write("Data stasioner")
else:
    st.write("Data tidak stasioner")

# lag 1
df_diff = df_close.diff().dropna()
adf_test_diff = adfuller(df_diff)

st.write(f"ADF Statistic (Differenced Data): {adf_test_diff[0]}")
st.write(f"P-value (Differenced Data): {adf_test_diff[1]}")

if adf_test_diff[1] < 0.05:
    st.write("Data yang telah di-differencing stasioner")
else:
    st.write("Data yang telah di-differencing masih tidak stasioner")

plt.figure(figsize=(10, 6))
plt.plot(df_diff)
plt.title('Harga Adjusted Close (Setelah Differencing)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Adjusted Close')
plt.grid(True)
st.pyplot()

model = ARIMA(df_close, order=(2, 1, 2))
model_fit = model.fit()

residuals = model_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residual ARIMA')
plt.show()
st.pyplot()

