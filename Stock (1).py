# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf

# ===== READ CSV FILE =====
df = pd.read_csv("dataset.csv")

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preview data
print("\n=== DATA PREVIEW ===")
print(df.head())

# ===== RE-INDEX WITH DAILY RANGE =====
new_idx = pd.date_range(start="2008-05-26", end="2021-04-30", freq="D")
df = df.reindex(new_idx)

# ===== SET SEABORN THEME =====
sb.set_theme()
sb.set(rc={'figure.figsize':(15,8)})

# ===== DAILY TRADING VOLUME (Plotly PNG) =====
fig = px.line(df, x=df.index, y="Volume", title="Daily Trading Volume")
fig.write_image("Daily_Trading_Volume.png")  # Requires 'kaleido'

# ===== DAILY HIGH PRICE (Plotly PNG) =====
fig = px.line(df, x=df.index, y="High", title="Daily High Price")
fig.write_image("Daily_High_Price.png")  # Requires 'kaleido'

# ===== SIMPLE MOVING AVERAGES ON VWAP (Matplotlib PNG) =====
df_sma = df.copy()
df_sma['SMA_10'] = df_sma['VWAP'].rolling(10, min_periods=1).mean()
df_sma['SMA_20'] = df_sma['VWAP'].rolling(20, min_periods=1).mean()

plt.figure(figsize=(15,8))
plt.plot(df_sma.index, df_sma['VWAP'], color='blue', label='VWAP')
plt.plot(df_sma.index, df_sma['SMA_10'], color='red', label='10-Day SMA')
plt.plot(df_sma.index, df_sma['SMA_20'], color='green', label='20-Day SMA')
plt.title("VWAP with 10 & 20 Day Moving Averages")
plt.legend()
plt.savefig("VWAP_Moving_Averages.png")
plt.close()

# ===== HANDLE MISSING VWAP VALUES & PLOT AUTOCORRELATION (Matplotlib PNG) =====
df['VWAP'] = df['VWAP'].interpolate(method='linear')
plt.figure(figsize=(15,8))
plot_acf(df['VWAP'], lags=50)
plt.title("Autocorrelation of VWAP")
plt.savefig("VWAP_Autocorrelation.png")
plt.close()

# ===== ADD DAY, MONTH, YEAR COLUMNS =====
df_temp = df.copy()
df_temp['day'] = df_temp.index.day
df_temp['month'] = df_temp.index.month
df_temp['year'] = df_temp.index.year

# ===== MONTHLY AVERAGES (NUMERIC ONLY) =====
numeric_cols = df_temp.select_dtypes(include=np.number)
df_m = numeric_cols.groupby([df_temp['month'], df_temp['year']]).mean()

# ===== CREATE HEATMAP OF VWAP (Matplotlib PNG) =====
heatmap_data = df_temp.pivot_table(values='VWAP', index='month', columns='year', aggfunc='mean')

plt.figure(figsize=(15,8))
sb.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Monthly Average VWAP Heatmap")
plt.ylabel("Month")
plt.xlabel("Year")
plt.savefig("VWAP_Heatmap.png")
plt.close()


