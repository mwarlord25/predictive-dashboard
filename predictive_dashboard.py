import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import ta

st.set_page_config(layout="wide")
st.title("📈 Predictive Trading Agent Dashboard")

# Sidebar inputs
st.sidebar.header("🔍 Select Parameters")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, EURUSD=X)", value="AAPL")
interval = st.sidebar.selectbox("Interval", options=["1m", "5m", "15m", "30m", "1h", "1d"], index=1)
period = st.sidebar.selectbox("Period", options=["1d", "2d", "5d", "7d", "1mo"], index=1)

@st.cache_data(ttl=300)
def load_data(symbol, interval, period):
    df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
    df.dropna(inplace=True)
    return df

df = load_data(symbol, interval, period)

if df.empty:
    st.error("Failed to load data. Please check the symbol or try again later.")
else:
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    df['Signal'] = 0
    df.loc[(df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1)), 'Signal'] = 1
    df.loc[(df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1)), 'Signal'] = -1

    df.dropna(inplace=True)

    features = ['SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'BB_width']
    X = df[features]
    y = df['Signal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    df.loc[X_test.index, 'Predicted_Signal'] = y_pred

    latest_price = df['Close'].iloc[-1]
    st.metric(label=f"Latest Price for {symbol}", value=f"${latest_price:.2f}")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['Close'], label='Close', alpha=0.6)
    ax.plot(df['SMA20'], label='SMA20', linestyle='--')
    ax.plot(df['SMA50'], label='SMA50', linestyle='--')
    ax.fill_between(df.index, df['BB_upper'], df['BB_lower'], color='gray', alpha=0.1, label='Bollinger Bands')

    buy_signals = df[df['Predicted_Signal'] == 1]
    sell_signals = df[df['Predicted_Signal'] == -1]
    ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Predicted Buy', alpha=0.8)
    ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Predicted Sell', alpha=0.8)

    ax.set_title(f"{symbol} Price with Predicted Buy/Sell Signals")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📋 Recent Predictions")
    st.dataframe(df[['Close', 'SMA20', 'SMA50', 'RSI', 'MACD', 'MACD_signal', 'Predicted_Signal']].tail(10))
