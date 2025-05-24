import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import ta

st.set_page_config(layout="wide")

@st.cache_data

def fetch_binance_ohlcv(symbol='BTC/USDT', timeframe='5m', limit=500):
    exchange = ccxt.binance()
    exchange.load_markets()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

def add_indicators(df):
    donchian_len = 10
    df['donchian_high'] = df['High'].rolling(donchian_len).max()
    df['donchian_low'] = df['Low'].rolling(donchian_len).min()

    rsi_len = 14
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], rsi_len).rsi()

    atr_len = 10
    atr_mult = 2
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_len).average_true_range()
    hl2 = (df['High'] + df['Low']) / 2
    upper = hl2 + atr_mult * atr
    lower = hl2 - atr_mult * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = np.ones(len(df))

    for i in range(1, len(df)):
        final_upper.iloc[i] = max(upper.iloc[i], final_upper.iloc[i-1]) if df['Close'].iloc[i-1] > final_upper.iloc[i-1] else upper.iloc[i]
        final_lower.iloc[i] = min(lower.iloc[i], final_lower.iloc[i-1]) if df['Close'].iloc[i-1] < final_lower.iloc[i-1] else lower.iloc[i]
        if df['Close'].iloc[i] > final_lower.iloc[i-1]:
            direction[i] = 1
        elif df['Close'].iloc[i] < final_upper.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

    df['direction'] = direction
    df['vol_sma20'] = df['Volume'].rolling(20).mean()
    df['valid_volume'] = df['Volume'] > df['vol_sma20'] * 1.2

    macd_indicator = ta.trend.MACD(df['Close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()

    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Support and Resistance levels (using local min/max)
    df['support'] = df['Low'].rolling(window=20).min()
    df['resistance'] = df['High'].rolling(window=20).max()

    # Trendline approximation using linear regression
    df['trendline'] = df['Close'].rolling(window=30).apply(lambda x: np.poly1d(np.polyfit(range(len(x)), x, 1))(len(x)-1), raw=False)

    return df

def generate_signals(df):
    long_entry = (df['Close'] > df['donchian_high']) & \
                 (df['rsi'] < 80) & \
                 (df['direction'] == 1) & \
                 (df['valid_volume']) & \
                 (df['macd'] > df['macd_signal']) & \
                 (df['Close'] > df['ema50'])

    short_entry = (df['Close'] < df['donchian_low']) & \
                  (df['rsi'] > 20) & \
                  (df['direction'] == -1) & \
                  (df['valid_volume']) & \
                  (df['macd'] < df['macd_signal']) & \
                  (df['Close'] < df['ema50'])

    df['signal'] = 0
    df.loc[long_entry, 'signal'] = 1
    df.loc[short_entry, 'signal'] = -1

    # Exit points: Signal reversal or hitting support/resistance
    df['exit'] = 0
    df['exit'] = np.where((df['signal'].shift(1) == 1) & (df['Close'] < df['support']), -1, df['exit'])
    df['exit'] = np.where((df['signal'].shift(1) == -1) & (df['Close'] > df['resistance']), 1, df['exit'])

    return df

def plot_chart(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df['Close'], label='Close Price', color='black')
    ax.plot(df.index, df['donchian_high'], label='Donchian High', linestyle='--', color='green')
    ax.plot(df.index, df['donchian_low'], label='Donchian Low', linestyle='--', color='red')
    ax.plot(df.index, df['ema50'], label='EMA50', color='blue')
    ax.plot(df.index, df['support'], label='Support', linestyle=':', color='cyan')
    ax.plot(df.index, df['resistance'], label='Resistance', linestyle=':', color='magenta')
    ax.plot(df.index, df['trendline'], label='Trendline', linestyle='-.', color='orange')
    ax.scatter(df.index[df['signal'] == 1], df['Close'][df['signal'] == 1], marker='^', color='green', label='Long Signal', s=100)
    ax.scatter(df.index[df['signal'] == -1], df['Close'][df['signal'] == -1], marker='v', color='red', label='Short Signal', s=100)
    ax.scatter(df.index[df['exit'] == 1], df['Close'][df['exit'] == 1], marker='x', color='blue', label='Exit Short', s=100)
    ax.scatter(df.index[df['exit'] == -1], df['Close'][df['exit'] == -1], marker='x', color='purple', label='Exit Long', s=100)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.title("📈 Crypto Signal Dashboard with Trendlines & S/R Levels")

symbol = st.text_input("Enter trading pair (e.g., BTC/USDT):", value="BTC/USDT")
timeframe = st.selectbox("Select timeframe:", options=["1m", "5m", "15m", "1h", "4h"], index=2)

if st.button("🔍 Run Analysis"):
    try:
        df = fetch_binance_ohlcv(symbol, timeframe)
        df = add_indicators(df)
        df = generate_signals(df)

        st.subheader("📊 Latest Signals:")
        st.dataframe(df[['Close', 'signal', 'exit']].tail(20))

        st.subheader("📈 Chart:")
        plot_chart(df)
    except Exception as e:
        st.error(f"Error: {str(e)}")
