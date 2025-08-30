import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

# Preprocess financial data
def preprocessing_yf(tickers, period="1000d"):
    df = yf.download(tickers, period=period, auto_adjust=True)

    if df.empty:
        raise ValueError(f"Aucune donnée reçue pour {tickers}.")

    df.rename(columns=str.lower, inplace=True)
    df.index.name = 'time'
    return df

# Financials indicators
def compute_trading_indicators(df, sma_fast=10, sma_slow=30, rsi_window=10):
    df = df.copy()

    df['SMA_fast'] = df['Close'].rolling(sma_fast).mean()
    df['SMA_slow'] = df['Close'].rolling(sma_slow).mean()

    rsi_calc = RSIIndicator(df['Close'], window=rsi_window)
    df['RSI'] = rsi_calc.rsi()
    df['RSI_yesterday'] = df['RSI'].shift(1)

    # Statisticals conditions to generate first insights
    buy_cond = (df['SMA_fast'] > df['SMA_slow']) & (df['RSI'] < df['RSI_yesterday'])
    sell_cond = (df['SMA_fast'] < df['SMA_slow']) & (df['RSI'] > df['RSI_yesterday'])

    df['Position'] = 0
    df.loc[buy_cond, 'Position'] = 1
    df.loc[sell_cond, 'Position'] = -1

    df['Pct'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Pct'] * df['Position'].shift(1)

    return df


# Candelstick plot with plotly
def plot_stock_chart(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_width=[0.2, 0.7],
                        subplot_titles=(f"{ticker} Price", "Volume"))

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_slow'], line=dict(color='grey'), name="SMA Slow"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_fast'], line=dict(color='blue'), name="SMA Fast"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='orange'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", width=900, height=600)
    return fig

# features for model
def prepare_features(df):
    df = compute_trading_indicators(df)
    df.dropna(subset=['Position_soft'], inplace=True)


    mapping = {-1: 0, 0: 1, 1: 2}
    df['Target'] = df['Position_soft'].map(mapping)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_fast', 'SMA_slow']
    X = df[features].values
    y = df['Target'].values

    # Standarisation
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X, y, df
