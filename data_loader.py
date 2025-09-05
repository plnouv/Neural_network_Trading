import pandas as pd
from metrics import compute_trading_indicators
import yfinance as yf

MAX_ROWS = 1500
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

def strip(s: str) -> str:
    return str(s).strip().replace(" ", "").replace("_", "").lower()

def normalize_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    word_modif = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'adjclose': 'Close',   
        'volume': 'Volume',
    }

    existing_norm = {strip(c): c for c in df.columns}
    rename_map = {}

    for req in REQUIRED_COLS:
        key = strip(req)  
        if key in existing_norm:
            rename_map[existing_norm[key]] = word_modif[key]
            continue

        for k, target in word_modif.items():
            if target == req and k in existing_norm:
                rename_map[existing_norm[k]] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain: {REQUIRED_COLS}. Columns found: {list(df.columns)}")

    return df

def datetime_and_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    date_candidates = ['date', 'datetime', 'timestamp', 'time']
    cols_norm = {strip(c): c for c in df.columns}
    found = None
    for cand in date_candidates:
        if cand in cols_norm:
            found = cols_norm[cand]
            break

    if found is not None:
        df[found] = pd.to_datetime(df[found], errors='not_found')
        df = df[df[found].notna()].copy()
        if found != 'Date':
            df = df.rename(columns={found: 'Date'})
            found = 'Date'
        df = df.set_index(found, drop=False)
    else:
        
        if isinstance(df.index, pd.DatetimeIndex):
            if 'Date' not in df.columns:
                df = df.copy()
                df['Date'] = df.index

    return df

# Download from Yfinance
def load_from_yfinance(tickers, period="1500d"):
    print(f"[INFO] Downloading last {period} for {tickers}...")
    df = yf.download(tickers, period=period, auto_adjust=True)

    if df.empty:
        raise ValueError("No data received from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        first_ticker = df.columns.levels[1][0]
        df = df.xs(first_ticker, axis=1, level=1)
    df = normalize_and_rename(df)

    #  datetime index and date column
    df = datetime_and_date_column(df)
    df = df[[c for c in REQUIRED_COLS if c in df.columns] + (['Date'] if 'Date' in df.columns else [])]
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)

    df = df.sort_index(ascending=True)
    print(f"[INFO] Data downloaded: {len(df)} rows, columns: {list(df.columns)}")
    return df

# Load from local CSV
def load_from_csv(filepath):
    print(f"[INFO] Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    df = normalize_and_rename(df)

    df = datetime_and_date_column(df)
    keep_cols = REQUIRED_COLS + (['Date'] if 'Date' in df.columns else [])
    df = df[keep_cols]

    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS)

    print(f"[INFO] CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df

# Features for Softmax algorithm
def prepare_features(df):
    df = df.copy()

    df = compute_trading_indicators(df, sma_fast=10, sma_slow=30, rsi_window=10)
    df.dropna(subset=['SMA_fast', 'SMA_slow'], inplace=True)

    # relaxed cond
    df['Signal'] = 0  
    df.loc[df['SMA_fast'] > df['SMA_slow'] * 1.005, 'Signal'] = 1  
    df.loc[df['SMA_fast'] < df['SMA_slow'] * 0.995, 'Signal'] = -1  

    # Mapping + stand 
    mapping = {-1: 0, 0: 1, 1: 2}
    df['Target'] = df['Signal'].map(mapping)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_fast', 'SMA_slow']
    X = df[features].values
    y = df['Target'].values

    X = (X - X.mean(axis=0)) / (X.std(axis=0))
    print(f"[INFO] Features ready: {X.shape[0]} samples, {X.shape[1]} variables")
    return X, y, df
