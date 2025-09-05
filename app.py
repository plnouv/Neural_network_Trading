import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from data_loader import load_from_yfinance, load_from_csv, prepare_features
from neural_network import PerceptronSoftmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.markdown("""
    <style>
    /* === Central color === */
    .main {
        background-color: #1e1e1e;
        color: white;
        padding: 2rem;
        border-radius: 10px;
    }

    /* === BACKGROUND IMAGE SIDEBAR === */
    [data-testid="stSidebar"] {
        background-image: url('finance_background.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    /* === BOUTONS JAUNES === */
    .stButton > button {
        background-color: #ffd700;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid black;
    }

    .stButton > button:hover {
        background-color: #ffcc00;
        color: black;
        border: 1px solid #555;
    }
    </style>
""", unsafe_allow_html=True)

# APP COMFING STRML
st.set_page_config(page_title="Neural Network Trading App", layout="wide")
st.title("📊 Neural Network Trading Signals (BUY / SELL / HOLD)")

for key in ["df", "df_feat", "X", "y", "X_train", "X_test", "y_train", "y_test", "model", "ticker_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Backround IMG
with st.sidebar:
    st.image("finance_background.png", use_container_width=True)
    st.header("Load Data")
    data_source = st.selectbox("Data Source:", ["Yahoo Finance", "Local CSV"])

# FROM Y FINANCE
if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", value="AAPL")
    if st.sidebar.button("Load Data"):
        try:
            st.session_state.df = load_from_yfinance(ticker)
            st.session_state.ticker_name = ticker

            company_name = yf.Ticker(ticker).info.get("longName", "")
            if company_name:
                st.success(f"Data for {company_name} ({ticker}) loaded successfully!")
            else:
                st.success(f"Data for {ticker} loaded successfully! (Unknown name)")
        except Exception as e:
            st.error(f"Error loading data: {e}")
# LOCAL CSV
elif data_source == "Local CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = load_from_csv(uploaded_file)
        st.session_state.ticker_name = os.path.splitext(uploaded_file.name)[0]
        st.success("CSV loaded successfully!")

# PREVIEW
df = st.session_state.df

if df is not None:
    df_sorted = df.sort_index(ascending=True)

    st.subheader("Raw Data Preview")
    st.dataframe(df_sorted.head(20))
    st.write(f"**Total rows: {len(df_sorted)}**")

    ticker_name = st.session_state.get("ticker_name", "data")
    csv = df_sorted.to_csv(index=True).encode('utf-8')
    st.download_button("Save DATA as CSV file", csv, f"{ticker_name}_full.csv", "text/csv")

    # FEATURE 
    X, y, df_feat = prepare_features(df_sorted)

    if df_feat is None or len(df_feat) == 0:
        st.error("No usable rows after feature preparation. "
                 "Try a longer CSV (>= 100 rows) and ensure columns: Open, High, Low, Close, Volume.")
        st.stop()

    # SPLIT 
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    st.session_state.update({
        "X": X, "y": y, "df_feat": df_feat,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test
    })

    # TECH METRICS 
    st.subheader(f"Technical Metrics for {ticker_name} (Last 10 days)")
    st.markdown("""
    <div style="font-size:16px; font-weight:bold; color:#333; padding:5px 0;">
    Positions:&nbsp;&nbsp;&nbsp;
    <span style="color:red;">SELL (0)</span>&nbsp;&nbsp;&nbsp;
    <span style="color:orange;">HOLD (1)</span>&nbsp;&nbsp;&nbsp;
    <span style="color:green;">BUY (2)</span>
    </div>
    """, unsafe_allow_html=True)

    df_recent = df_feat.sort_index(ascending=False).head(10)
    df_recent_display = df_recent.copy()
    if 'Position' in df_recent_display.columns:
        df_recent_display['Position'] = df_recent_display['Position'].map({-1: 0, 0: 1, 1: 2})

    st.dataframe(df_recent_display[['Close', 'SMA_fast', 'SMA_slow', 'RSI', 'Position']])

    # CANDLESTICK
    st.subheader("📈 Interactive Candlestick Chart with SMA")
    if df_feat is not None and len(df_feat) > 0:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=(f"{ticker_name} Price", "Volume"), row_width=[0.2, 0.7]
        )
        fig.add_trace(go.Candlestick(x=df_feat.index, open=df_feat['Open'], high=df_feat['High'],
                                     low=df_feat['Low'], close=df_feat['Close'], name="OHLC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_slow'], line=dict(color='grey'), name="SMA Slow"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_fast'], line=dict(color='lightblue'), name="SMA Fast"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_feat.index, y=df_feat['Volume'], marker_color='red', showlegend=False), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", width=900, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart unavailable")

    # TRAIN MODEL
    if st.button(" Train the model (Optimized)"):
        with st.spinner("Optimizing model..."):
            learning_rates = [0.01, 0.05, 0.1]
            epochs_list = [50, 100, 200]

            best_acc = 0
            best_model = None
            best_params = None

            for lr in learning_rates:
                for epochs in epochs_list:
                    try:
                        model = PerceptronSoftmax(learning_rate=lr, n_iter=epochs)
                        model.fit(X_train, y_train)
                        acc = model.score(X_test, y_test)

                        if acc > best_acc:
                            best_acc = acc
                            best_model = model
                            best_params = (lr, epochs)
                    except Exception as e:
                        st.warning(f"Error with lr={lr}, epochs={epochs}: {e}")

            if best_model is not None:
                st.session_state.model = best_model
                st.success(" Model successfully trained!")
            else:
                st.error(" Model training failed. Check parameters or data.")

# MODEL RESULTS
if st.session_state.model is not None:
    model = st.session_state.model
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    X_all = st.session_state.X
    df_feat = st.session_state.df_feat

    train_acc = model.score(X_train, y_train) if X_train is not None and len(X_train) > 0 else 0
    test_acc = model.score(X_test, y_test) if X_test is not None and len(X_test) > 0 else 0

    if X_all is not None and len(X_all) > 0:
        try:
            last_signal = int(model.predict(X_all[-1].reshape(1, -1))[0])
            signal_txt = {2: "BUY ", 0: "SELL ", 1: "HOLD "}.get(last_signal, "HOLD ")
        except Exception as e:
            signal_txt = f"N/A (prediction error: {e})"
    else:
        signal_txt = "N/A (not enough data)"

    # RSI / SMA
    rule_signal_txt, last_date_str = "N/A", "N/A"
    if df_feat is not None and len(df_feat) > 0 and "Position" in df_feat.columns:
        last_idx = df_feat.index[-1]
        last_pos_val = df_feat.loc[last_idx, "Position"]

    # MAPPING
        rule_map = {
            -1: "SELL ",
            0:  "HOLD ",
            1:  "BUY ",
        }
        try:
            rule_signal_txt = rule_map.get(int(last_pos_val), "HOLD ")
        except Exception:
            rule_signal_txt = "HOLD "

        try:
            last_date_str = pd.to_datetime(last_idx).strftime("%Y-%m-%d")
        except Exception:
            last_date_str = str(last_idx)


    st.markdown("---")
    st.markdown("### Model Summary")
    st.markdown(f"""
    <div style="background-color:#f0f8ff;padding:20px;border-radius:10px">
    <ul style="list-style-type:none;padding-left:10px;font-size:16px">
      <li><b>Status:</b> Model successfully trained</li>
      <li><b>Best parameters:</b> Learning rate = <code>{model.learning_rate:.3f}</code>, Epochs = <code>{model.n_iter}</code></li>
      <li><b>Accuracy (Train):</b> <span style="color:green;font-weight:bold;">{train_acc * 100:.2f}%</span></li>
      <li><b>Accuracy (Test):</b> <span style="color:blue;font-weight:bold;">{test_acc * 100:.2f}%</span></li>
      <li><b>Next-day Signal (statistical: RSI & MAs, based on {last_date_str}):</b>
          <span style="font-weight:bold;">{rule_signal_txt}</span></li>
      <li><b>Tomorrow's Prediction (model):</b> <span style="font-weight:bold;">{signal_txt}</span></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # CONFUSION MATRIX 
    if X_test is not None and len(X_test) > 0 and y_test is not None and len(y_test) > 0:
        st.subheader("Confusion Matrix (Test Set)")
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(3, 3)) 
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["SELL", "HOLD", "BUY"])
            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
            ax_cm.set_title("Confusion Matrix", fontsize=12)
            st.pyplot(fig_cm)

            
            labels = ["SELL", "HOLD", "BUY"]
            total_by_class = cm.sum(axis=1)
            true_positives = np.diag(cm)
            false_negatives = total_by_class - true_positives
            precisions = 100 * true_positives / (total_by_class + 1e-10)

            st.markdown("###  Prediction breakdown by class")
            for i, label in enumerate(labels):
                st.markdown(f"""
                <div style="background-color:#f8f9fa;padding:10px 20px;margin-bottom:10px;border-radius:8px">
                    <b>{label}</b>: Good predictions = <b>{true_positives[i]}</b>,
                    Wrong predictions = <b>{false_negatives[i]}</b>,
                    Accuracy = <b>{precisions[i]:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating confusion matrix: {e}")
    else:
        st.info("Not enough data to display the confusion matrix.")

    # LEARNING CURVE
    st.subheader("📈 Log-Loss Curve")
    if hasattr(model, 'losses') and model.losses is not None and len(model.losses) > 0:
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.losses, label="Log-loss", color='red')
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Num Epochs")
        ax_loss.legend()
        st.pyplot(fig_loss)
    else:
        st.info("No loss curve available for this model.")
else:
    st.warning("Please load the data and train the model to see results.")
