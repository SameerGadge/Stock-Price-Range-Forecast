import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.deep_models import DeepQuantileModel # <--- NEW IMPORT
from src.strategy import SignalGenerator
from src.utils import evaluate_metrics
from src.sentiment import NewsSentiment 

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Stock Forecaster", layout="wide", page_icon="📈")

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state['history'] = []
if 'selected_ticker' not in st.session_state: st.session_state['selected_ticker'] = "RELIANCE.NS"

def set_ticker(ticker): st.session_state['selected_ticker'] = ticker

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 20px; border-radius: 10px; text-align: center; }
    div[data-testid="column"] { width: fit-content !important; flex: 1 1 auto; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")

# 1. MODEL SELECTION TOGGLE (NEW)
st.sidebar.subheader("🧠 AI Engine")
MODEL_TYPE = st.sidebar.radio(
    "Choose Model Architecture:",
    ("LightGBM (Fast & Stable)", "LSTM Deep Learning (Experimental)"),
    index=0
)

# History
if st.session_state['history']:
    st.sidebar.subheader("🕒 Recent")
    for stock in reversed(st.session_state['history']):
        if st.sidebar.button(stock, key=f"hist_{stock}"):
            set_ticker(stock)
            st.rerun()

TICKER = st.sidebar.text_input("Ticker Symbol", value=st.session_state['selected_ticker']).upper()
HORIZON = st.sidebar.selectbox("Horizon", [5, 21, 60], format_func=lambda x: f"{x} Days")
CONFIDENCE = st.sidebar.slider("Confidence", 0.70, 0.99, 0.90)
SPLIT_RATIO = 0.80

if TICKER.endswith((".NS", ".BO")): CURRENCY = "₹"
else: CURRENCY = "$"

# ... (Keep Sentiment Code Here) ...

st.title(f"📈 {TICKER} Price Interval Forecasting")
st.markdown(f"**Engine:** {MODEL_TYPE} • {int(CONFIDENCE*100)}% Confidence Interval")

# --- MAIN APP LOGIC ---

if st.button("🚀 Run Forecast Model"):
    with st.spinner(f"Training {MODEL_TYPE} on {TICKER}..."):
        try:
            # 1. Data Pipeline
            loader = DataLoader(TICKER)
            raw_df = loader.fetch_data()
            
            if raw_df is None or len(raw_df) < 200:
                st.error("❌ Insufficient data."); st.stop()

            # Update History
            if TICKER in st.session_state['history']: st.session_state['history'].remove(TICKER)
            st.session_state['history'].append(TICKER)

            # 2. Features
            fe = FeatureEngineer(raw_df)
            fe.add_technical_indicators()
            df = fe.create_targets(HORIZON)

            feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']
            X = df[feature_cols]; y = df['Target_Return']
            split_idx = int(len(X) * SPLIT_RATIO)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # 3. MODEL SWITCHING LOGIC (NEW)
            if "LightGBM" in MODEL_TYPE:
                # --- LightGBM Path ---
                alpha_lower = (1 - CONFIDENCE) / 2
                qm = QuantileModels(alpha_lower, 1 - alpha_lower)
                m_low, m_high = qm.train_lgbm(X_train, y_train)
                
                # Predictions
                last_row = fe.df.iloc[[-1]][feature_cols]
                f_low = m_low.predict(last_row)[0]
                f_high = m_high.predict(last_row)[0]
                
                # Backtest Predictions
                pred_low = m_low.predict(X_test)
                pred_high = m_high.predict(X_test)

            else:
                # --- LSTM Path (Deep Learning) ---
                dl = DeepQuantileModel(input_shape=(1, len(feature_cols)))
                dl.train(X_train, y_train, epochs=20) # Fast training for demo
                
                # Predictions (DeepQuantileModel handles reshaping internally)
                last_row = fe.df.iloc[[-1]][feature_cols]
                f_low, f_high = dl.predict(last_row)
                f_low = f_low[0]; f_high = f_high[0] # Extract scalar
                
                # Backtest Predictions
                pred_low, pred_high = dl.predict(X_test)

            # 4. Calibration & Display (Shared Logic)
            calib_factor = 0.6 if HORIZON == 5 else (1.0 if HORIZON == 21 else 1.5)
            
            latest_price = raw_df['Close'].iloc[-1]
            latest_date = raw_df.index[-1]
            future_date = latest_date + pd.Timedelta(days=HORIZON)

            center = (f_high + f_low) / 2
            width = f_high - f_low
            f_low = center - (width * calib_factor)
            f_high = center + (width * calib_factor)
            
            p_low = latest_price * (1 + f_low)
            p_high = latest_price * (1 + f_high)

            # ... (Display Logic & Charts - SAME AS BEFORE) ...
            
            st.markdown("---")
            st.subheader(f"🔮 Future Forecast (Target: {future_date.date()})")
            c1, c2, c3 = st.columns(3)
            c1.metric("📉 Bearish Limit", f"{CURRENCY}{p_low:.2f}", delta=f"{f_low*100:.2f}%", delta_color="inverse")
            c2.metric("📍 Current Price", f"{CURRENCY}{latest_price:.2f}")
            c3.metric("📈 Bullish Limit", f"{CURRENCY}{p_high:.2f}", delta=f"{f_high*100:.2f}%")
            st.markdown("---")
            
            # Backtest Visualization (Shared)
            st.subheader("📊 Historical Backtest")
            
            ctr = (pred_high + pred_low)/2; w = pred_high - pred_low
            pred_low = ctr - (w*calib_factor); pred_high = ctr + (w*calib_factor)
            
            current_prices = df['Close'].iloc[split_idx:].values 
            pl_price = current_prices * (1+pred_low)
            ph_price = current_prices * (1+pred_high)
            y_price = current_prices * (1+y_test.values)
            
            strat = SignalGenerator()
            _, signals, pnl = strat.run_mean_reversion(df.index[split_idx:], y_price, pl_price, ph_price)
            
            k1, k2, k3, k4 = st.columns(4)
            picp = ((y_price >= pl_price) & (y_price <= ph_price)).mean()
            k1.metric("Coverage", f"{picp:.1%}")
            k2.metric("Avg Width", f"{CURRENCY}{(ph_price-pl_price).mean():.2f}")
            k3.metric("PnL", f"{pnl:.2f}")
            k4.metric("Trades", f"{len(signals)}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[split_idx:], y=ph_price, fill=None, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index[split_idx:], y=pl_price, fill='tonexty', fillcolor='rgba(0, 100, 255, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Confidence'))
            fig.add_trace(go.Scatter(x=df.index[split_idx:], y=y_price, mode='lines', name='Actual', line=dict(color='#00F0FF')))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")