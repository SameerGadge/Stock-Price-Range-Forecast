import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.deep_models import DeepQuantileModel
from src.strategy import SignalGenerator
from src.utils import evaluate_metrics

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

# Model Selection
st.sidebar.subheader("🧠 AI Engine")
MODEL_TYPE = st.sidebar.radio(
    "Choose Model Architecture:",
    ("LightGBM (Fast & Stable)", "LSTM Deep Learning (Experimental)", "Ensemble (Best Accuracy)"),
    index=2
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
            if len(st.session_state['history']) > 5: st.session_state['history'].pop(0)

            # 2. Features
            fe = FeatureEngineer(raw_df)
            fe.add_technical_indicators()
            df = fe.create_targets(HORIZON)

            feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']
            X = df[feature_cols]; y = df['Target_Return']
            split_idx = int(len(X) * SPLIT_RATIO)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Prepare Inputs
            last_row = fe.df.iloc[[-1]][feature_cols]

            # 3. Model Logic
            f_low, f_high = 0, 0
            pred_low, pred_high = 0, 0
            
            # --- MODEL A: LightGBM ---
            if "LightGBM" in MODEL_TYPE or "Ensemble" in MODEL_TYPE:
                alpha_lower = (1 - CONFIDENCE) / 2
                qm = QuantileModels(alpha_lower, 1 - alpha_lower)
                m_low_lgb, m_high_lgb = qm.train_lgbm(X_train, y_train)
                
                # Predictions
                f_low_lgb = m_low_lgb.predict(last_row)[0]
                f_high_lgb = m_high_lgb.predict(last_row)[0]
                pred_low_lgb = m_low_lgb.predict(X_test)
                pred_high_lgb = m_high_lgb.predict(X_test)

            # --- MODEL B: LSTM ---
            if "LSTM" in MODEL_TYPE or "Ensemble" in MODEL_TYPE:
                dl = DeepQuantileModel(input_shape=(1, len(feature_cols)))
                dl.train(X_train, y_train, epochs=20) 
                
                # Predictions
                f_low_lstm, f_high_lstm = dl.predict(last_row)
                f_low_lstm = f_low_lstm[0]; f_high_lstm = f_high_lstm[0]
                pred_low_lstm, pred_high_lstm = dl.predict(X_test)

            # --- AGGREGATION (Ensemble Logic) ---
            if "Ensemble" in MODEL_TYPE:
                f_low = (f_low_lgb + f_low_lstm) / 2
                f_high = (f_high_lgb + f_high_lstm) / 2
                pred_low = (pred_low_lgb + pred_low_lstm) / 2
                pred_high = (pred_high_lgb + pred_high_lstm) / 2
            elif "LightGBM" in MODEL_TYPE:
                f_low, f_high = f_low_lgb, f_high_lgb
                pred_low, pred_high = pred_low_lgb, pred_high_lgb
            else: # LSTM
                f_low, f_high = f_low_lstm, f_high_lstm
                pred_low, pred_high = pred_low_lstm, pred_high_lstm


            # 4. Calibration & Display
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

            st.markdown("---")
            st.subheader(f"🔮 Future Forecast (Target: {future_date.date()})")
            c1, c2, c3 = st.columns(3)
            c1.metric("📉 Bearish Limit", f"{CURRENCY}{p_low:.2f}", delta=f"{f_low*100:.2f}%", delta_color="inverse")
            c2.metric("📍 Current Price", f"{CURRENCY}{latest_price:.2f}")
            c3.metric("📈 Bullish Limit", f"{CURRENCY}{p_high:.2f}", delta=f"{f_high*100:.2f}%")
            
            st.info(f"Model predicts {TICKER} will likely stay between **{CURRENCY}{p_low:.2f}** and **{CURRENCY}{p_high:.2f}** over the next {HORIZON} days.")
            st.markdown("---")
            
            # Backtest Visualization
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