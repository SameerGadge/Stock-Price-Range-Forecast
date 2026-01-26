import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.strategy import SignalGenerator
from src.utils import evaluate_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Stock Forecaster", layout="wide", page_icon="📈")

# --- SESSION STATE INITIALIZATION (History Tracker) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- CSS STYLING ---
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 20px; border-radius: 10px; text-align: center; }
    div[data-testid="column"] { width: fit-content !important; flex: 1 1 auto; }
    .history-btn { width: 100%; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")

# History Selection Logic
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = "RELIANCE.NS"

def set_ticker(ticker):
    st.session_state['selected_ticker'] = ticker

# Render History in Sidebar
if st.session_state['history']:
    st.sidebar.subheader("🕒 Recent History")
    for stock in reversed(st.session_state['history']):
        if st.sidebar.button(stock, key=f"hist_{stock}", use_container_width=True):
            set_ticker(stock)
            st.rerun() # Immediate reload
    st.sidebar.markdown("---")

# Inputs
TICKER = st.sidebar.text_input("Ticker Symbol", value=st.session_state['selected_ticker']).upper()
HORIZON = st.sidebar.selectbox("Forecast Horizon", options=[5, 21, 60], format_func=lambda x: f"{x} Days")
CONFIDENCE = st.sidebar.slider("Confidence Level", 0.70, 0.99, 0.90)
SPLIT_RATIO = 0.80

# Currency Detection
if TICKER.endswith(".NS") or TICKER.endswith(".BO"):
    CURRENCY = "₹"
else:
    CURRENCY = "$"

st.title(f"📈 {TICKER} Price Interval Forecasting")
st.markdown(f"**Quantile Regression Model** • {int(CONFIDENCE*100)}% Confidence Interval • Mean Reversion Strategy")

# --- MAIN APP LOGIC ---

if st.button("🚀 Run Forecast Model") or ('history' in st.session_state and TICKER in st.session_state.history and TICKER == st.session_state.selected_ticker):
    # Only run if button pressed OR we just clicked a history item
    
    with st.spinner(f"Fetching data and training models for {TICKER}..."):
        try:
            # 1. Data Pipeline
            loader = DataLoader(TICKER)
            raw_df = loader.fetch_data()
            
            if raw_df is None or raw_df.empty:
                st.error(f"❌ Could not fetch data for '{TICKER}'.")
                st.stop()
            
            if len(raw_df) < 200:
                st.error(f"❌ Not enough historical data. Needs 200+ days.")
                st.stop()

            # --- UPDATE HISTORY ---
            # Remove if exists to move to top, keep only last 5
            if TICKER in st.session_state['history']:
                st.session_state['history'].remove(TICKER)
            st.session_state['history'].append(TICKER)
            if len(st.session_state['history']) > 5:
                st.session_state['history'].pop(0)

            # 2. Feature Engineering
            fe = FeatureEngineer(raw_df)
            fe.add_technical_indicators()
            df = fe.create_targets(HORIZON)

            # 3. Train/Test Split
            feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']
            X = df[feature_cols]
            y = df['Target_Return']
            
            split_idx = int(len(X) * SPLIT_RATIO)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            test_dates = df.index[split_idx:]
            current_prices = df['Close'].iloc[split_idx:].values 

            # 4. Model Training
            alpha_lower = (1 - CONFIDENCE) / 2
            alpha_upper = 1 - alpha_lower
            qm = QuantileModels(alpha_lower, alpha_upper)
            model_low, model_high = qm.train_lgbm(X_train, y_train)

            # 5. Dynamic Calibration
            if HORIZON == 5: calib_factor = 0.6
            elif HORIZON == 21: calib_factor = 1.0
            else: calib_factor = 1.5

            # ==========================================
            # 🔮 FUTURE FORECAST
            # ==========================================
            last_row_df = fe.df.iloc[[-1]][feature_cols]
            latest_price = raw_df['Close'].iloc[-1]
            latest_date = raw_df.index[-1]
            
            future_ret_low = model_low.predict(last_row_df)[0]
            future_ret_high = model_high.predict(last_row_df)[0]
            
            f_center = (future_ret_high + future_ret_low) / 2
            f_width = (future_ret_high - future_ret_low)
            future_ret_low = f_center - (f_width * calib_factor) 
            future_ret_high = f_center + (f_width * calib_factor)
            
            future_price_low = latest_price * (1 + future_ret_low)
            future_price_high = latest_price * (1 + future_ret_high)
            future_date = latest_date + pd.Timedelta(days=HORIZON)

            st.markdown("---")
            st.subheader(f"🔮 Future Forecast (Target: {future_date.date()})")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("📉 Bearish Limit", f"{CURRENCY}{future_price_low:.2f}", delta=f"{future_ret_low*100:.2f}%", delta_color="inverse")
            c2.metric("📍 Current Price", f"{CURRENCY}{latest_price:.2f}")
            c3.metric("📈 Bullish Limit", f"{CURRENCY}{future_price_high:.2f}", delta=f"{future_ret_high*100:.2f}%")
            
            st.info(f"Calibration Factor: {calib_factor}x | Horizon: {HORIZON} Days")
            st.markdown("---")

            # ==========================================
            # 🔙 BACKTEST RESULTS
            # ==========================================
            st.subheader(f"📊 Historical Backtest")

            pred_ret_low = model_low.predict(X_test)
            pred_ret_high = model_high.predict(X_test)

            center = (pred_ret_high + pred_ret_low) / 2
            width = (pred_ret_high - pred_ret_low)
            pred_ret_low = center - (width * calib_factor)
            pred_ret_high = center + (width * calib_factor)

            p_low_price = current_prices * (1 + pred_ret_low)
            p_high_price = current_prices * (1 + pred_ret_high)
            y_test_price = current_prices * (1 + y_test.values)

            strat = SignalGenerator()
            df_sig, signals, total_pnl = strat.run_mean_reversion(test_dates, y_test_price, p_low_price, p_high_price)
            
            picp = ((y_test_price >= p_low_price) & (y_test_price <= p_high_price)).mean()
            mpiw = (p_high_price - p_low_price).mean()
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Coverage", f"{picp:.1%}", delta=f"{picp-CONFIDENCE:.1%}")
            k2.metric("Avg Width", f"{CURRENCY}{mpiw:.2f}")
            k3.metric("PnL", f"{total_pnl:.2f}")
            k4.metric("Trades", f"{len(signals)}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates.tolist() + test_dates.tolist()[::-1], y=p_high_price.tolist() + p_low_price.tolist()[::-1], fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Confidence'))
            fig.add_trace(go.Scatter(x=test_dates, y=y_test_price, mode='lines', name='Actual', line=dict(color='#00F0FF', width=2)))
            
            buy_x, buy_y = zip(*[(s[0], s[2]) for s in signals if s[1] == 'BUY']) if any(s[1]=='BUY' for s in signals) else ([],[])
            sell_x, sell_y = zip(*[(s[0], s[2]) for s in signals if s[1] == 'SELL']) if any(s[1]=='SELL' for s in signals) else ([],[])
            
            fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='BUY', marker=dict(symbol='triangle-up', size=12, color='#00FF00')))
            fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='SELL', marker=dict(symbol='triangle-down', size=12, color='#FF0000')))

            fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1, x=0))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")