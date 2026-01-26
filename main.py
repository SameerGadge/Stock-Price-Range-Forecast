from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.strategy import SignalGenerator
from src.utils import evaluate_metrics
from src.dashboard import DashboardGenerator
import pandas as pd
import numpy as np

# CONFIGURATION
# Add as many stocks as you want here!
WATCHLIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
HORIZON_DAYS = 5 
SPLIT_RATIO = 0.80

def run_pipeline(ticker, is_first=False):
    print(f"\n--- 🚀 Processing {ticker} ---")
    
    # 1. Data Pipeline
    loader = DataLoader(ticker)
    raw_df = loader.fetch_data()
    
    if raw_df is None or len(raw_df) < 200:
        print(f"Skipping {ticker}: Insufficient data.")
        return

    fe = FeatureEngineer(raw_df)
    fe.add_technical_indicators()
    df = fe.create_targets(HORIZON_DAYS)
    
    # 2. Train/Test Split
    feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']
    X = df[feature_cols]
    y = df['Target_Return']
    
    split_idx = int(len(X) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    test_dates = df.index[split_idx:]
    current_prices = df['Close'].iloc[split_idx:].values 
    
    # 3. Model Training
    qm = QuantileModels()
    model_low, model_high = qm.train_lgbm(X_train, y_train)
    
    # 4. Forecast Returns & Calibration
    pred_ret_low = model_low.predict(X_test)
    pred_ret_high = model_high.predict(X_test)
    
    # Dynamic Calibration
    if HORIZON_DAYS == 5: calib = 0.6
    elif HORIZON_DAYS == 21: calib = 1.0
    else: calib = 1.5
    
    center = (pred_ret_high + pred_ret_low) / 2
    width = (pred_ret_high - pred_ret_low)
    pred_ret_low = center - (width * calib)
    pred_ret_high = center + (width * calib)
    
    # 5. Reconstruct Prices
    p_low_price = current_prices * (1 + pred_ret_low)
    p_high_price = current_prices * (1 + pred_ret_high)
    y_test_price = current_prices * (1 + y_test.values)
    
    # 6. Evaluation
    picp, mpiw = evaluate_metrics(y_test_price, p_low_price, p_high_price)
    
    # 7. Strategy & Dashboard
    strat = SignalGenerator()
    df_sig, signals, total_pnl = strat.run_mean_reversion(test_dates, y_test_price, p_low_price, p_high_price)
    
    print(f"Strategy PnL: {total_pnl:.2f}")
    
    # GENERATE DASHBOARD PAGE
    # We pass the full WATCHLIST so the sidebar knows all links
    dash = DashboardGenerator(ticker)
    dash.generate_html(
        dates=test_dates, 
        actuals=y_test_price, 
        lower=p_low_price, 
        upper=p_high_price, 
        signals=signals,
        metrics=(picp, mpiw, total_pnl),
        recent_stocks=WATCHLIST 
    )
    
    # If this is the first stock, also save it as index.html (Homepage)
    if is_first:
        import shutil
        shutil.copy(f"{ticker}.html", "index.html")
        print(f"Set {ticker} as Homepage (index.html)")

if __name__ == "__main__":
    # Loop through the watchlist
    for i, stock in enumerate(WATCHLIST):
        run_pipeline(stock, is_first=(i==0))