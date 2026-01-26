# 📈 AI Stock Price Interval Forecaster

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-price-range-forecast.streamlit.app/)
[![Daily Forecast Action](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml/badge.svg)](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml)

An end-to-end Machine Learning pipeline that forecasts **Price Intervals** (Confidence Bands) for any stock using Quantile Regression (LightGBM). Unlike traditional models that predict a single "wrong" price, this model predicts a **Probabilistic Range** (90% Confidence Interval) to help traders manage risk.

## 🚀 Live Demos
* **🔮 Dynamic Analysis (Streamlit):** [Launch Interactive App](https://stock-price-range-forecast.streamlit.app/)  
    *(Analyze ANY stock in real-time: `RELIANCE.NS`, `NVDA`, `TCS.NS`, etc.)*
* **📊 Daily Static Report:** [View GitHub Pages Dashboard](https://sameergadge.github.io/Stock-Price-Range-Forecast/)  
    *(Automated daily report for watchlist stocks)*

## 📊 Key Features
* **Dual-Mode Forecasting:**
    * **Short Term (5 Days):** Tight confidence bands for swing trading.
    * **Long Term (21/60 Days):** Wider bands adjusting for long-term volatility.
* **Global Market Support:**
    * **🇮🇳 Indian Stocks:** Full support for NSE/BSE tickers (e.g., `INFY.NS`) with automatic **Rupee (₹)** formatting and **India VIX** integration.
    * **🇺🇸 US Stocks:** Support for NYSE/NASDAQ tickers (e.g., `AAPL`) with **Dollar ($)** formatting and **CBOE VIX**.
* **Advanced AI Logic:**
    * **Quantile Regression:** Predicts the 5th and 95th percentile of future returns.
    * **Dynamic Calibration:** Automatically expands risk cones as the forecast horizon increases.
    * **Mean Reversion Strategy:** Generates BUY signals when price hits the lower bound and SELL signals at the upper bound.
* **Full Automation:**
    * Daily **GitHub Actions** workflow retrains models every night at market close.

## 🛠️ Tech Stack
* **Core:** Python 3.9+
* **ML Models:** LightGBM (Quantile Objective), XGBoost
* **Data Source:** `yfinance` (Yahoo Finance API)
* **Visualization:** Plotly (Interactive Charts)
* **Deployment:** Streamlit Cloud (Frontend) & GitHub Pages (Static Reports)

## 📉 How It Works
The model avoids "predicting the exact future price." Instead, it answers:
1.  **Lower Bound:** "There is a 95% chance the price will stay *above* this line." (Support)
2.  **Upper Bound:** "There is a 95% chance the price will stay *below* this line." (Resistance)

**Trading Logic:**
* **BUY:** When Price dips below the Lower Bound (Statistical Oversold).
* **SELL:** When Price spikes above the Upper Bound (Statistical Overbought).

## 📦 Local Installation
To run this project on your own machine:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/SameerGadge/Stock-Price-Range-Forecast.git](https://github.com/SameerGadge/Stock-Price-Range-Forecast.git)
    cd Stock-Price-Range-Forecast
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

## 📜 License
MIT License - feel free to use this for your own trading or research!