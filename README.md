# 📈 AI Stock Price Interval Forecaster

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-price-range-forecast.streamlit.app/)
[![Daily Forecast Action](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml/badge.svg)](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml)

An end-to-end Machine Learning pipeline that forecasts **Price Intervals** (Confidence Bands) for any stock. Unlike traditional models that predict a single "wrong" price, this model predicts a **Probabilistic Range** (90% Confidence Interval) to help traders manage risk.

It features a **Dual-Engine Architecture**, allowing you to switch between Statistical Machine Learning and Deep Learning in real-time.

## 🚀 Live Demos
* **🔮 Dynamic Analysis (Streamlit):** [Launch Interactive App](https://stock-price-range-forecast.streamlit.app/)  
    *(Analyze ANY stock in real-time: `RELIANCE.NS`, `NVDA`, `TCS.NS`, etc.)*
* **📊 Daily Static Report:** [View GitHub Pages Dashboard](https://sameergadge.github.io/Stock-Price-Range-Forecast/)  
    *(Automated daily report for watchlist stocks)*

## 🧠 Dual AI Engines
The dashboard allows you to toggle between two powerful forecasting engines:

1.  **⚡ LightGBM (Statistical Boosting):**
    * **Best for:** Speed, tabular data (RSI, ATR, VIX), and stability.
    * **Method:** Gradient Boosted Decision Trees with Quantile Objective.
    * **Performance:** Excellent on daily time-frames with limited history.

2.  **🧠 LSTM (Deep Learning):**
    * **Best for:** Complex sequential patterns and non-linear trends.
    * **Method:** Long Short-Term Memory (Recurrent Neural Network) with **Custom Quantile Loss**.
    * **Performance:** Experimental engine that captures temporal dependencies better than trees.

## 📊 Key Features
* **Probabilistic Forecasting:** Predicts the **5th** and **95th** percentile of future price action.
* **Multi-Horizon Support:** Forecasts for **5 Days** (Swing), **21 Days** (Monthly), and **60 Days** (Quarterly).
* **Global Market Support:**
    * **🇮🇳 Indian Stocks:** NSE/BSE tickers (e.g., `INFY.NS`) with **₹** formatting and **India VIX** integration.
    * **🇺🇸 US Stocks:** NYSE/NASDAQ tickers (e.g., `AAPL`) with **$** formatting and **CBOE VIX**.
* **Dynamic Calibration:** Uncertainty bands automatically widen during high volatility (High VIX/ATR).
* **Full Automation:** GitHub Actions workflow retrains models every night at market close.

## 🛠️ Tech Stack
* **Core:** Python 3.9+
* **ML Engines:**
    * `LightGBM` (Gradient Boosting)
    * `TensorFlow/Keras` (Deep Learning/LSTM)
* **Data Source:** `yfinance` (Yahoo Finance API)
* **Visualization:** Plotly (Interactive Charts)
* **Deployment:** Streamlit Cloud (Frontend) & GitHub Pages (Static Reports)

## 📉 How It Works
The model avoids "predicting the exact future price." Instead, it answers:
1.  **Lower Bound:** "There is a 95% chance the price will stay *above* this line." (Support)
2.  **Upper Bound:** "There is a 95% chance the price will stay *below* this line." (Resistance)

**Trading Logic (Mean Reversion):**
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