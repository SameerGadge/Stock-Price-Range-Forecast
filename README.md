# Stock Price Range Forecast

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-price-range-forecast.streamlit.app/)
[![Daily Forecast Action](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml/badge.svg)](https://github.com/SameerGadge/Stock-Price-Range-Forecast/actions/workflows/update_dashboard.yml)

An end-to-end Deep Learning & Machine Learning pipeline that forecasts **Price Intervals** (Confidence Bands) for any stock. Unlike traditional models that predict a single "wrong" price, this model predicts a **Probabilistic Range** (90% Confidence Interval) to help traders manage risk.

It features a **Triple-Engine Architecture**, giving you the ultimate flexibility in forecasting power.

##  Triple Models
The dashboard allows you to toggle between three powerful forecasting modes:

1.  ** LightGBM (Statistical):** Fast, stable, and excellent for volatility-based predictions.
2.  ** LSTM (Deep Learning):** A Recurrent Neural Network designed to catch non-linear sequential patterns.
3.  ** Ensemble (Hybrid):** The "Gold Standard" mode. Trains **both** models simultaneously and averages their outputs. This reduces individual model errors and typically provides the most robust forecast.

##  Key Features
* **Probabilistic Forecasting:** Predicts the **5th** and **95th** percentile of future price action.
* **Multi-Horizon Support:** Forecasts for **5 Days** (Swing), **21 Days** (Monthly), and **60 Days** (Quarterly).
* **Global Market Support:**
    * ** Indian Stocks:** NSE/BSE tickers (e.g., `INFY.NS`) with **₹** formatting and **India VIX** integration.
    * ** US Stocks:** NYSE/NASDAQ tickers (e.g., `AAPL`) with **$** formatting and **CBOE VIX**.
* **Dynamic Calibration:** Uncertainty bands automatically widen during high volatility (High VIX/ATR).
* **Full Automation:** GitHub Actions workflow retrains models every night at market close.

##  Tech Stack
* **Core:** Python 3.9+
* **ML/DL Engines:**
    * `LightGBM` (Gradient Boosting)
    * `TensorFlow/Keras` (Deep Learning/LSTM)
* **Data Source:** `yfinance` (Yahoo Finance API)
* **Visualization:** Plotly (Interactive Charts)
* **Deployment:** Streamlit Cloud (Frontend) & GitHub Pages (Static Reports)

##  How It Works
The model avoids "predicting the exact future price." Instead, it answers:
1.  **Lower Bound:** "There is a 95% chance the price will stay *above* this line." (Support)
2.  **Upper Bound:** "There is a 95% chance the price will stay *below* this line." (Resistance)

**Trading Logic (Mean Reversion):**
* **BUY:** When Price dips below the Lower Bound (Statistical Oversold).
* **SELL:** When Price spikes above the Upper Bound (Statistical Overbought).

##  Local Installation
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

##  License
MIT License - feel free to use this for your own trading or research!
