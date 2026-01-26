import yfinance as yf
import pandas as pd
import os

class DataLoader:
    def __init__(self, ticker, start_date="2015-01-01", data_dir="data"):
        self.ticker = ticker
        self.start_date = start_date
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def fetch_data(self):
        # 1. Automatic Detection: Use India VIX for .NS or .BO tickers
        if self.ticker.endswith(".NS") or self.ticker.endswith(".BO"):
            vix_ticker = "^INDIAVIX" 
            print(f"🇮🇳 Detected Indian Stock. Using {vix_ticker}...")
        else:
            vix_ticker = "^VIX" # Default to US VIX
            print(f"🇺🇸 Using US {vix_ticker}...")

        print(f"Fetching data for {self.ticker} and {vix_ticker}...")
        
        try:
            df = yf.download(self.ticker, start=self.start_date, progress=False)
            vix = yf.download(vix_ticker, start=self.start_date, progress=False)
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        # Merge VIX
        df['VIX'] = vix['Close']
        
        # 2. Robust Gap Filling (India VIX often has missing data points)
        df['VIX'] = df['VIX'].ffill().bfill()
        
        # Save raw data
        file_path = os.path.join(self.data_dir, f"{self.ticker}_raw.csv")
        df.to_csv(file_path)
        
        return df[['Close', 'High', 'Low', 'Volume', 'VIX']].copy()