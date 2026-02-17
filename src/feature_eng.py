import pandas as pd
import pandas_ta as ta
import numpy as np

class FeatureEngineer:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def add_technical_indicators(self):
        # 1. ATR (Volatility)
        self.df['ATR'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        
        # 2. Bollinger Bands (Robust Column Handling)
        bb = ta.bbands(self.df['Close'], length=20, std=2)
        
        if bb is not None and not bb.empty:
            # Dynamically find Upper/Lower columns to avoid KeyErrors
            upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
            lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
            
            self.df['BB_Upper'] = bb[upper_col]
            self.df['BB_Lower'] = bb[lower_col]
            self.df['BB_Width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / self.df['Close']
        else:
            self.df['BB_Upper'] = self.df['Close']
            self.df['BB_Lower'] = self.df['Close']
            self.df['BB_Width'] = 0
            
        # 3. Returns
        self.df['Return'] = self.df['Close'].pct_change()
        self.df['Log_Return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        self.df.dropna(inplace=True)
        return self.df

    def create_targets(self, horizon_days):
        # FORECAST TARGET: Percent Return over the horizon
        # (Forecasting raw prices fails because trees cannot extrapolate to new highs)
        self.df['Target_Return'] = self.df['Close'].pct_change(periods=horizon_days).shift(-horizon_days)
        return self.df.dropna()
