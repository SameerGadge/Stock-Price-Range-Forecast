import numpy as np
import lightgbm as lgb
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class QuantileModels:
    def __init__(self, alpha_lower=0.05, alpha_upper=0.95):
        self.alpha_lower = alpha_lower
        self.alpha_upper = alpha_upper

    def train_lgbm(self, X, y):
        # Parameters for stability
        params = {
            'n_jobs': 1, 
            'verbose': -1,
            'force_col_wise': True
        }
        
        # Lower Bound Model (5th Percentile)
        m_low = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha_lower, n_estimators=100, **params)
        m_low.fit(X, y)
        
        # Upper Bound Model (95th Percentile)
        m_high = lgb.LGBMRegressor(objective='quantile', alpha=self.alpha_upper, n_estimators=100, **params)
        m_high.fit(X, y)
        
        return m_low, m_high