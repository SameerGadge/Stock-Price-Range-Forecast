import pandas as pd

class SignalGenerator:
    def run_mean_reversion(self, dates, actuals, lower, upper):
        signals = []
        current_pos = 0 # 0=None, 1=Long, -1=Short
        entry_price = 0
        pnl = []
        
        df = pd.DataFrame({'Date': dates, 'Actual': actuals, 'Lower': lower, 'Upper': upper})
        df['Midpoint'] = (df['Lower'] + df['Upper']) / 2
        
        for i, row in df.iterrows():
            price = row['Actual']
            
            # ENTRY LOGIC
            if current_pos == 0:
                if price < row['Lower']: # Oversold -> Buy
                    current_pos = 1
                    entry_price = price
                    signals.append((row['Date'], "BUY", price))
                    
                elif price > row['Upper']: # Overbought -> Sell
                    current_pos = -1
                    entry_price = price
                    signals.append((row['Date'], "SELL", price))
            
            # EXIT LOGIC
            elif current_pos == 1:
                if price >= row['Midpoint']: # Reverted to mean
                    pnl.append(price - entry_price)
                    current_pos = 0
                    signals.append((row['Date'], "EXIT_LONG", price))
                    
            elif current_pos == -1:
                if price <= row['Midpoint']: # Reverted to mean
                    pnl.append(entry_price - price)
                    current_pos = 0
                    signals.append((row['Date'], "EXIT_SHORT", price))
                    
        return df, signals, sum(pnl)