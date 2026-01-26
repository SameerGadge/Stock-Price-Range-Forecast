import matplotlib.pyplot as plt
import numpy as np

def evaluate_metrics(y_true, p_lower, p_upper):
    # Coverage (PICP)
    in_bound = np.where((y_true >= p_lower) & (y_true <= p_upper), 1, 0)
    picp = np.mean(in_bound)
    
    # Width (MPIW)
    mpiw = np.mean(p_upper - p_lower)
    
    print(f"PICP (Coverage): {picp:.2%}")
    print(f"MPIW (Width): {mpiw:.2f}")
    return picp, mpiw

def visualize_results(df_sig, signals, ticker):
    plt.figure(figsize=(12, 6))
    plt.title(f"{ticker} Forecast & Mean Reversion Signals")
    
    # Plot Price & Bands
    plt.plot(df_sig['Date'], df_sig['Actual'], color='black', alpha=0.6, label='Price')
    plt.fill_between(df_sig['Date'], df_sig['Lower'], df_sig['Upper'], color='blue', alpha=0.1, label='90% CI')
    
    # Plot Signals
    for sig in signals:
        date, type_, price = sig
        if type_ == "BUY": 
            plt.scatter(date, price, color='green', marker='^', s=100, zorder=5)
        elif type_ == "SELL": 
            plt.scatter(date, price, color='red', marker='v', s=100, zorder=5)
        elif "EXIT" in type_: 
            plt.scatter(date, price, color='blue', marker='x', s=50, zorder=5)
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()