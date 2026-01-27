import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import pytz # Optional: For specific timezones if needed

class DashboardGenerator:
    def __init__(self, ticker):
        self.ticker = ticker
        if self.ticker.endswith(".NS") or self.ticker.endswith(".BO"):
            self.currency = "₹"
        else:
            self.currency = "$"

    def generate_html(self, dates, actuals, lower, upper, signals, metrics, recent_stocks=[]):
        # 1. Prepare Data
        df = pd.DataFrame({'Date': dates, 'Actual': actuals, 'Lower': lower, 'Upper': upper})
        last_price = df['Actual'].iloc[-1]
        
        last_signal = "HOLD"
        if signals:
            if signals[-1][0] == df['Date'].iloc[-1]:
                last_signal = signals[-1][1]

        # 2. Build Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'].tolist() + df['Date'].tolist()[::-1], y=df['Upper'].tolist() + df['Lower'].tolist()[::-1], fill='toself', fillcolor='rgba(0, 100, 255, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='90% Confidence'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], mode='lines', name='Actual Price', line=dict(color='#00F0FF', width=2)))
        
        buy_x, buy_y = zip(*[(s[0], s[2]) for s in signals if s[1] == 'BUY']) if any(s[1]=='BUY' for s in signals) else ([],[])
        sell_x, sell_y = zip(*[(s[0], s[2]) for s in signals if s[1] == 'SELL']) if any(s[1]=='SELL' for s in signals) else ([],[])
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='BUY', marker=dict(symbol='triangle-up', size=12, color='#00FF00')))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='SELL', marker=dict(symbol='triangle-down', size=12, color='#FF0000')))

        fig.update_layout(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font=dict(color='white'), xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333', title=f"Price ({self.currency})"), hovermode="x unified", margin=dict(l=0, r=0, t=0, b=0), height=500)
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        picp, mpiw, pnl = metrics

        # 3. Generate Sidebar Links
        sidebar_html = ""
        for stock in recent_stocks:
            active_class = "active" if stock == self.ticker else ""
            # Link assumes file naming convention TICKER.html
            sidebar_html += f'<a href="{stock}.html" class="list-group-item list-group-item-action bg-dark text-white {active_class}">{stock}</a>'

        # 4. Generate Timestamp
        # Uses UTC by default, or server time. 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.ticker} Forecast</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ background-color: #121212; color: #e0e0e0; }}
                .card {{ background-color: #1e1e1e; border: 1px solid #333; }}
                .metric-value {{ font-size: 2rem; font-weight: bold; color: #fff; }}
                .signal-BUY {{ color: #00FF00; }} .signal-SELL {{ color: #FF0000; }} .signal-HOLD {{ color: #888; }}
                .sidebar {{ height: 100vh; border-right: 1px solid #333; padding-top: 20px; }}
                .footer-text {{ color: #666; font-size: 0.8rem; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-2 sidebar bg-dark">
                        <h4 class="text-center mb-4"> Watchlist</h4>
                        <div class="list-group list-group-flush">
                            {sidebar_html}
                        </div>
                    </div>
                    
                    <div class="col-md-10 p-4">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h1> {self.ticker} AI Forecast</h1>
                            <span class="badge bg-secondary">Latest Data: {df['Date'].iloc[-1].date()}</span>
                        </div>
                        <div class="row mb-4">
                            <div class="col-md-3"><div class="card p-3"><div class="text-muted">Price</div><div class="metric-value">{self.currency}{last_price:.2f}</div></div></div>
                            <div class="col-md-3"><div class="card p-3"><div class="text-muted">Coverage</div><div class="metric-value">{picp*100:.1f}%</div></div></div>
                            <div class="col-md-3"><div class="card p-3"><div class="text-muted">PnL</div><div class="metric-value" style="color:{'#0f0' if pnl>0 else '#f00'}">{pnl:+.2f}</div></div></div>
                            <div class="col-md-3"><div class="card p-3"><div class="text-muted">Signal</div><div class="metric-value signal-{last_signal}">{last_signal}</div></div></div>
                        </div>
                        <div class="card p-4">{plot_div}</div>
                        
                        <div class="text-center footer-text">
                            Last Updated: {current_time} (Server Time) <br>
                            Generated by Python Quant Pipeline • Hosted on GitHub Pages
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        filename = f"{self.ticker}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        if self.ticker == recent_stocks[0]: 
            with open("index.html", "w", encoding="utf-8") as f:
                f.write(html_content)
                
        print(f"Dashboard generated: {filename}")
