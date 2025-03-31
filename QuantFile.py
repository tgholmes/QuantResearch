#New venture! This file will be a hub for learning and trialling different strategies in quant research

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
import mplfinance as mpf
from datetime import datetime
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objects as go
from arch import arch_model
sns.set_style("darkgrid")
 

#Stating dates of interest and interval frequency
StartDate = '2024-03-10' #YY-MM-DD
EndDate = '2025-03-10' #YY-MM-DD
interval = '1wk' #e.g. 1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo

#Downloading share data for a range of shares in S&P500
ShareData = {}
tickers = ['AAPL','NVDA','MSFT','META','GSPC'] #Add any tickers you want to analyse

colors = ['cornflowerblue','coral','mediumseagreen','grey','violet','lightpink']

def DownloadData(tickers, Start, End, interval):
    ShareData = {}
    for ticker in tickers:
        print(f'Downloading Data for {ticker}')
        stock = yf.Ticker(ticker)
        data = stock.history(start=Start, end=End, interval='1d')           
        ShareData[ticker] = data
    return ShareData

ShareData = DownloadData(tickers, StartDate, EndDate, interval)

def PerformEDA(ShareData, tickers, StartDate, EndDate, interval):
    colors = ['cornflowerblue','coral','mediumseagreen','grey','violet','lightpink']
    
    for ticker, color in zip(tickers, colors):
        data = ShareData[ticker]
        
        # Performing EDA
        data['20d_MA'] = data['Close'].rolling(window=20).mean()
        data['50d_MA'] = data['Close'].rolling(window=50).mean()
        data['returns'] = data['Close'].pct_change().dropna()
        data['HistVol'] = data['returns'].std()
        
        plt.figure(figsize=(10, 7))
        
        # Using MPL Finance to make candlestick plots
        filepath = f'{ticker}/candlestick_chart_{ticker}.png'
        saveparams = dict(fname=filepath, dpi=610)
        mpf.plot(data, type='candle', mav=(20), title=f'Candlestick chart for {ticker}', volume=True, style='nightclouds', savefig=saveparams)
        
        # Plotting Close and 20d MA on separate plot
        plt.plot(data['Close'], label=ticker, linewidth=2, color=color)
        plt.plot(data['20d_MA'], linestyle=':', linewidth=2, color=color)
        plt.plot(data['50d_MA'], linestyle='--', linewidth=2, color=color)
        
        plt.plot([], [], linestyle=':', linewidth=2, color=color, label='20-day MA')
        plt.plot([], [], linestyle='--', linewidth=2, color=color, label='50-day MA')

        # X-axis formatting based on interval
        if interval in ['1m', '2m']:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        elif interval in ['5m', '15m', '30m', '60m']:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=60))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        # Reduce the number of x-axis ticks
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        
        # Rotate and format x-axis labels
        plt.gcf().autofmt_xdate()
        
        # Add grid, legend, title, and labels
        plt.grid(True)
        plt.legend(fontsize=11)
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Share Price ($)', fontsize=11)
        plt.tick_params(axis='both', which='major', labelsize=11)
        
        # Show the plot
        plt.savefig(f'{ticker}/{ticker}_MA_Analysis.png',dpi=610)
        
        # Calculating returns (for interval specified earlier) for each ticker
        print(f'The Historical Volatility of {ticker} is: {data["HistVol"].mean()}')
        
        plt.figure(figsize=(10,7))
        
        plt.hist(data['returns'], bins=50, alpha=1, label=ticker, color=color)
        
        plt.grid(True)
        plt.legend(fontsize=15)
        plt.xlabel('Daily Returns')
        plt.ylabel('Count')
        plt.title(f'Histogram of Returns for {ticker} ({StartDate} - {EndDate})')
        
        plt.xlim(-0.1, 0.1)
        plt.show()
        
        annual_drift = data['returns'].mean() * 252
        annual_volatility = data['returns'].std() * np.sqrt(252)
        
        # Convert annual parameters to daily parameters
        daily_drift = annual_drift / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        print(f"Annual Drift for {ticker}: ", annual_drift)
        print(f"Annual Volatility for {ticker}: ", annual_volatility)
        
        current_price = data['Close'].iloc[-1]
        num_simulations = 500
        T = 125  # Time horizon in days (1/2 year)
        
        # Simulate the GBM paths
        simulations = np.zeros((num_simulations, T))
        simulations[:, 0] = current_price
        for t in range(1, T):
            z = np.random.standard_normal(num_simulations)
            simulations[:, t] = simulations[:, t-1] * np.exp(daily_drift + daily_volatility * z)
        
        # Final prices from simulations (not returns)
        final_prices = simulations[:, -1]

        # --- CDF for GBM Simulated Final Prices ---
        sorted_final_prices = np.sort(final_prices)  # Sort final prices
        cdf = np.arange(1, len(sorted_final_prices) + 1) / len(sorted_final_prices)  # CDF values

        # --- Create the figure and grid layout ---
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[3, 1])

        # --- 1. Plot GBM simulation paths (Left, Top) ---
        ax1 = fig.add_subplot(gs[0, 0])  # Top left
        for i in range(num_simulations):
            ax1.plot(simulations[i, :], color=color, alpha=0.2)
        
        ax1.set_title(f"GBM Simulation for {ticker} ({num_simulations} Simulations)")
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True)

        # --- 2. Plot the histogram of final prices (Right, Top) ---
        ax2 = fig.add_subplot(gs[0, 1])  # Top right
        ax2.hist(final_prices, bins=50, orientation='horizontal', color=color, density=True, alpha=0.2, label='Simulated Prices')

        # Fit a normal distribution to the final prices
        mean, std = norm.fit(final_prices)

        # Generate points for the Gaussian curve
        xmin, xmax = ax2.get_ylim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)

        # Plot the Gaussian curve
        ax2.plot(p, x, 'k', linewidth=2, label=f'Gaussian Fit\nμ={mean:.2f}, σ={std:.2f}', color=color)
        ax2.set_title('Price Distribution (PDF)')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True)

        # --- 3. Plot the CDF for the final GBM-simulated prices (Bottom row, full width) ---
        ax3 = fig.add_subplot(gs[1, :])  # Bottom row, spans both columns
        ax3.plot(sorted_final_prices, cdf, marker='.', linestyle='-', color=color, label=f'{ticker} GBM CDF')
        ax3.set_title(f'Cumulative Distribution Function (CDF) for {ticker}')
        ax3.set_xlabel('Simulated Final Prices ($)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True)
        ax3.legend()

        # Adjust the layout and save
        plt.tight_layout()
        plt.savefig(f'{ticker}/{ticker}_GBM_Analysis_with_CDF.png', dpi=610)
        plt.show()
        
        
def CorporateActions(ShareData, tickers, StartDate, EndDate, interval):
    for ticker in tickers:
        ticker_data = yf.Ticker(ticker)
        
        # Fetching dividend information
        dividends = ticker_data.dividends
        ShareData[ticker]['Dividends'] = dividends
        print(f'Dividends for {ticker}:')
        print(dividends)
        
        # Fetching stock split information
        splits = ticker_data.splits
        ShareData[ticker]['Stock_Splits'] = splits
        print(f'Stock Splits for {ticker}:')
        print(splits)


def BacktestGBM(ShareData, tickers, StartDate, EndDate, interval, backtest_days=125):
    """
    Perform backtesting on GBM simulation by simulating previous time periods
    and comparing the true path taken by the share price.
    """
    colors = ['cornflowerblue','coral','mediumseagreen','grey','violet','lightpink']
    TrueColors = ['paleturquoise','red','lime','white','purple','crimson']
    
    for ticker, color, true_color in zip(tickers, colors, TrueColors):
        data = ShareData[ticker]

        # Calculating returns and volatility over historical data
        data['returns'] = data['Close'].pct_change().dropna()
        
        # Select the backtest period (the last `backtest_days` days)
        backtest_data = data[-backtest_days:]
        training_data = data[:-backtest_days]  # Use data before this period to calculate drift and volatility

        # Calculate the daily drift and volatility based on the training data
        annual_drift = training_data['returns'].mean() * 252
        annual_volatility = training_data['returns'].std() * np.sqrt(252)
        daily_drift = annual_drift / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Get the current price at the start of the backtest period
        start_price = backtest_data['Close'].iloc[0]
        num_simulations = 500
        T = len(backtest_data)  # Simulate over the same length as the backtest period
        
        # Simulate the GBM paths
        simulations = np.zeros((num_simulations, T))
        simulations[:, 0] = start_price
        for t in range(1, T):
            z = np.random.standard_normal(num_simulations)
            simulations[:, t] = simulations[:, t-1] * np.exp(daily_drift + daily_volatility * z)
        
        # Extract the dates for the backtest period
        backtest_dates = backtest_data.index
        
        # Plot the GBM simulation paths and the true price path
        plt.figure(figsize=(10, 7))
        
        # --- Plot GBM simulation paths ---
        for i in range(num_simulations):
            plt.plot(backtest_dates, simulations[i, :], color=color, alpha=0.2)  # Use dates for x-axis
        
        # --- Plot the true price path ---
        plt.plot(backtest_dates, backtest_data['Close'].values, color=true_color, label=f'True Price Path for {ticker}', linewidth=2)

        # Format the x-axis to show dates properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Date format
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto locator for dates
        plt.gcf().autofmt_xdate()  # Auto rotate date labels
        
        plt.title(f"GBM Backtest for {ticker} ({num_simulations} Simulations vs. Actual)")
        plt.xlabel('Date')  # Changed label to 'Date'
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        # Save the backtest plot
        plt.tight_layout()
        plt.savefig(f'{ticker}/{ticker}_GBM_Backtest.png', dpi=610)
        plt.show()
        
def GBM_with_GARCH(ShareData, tickers, forecast_horizon=10, confidence_level=0.95):
    for ticker in tickers:
        data = ShareData[ticker].copy()

        # Ensure 'returns' column exists
        if 'returns' not in data.columns:
            data['returns'] = data['Close'].pct_change()

        # Drop NaN values (first row will be NaN)
        data = data.dropna(subset=['returns'])

        # Ensure there are enough data points
        if len(data['returns']) < 30:
            print(f"⚠️ Not enough return data for {ticker}. Skipping GARCH model.")
            continue

        # Fit GARCH(1,1) model
        garch_model = arch_model(data['returns'], vol='Garch', p=1, q=1)
        try:
            garch_fit = garch_model.fit(disp="off")
        except ValueError as e:
            print(f"❌ GARCH model failed for {ticker}: {e}")
            continue

        print(f"\n📈 GARCH Model Summary for {ticker}:\n{garch_fit.summary()}")

        # Extract estimated volatility
        garch_volatility = np.sqrt(garch_fit.conditional_volatility)

        # Plot Estimated Volatility Over Time
        plt.figure(figsize=(10, 5))
        plt.plot(garch_volatility, label="GARCH Estimated Volatility", color="blue")
        plt.title(f"Estimated Volatility Over Time ({ticker})")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()

        # Extract fitted values (expected returns)
        fitted_returns = garch_fit.resid

        # Plot Actual vs. Fitted Returns
        plt.figure(figsize=(10, 5))
        plt.plot(data['returns'], label="Actual Returns", alpha=0.6)
        plt.plot(fitted_returns, label="Fitted Returns", linestyle="dashed", color="red")
        plt.title(f"Actual vs. GARCH Fitted Returns ({ticker})")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.legend()
        plt.show()

        # Forecast future volatility
        garch_forecast = garch_fit.forecast(start=len(data), horizon=forecast_horizon)
        forecasted_volatility = np.sqrt(garch_forecast.variance.values[-1])

        # Plot Forecasted Volatility
        plt.figure(figsize=(10, 5))
        plt.plot(forecasted_volatility, marker="o", linestyle="dashed", color="green")
        plt.title(f"Forecasted Volatility (Next {forecast_horizon} Days) - {ticker}")
        plt.xlabel("Days Ahead")
        plt.ylabel("Volatility")
        plt.show()

        # Compute rolling VaR
        z_score = norm.ppf(1 - confidence_level)
        rolling_VaR = z_score * garch_volatility

        # Plot Rolling Value-at-Risk (VaR)
        plt.figure(figsize=(10, 5))
        plt.plot(rolling_VaR, color="red", label=f"1-Day {int(confidence_level*100)}% VaR")
        plt.title(f"Rolling 1-Day VaR Over Time ({ticker})")
        plt.xlabel("Time")
        plt.ylabel("Value-at-Risk")
        plt.legend()
        plt.show()

        print(f"✅ GARCH analysis completed for {ticker}!\n")

PerformEDA(ShareData, tickers, StartDate, EndDate, interval)

CorporateActions(ShareData, tickers, StartDate, EndDate, interval)

BacktestGBM(ShareData, tickers, StartDate, EndDate, interval)

GBM_with_GARCH(ShareData,tickers)


