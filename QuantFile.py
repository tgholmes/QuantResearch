#New venture! This file will be a hub for learning and trialling different strategies in quant research

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from scipy import stats
import yfinance as yf
import mplfinance as mpf

#Stating dates of interest and interval frequency
StartDate = '2016-01-01' #YY-MM-DD
EndDate = '2024-07-01' #YY-MM-DD
interval = '1d' #e.g. 1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo

#Downloading share data for a range of shares in S&P500
ShareData = {}
tickers = ['MSFT'] #Add any tickers you want to analyse

colors = ['cornflowerblue','coral','mediumseagreen','grey','violet']

for ticker,color in zip(tickers,colors):
    print(f'Downloading Data for {ticker}')
    data = yf.download(ticker, start=StartDate, end=EndDate, interval=interval)
    ShareData[ticker] = data
    
    #Performing EDA
    ShareData[ticker]['20d_MA'] = ShareData[ticker]['Adj Close'].rolling(window=20).mean()
    ShareData[ticker]['50d_MA'] = ShareData[ticker]['Adj Close'].rolling(window=50).mean()
    ShareData[ticker]['returns'] = ShareData[ticker]['Adj Close'].pct_change().dropna()
    ShareData[ticker]['HistVol'] = ShareData[ticker]['returns'].std()

    plt.figure(figsize=(10, 7))

    mpf.plot(ShareData[ticker],type='candle',mav=(20,50),title=f'Candlestick chart for {ticker}', style='yahoo')
    plt.plot(ShareData[ticker]['Adj Close'], label=ticker, linewidth=2, color=color)
    plt.plot(ShareData[ticker]['20d_MA'], linestyle=':', linewidth=2, color=color)    

    plt.plot([], [], linestyle=':', linewidth=2, color='black', label='20-day MA')

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
    plt.legend(fontsize=15)
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Share Price ($)',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=15)
    
    # Show the plot
    plt.show()

    # Calculating returns (for interval specified earlier) for each ticker
    print(f'The Historical Volatility of {ticker} is: ',ShareData[ticker]['HistVol'])
    
    plt.figure(figsize=(10,7))

    plt.hist(ShareData[ticker]['returns'],bins=100,alpha=0.6,label=ticker,color=color)
    
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xlabel('Daily Returns')
    plt.ylabel('Count')
    plt.title(f'Histogram of Returns for {ticker} ({StartDate} - {EndDate}')

    plt.xlim(-0.1,0.1)




