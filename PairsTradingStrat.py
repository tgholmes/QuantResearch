#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:22:51 2024

@author: tomholmes
"""

#This file is dedicated to testing the Pairs Trading Strategy, developed in 1980s by Morgan Stanley Analysts
#Pairs Strategy revolves around the relation between 2 stocks that have a strong correlation over long period
#Then, when the correlation between the stocks drops/rises, the trader can take a short and long position on the overvalued and undervalued share respectively

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
import mplfinance as mpf
from datetime import datetime
from scipy.stats import norm
from statsmodels.tsa.stattools import coint


#Dates and Tickers of Interest
StartDate = '2024-02-10' #YY-MM-DD
EndDate = datetime.now().strftime('%Y-%m-%d') #YY-MM-DD
interval = '1d' #e.g. 1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo
tickers = ['JPM','GS',] #Add any tickers you want to analyse
PriceData = {}

colors = ['grey','cornflowerblue']

def DownloadData(tickers, StartDate, EndDate, interval):
    PriceData = pd.DataFrame()
    for ticker in tickers:
        print(f'Downloading Data for {ticker}')
        data = yf.download(ticker, start=StartDate, end=EndDate, interval=interval)['Close']
        PriceData[ticker] = data
    return PriceData

PriceData = DownloadData(tickers, StartDate, EndDate, interval)
returns = PriceData.pct_change().dropna()


plt.figure(figsize=(10,7))
plt.plot(PriceData[tickers[0]],label=tickers[0],linewidth=2)
plt.plot(PriceData[tickers[1]],label=tickers[1],linewidth=2)
plt.title(f"Historical Prices of {tickers[0]} and {tickers[1]}")
plt.xlabel("Date")
plt.ylabel('Share Price ($)')
plt.ylim(0,PriceData[tickers[0]].max()+PriceData[tickers[0]].max()/4)
plt.legend()

score, p_value, _ = coint(PriceData[tickers[0]], PriceData[tickers[1]])
print(f'Cointegration test p-value: {p_value}')

rollingCorrelation = returns[tickers[0]].rolling(window=60).corr(returns[tickers[1]])

# Plot the spread
plt.figure(figsize=(10, 8))
plt.plot(rollingCorrelation, label=f"60-day Rolling Correlation ({tickers[0]}, {tickers[1]})",linewidth=2)
plt.axhline(rollingCorrelation.mean(), color='red', linestyle='--',linewidth=2)
plt.ylim(0,1)
plt.legend()
plt.title("60-Day Rolling Correlation")
plt.xlabel("Date")
plt.ylabel("Rolling Correlation")
plt.show()

# Calculate price ratio
price_ratio = PriceData[tickers[0]] / PriceData[tickers[1]]

# Calculate Z-score of the price ratio
price_ratio_mean = price_ratio.mean()
price_ratio_std = price_ratio.std()
z_score = (price_ratio - price_ratio_mean) / price_ratio_std

# Plotting the Z-score of the price ratio
plt.figure(figsize=(12, 6))
plt.plot(z_score, label='Z-Score of Price Ratio')
plt.axhline(1, color='red', linestyle='--', label='+1 Z')
plt.axhline(-1, color='blue', linestyle='--', label='-1 Z')
plt.axhline(0, color='white', linestyle='-', label='Mean')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.title("Z-Score of Price Ratio for Trading Signals")
plt.show()
