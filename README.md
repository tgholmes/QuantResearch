# Quantitative Equity Analysis: GBM Simulation, EDA & Backtesting

## Overview  
This repository serves as a starting point for exploring and developing quantitative strategies in equity markets using Python. The focus is on performing exploratory data analysis (EDA), modelling stock prices using Geometric Brownian Motion (GBM), and validating simulations via historical backtesting. This script is designed to be extendable for future models and trading strategies.

## Abstract  
This project builds a framework for simulating and evaluating equity price movements through statistical and quantitative techniques. A GBM model is applied to historical S&P500 constituents to simulate future price paths under the risk-neutral measure, estimate distributions of terminal prices, and visualize uncertainty in forecasted returns. Exploratory analysis includes moving average overlays, historical volatility calculations, return distributions, and candlestick charting. Furthermore, the model includes a simple backtesting module, comparing simulated GBM paths against realized historical returns over defined periods.  

These models lay foundational work for the development of future strategies, including options pricing, signal generation, and volatility surface calibration.

## Data  
Price, dividend, and split data are retrieved live using the [Yahoo Finance API](https://pypi.org/project/yfinance/).  
The framework currently supports configurable ticker symbols and time intervals.

- **Tickers Used (Example)**: `['SPY']`
- **Date Range**: April 1, 2025 – June 10, 2025
- **Interval**: `'1d'` (daily)

## File Overview  

- `DownloadData()`: Downloads and stores time series OHLCV data using `yfinance`.  
- `PerformEDA()`:  
  - Calculates 20-day and 50-day moving averages  
  - Computes daily returns, historical volatility  
  - Generates candlestick charts (via `mplfinance`)  
  - Plots return histograms, simulated GBM paths, PDFs, and CDFs  
- `CorporateActions()`: Fetches and prints dividend and stock split events for the tickers.  
- `BacktestGBM()`:  
  - Splits historical data into a training and test set  
  - Calibrates GBM parameters on training set  
  - Simulates price paths and compares simulated vs. actual outcomes  
  - Visualises discrepancies between stochastic model and market evolution

## Methodology & Packages Used  

- **Python Libraries**:  
  `NumPy`, `Pandas`, `Matplotlib`, `mplfinance`, `Seaborn`, `yfinance`, `SciPy`, `Plotly`  

- **Quantitative Techniques**:  
  - Daily return & volatility calculation  
  - GBM simulations (forward paths using stochastic differential equations)  
  - Fitting terminal price distributions (Normal PDF/CDF)  
  - Moving average strategies (20d, 50d crossover)  
  - Backtesting based on historical drift/vol estimates

## Results (Example Output for SPY)
- Estimated **annualized drift** and **volatility** using recent data  
- Generated 500 forward simulations of SPY for a 6-month horizon  
- Found that empirical return distributions deviate slightly from log-normal assumption  
- Backtesting module shows qualitative alignment with real-world movement, with variance due to fat tails and event risks

## Future Work  
- Expand to other equity indices and multi-asset classes  
- Integrate options pricing (Black-Scholes, implied vol surface)  
- Extend GBM model to include jump-diffusion or stochastic volatility  
- Build trading signals using crossover strategies or momentum indicators  
- Automate result saving and logging
