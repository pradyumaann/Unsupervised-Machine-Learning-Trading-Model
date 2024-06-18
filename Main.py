from statsmodels.regression.rolling import RollingOLS
import pandas_datareader as web
import matplotlib as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

#The first step is to dowload S&P 500 onstituents data, we'll take the list of constituents from a wikipedia page "List of S&P 500 companies"
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

#The list of symbols for the S&P 500 constituents might include '.' punctuations, so here we'll clean the data to get the list of symbols 
sp500['Symbol'] = sp500['Symbol'].str.replace('.','-')
symbols_list = sp500['Symbol'].unique().tolist()

#In this step we'll download the data from Yahoo Finance upto 18 July, 2024. The data starts from 18 July, 2016.
end_date = '2024-06-18'
start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

#With this function from Yahoo Finance package, we can download the data for all the stocks in symbols_list.
df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()
df.index.names = ['date','ticker']
df.columns = df.columns.str.lower()

#Calculating features and techhnical indicators for each stock
#Feature & tecnical indicators include: 
#Garman-Klass Volatility, Dollar Volume, MACD, RSI, BOllinger Bands
df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2 - (2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
df.xs('AAPL', level=1)['rsi'].plot()