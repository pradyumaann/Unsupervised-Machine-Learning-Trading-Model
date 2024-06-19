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

df['bb_low'] = df.groupby(level =1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level =1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level =1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

#Here we'll define a custom function to calculate ATR, as transform function only works on 1 column at a time
def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr) 

#Now we'll create a function to calculate MACD
def compute_macd(close):
    macd= pandas_ta.macd(close = close, length = 20).iloc[:,0]
    #Here we'll need to normalize the data because we're going to use it in a machine learning model and we're going to cluster the data
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

#When calculating the dollar volume it's better to divide it by 1Million, to make it easier to comprehend 
df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

#Now aggregate to monthly level and filter top 150 most liquid stocks for each month 
#this is done to reduce training time and experiment with feature and strategies, we convert the business-daily data to month-end frequency
#We will aggregate all the technical indicators and adjusted close price, and take the end value for each month  
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
          df.unstack()[last_cols].resample('M').last().stack('ticker')],
          axis=1)).dropna()

#now we use agreggate Dollar Volume to filter out top 150 most liquid stocks
#calculating 5 year-rolling average of Dollar Volume for each stocks before filtering
data['dollar_volume'] = (data['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())
data['dollar_volume_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

#from the 500 stocks, the top 150 are selected based on 5 year rolling average Dollar Volume
data = data[data['dollar_volume_rank']<150].drop(['dollar_volume', 'dollar_volume_rank'], axis = 1)

#The next step is to calculate monthly returns for different time horizons and add them to the feature set

#Because we may want to capture Time-series dynamics that reflect momentum patterns.
def calculate_returns(df):
    #we also need an outlier cut-off, so for all the values above outlier threshold, they will be assigned the threshold of that percentile
    outlier_cutoff = 0.005
    #now we need to calculate returns for the following Lags
    lags = [1, 2, 3, 6, 9, 12]
    
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                               .pct_change(lag)
                               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                               .add(1)
                               .pow(1/lag)
                               .sub(1))
    return df
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()


 






