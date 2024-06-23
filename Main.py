from statsmodels.regression.rolling import RollingOLS
import pandas_datareader as web
import matplotlib.pyplot as plt
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

#Calculating features and technical indicators for each stock
#Feature & tecnical indicators include: Garman-Klass Volatility, Dollar Volume, MACD, RSI, BOllinger Bands
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
last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
          df.unstack()[last_cols].resample('M').last().stack('ticker')],
          axis=1)).dropna()

#now we use agreggate Dollar Volume to filter out top 150 most liquid stocks
#calculating 5 year-rolling average of Dollar Volume for each stocks before filtering
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12).mean().stack())
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


#Download the Fama-French Factors and Calculate Rolling Factor Betas for each stock
#We can access the historical factor returns using the pandas-datareader and estimate historical exposures using the Rolling OLS Regression
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                             'famafrench',
                             start='2016')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'

#Now we can regress the End of Month Return with Factor Data and calculate Beta
factor_data = factor_data.join(data['return_1m']).sort_index() 

#In this step we're going to filter out stocks that have less than 10 month data, we are doing that because we're going to use Regression for 2 years
#and stocks that don't have enough data will break our function
observations = factor_data.groupby(level=1).size()
valid_stocks = observations[observations >=10]
factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

#now we calculate rolling factor betas by regressing 1 month return with all the other factors in factor data
betas = (factor_data.groupby(level=1, group_keys=False)
        .apply(lambda x: RollingOLS(endog=x['return_1m'],
                             exog=sm.add_constant(x.drop('return_1m', axis=1)),
                             window=min(23, x.shape[0]),
                             min_nobs=len(x.columns)+1)
               .fit(params_only=True)
               .params
               .drop('const', axis=1)))

#The data for Factor_data is only till 2024-04-30, so we'll have to remove the last 2 months from all the columns in 'data' till 2024-04-30

#Find the latest date in factor_data
latest_factor_date = factor_data.index.get_level_values('date').max()
# Filter out data beyond the latest date in factor_data
data = data[data.index.get_level_values('date')<= latest_factor_date]
data = data.drop('adj close', axis=1)
data = data.dropna()

factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
data = (data.join(betas))
data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))


#We're going to fit a K-Means Clustering algorithm and split all the stocks in a few clusters, then we're going to analyze the clusters   

#For each month fit a K-Means Clustering Algorithm to group similar assets based on their features, optimum number of Clusters each month is around 4
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#applying pre defined centroids
target_rsi_values = [35, 45, 55, 65]
initial_centroids = np.zeros((len(target_rsi_values), 18))
initial_centroids[:, 1] = target_rsi_values

def get_clusters(df):
   
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

#Plot function to plot the Clusters
def plot_clusters(data):
    cluster_0 = data[data['cluster']==0]
    cluster_1 = data[data['cluster']==1]
    cluster_2 = data[data['cluster']==2]
    cluster_3 = data[data['cluster']==3]
    
    plt.scatter(cluster_0.iloc[:,5], cluster_0.iloc[:,1], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:,5], cluster_1.iloc[:,1], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:,5], cluster_2.iloc[:,1], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:,5], cluster_3.iloc[:,1], color='black', label='cluster 3')
    
    plt.legend()
    plt.show()
    return

#For this model we need to follow stocks' momentum, for that we can use RSI as the main indicator. 
plt.style.use('ggplot')
for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)
    #plt.title(f'Date {i}')
    #plot_clusters(g)
    
#after fitting the model, we can observe that Cluster 3 represents all the stocks with an RSI around 65, Cluster 2 represents RSI around 55, and so on.  
#Now we can use the Cluster 3 in our portfolio to select every month the stocks which have a good momentum and RSI around 65

#For each month selets assets based on the cluster and form a portfolio
#based on Efficient Frontier Max Sharpe Ratio
#The Hypothesis is that Stocks which have an RSI of around 70 have good momentum and they should have a good momentum even in the next month 

#the idea here is to create a dictionary with: the first date of the next month, & a list of all the good momentum stocks from previous month    
filtered_df = data[data['cluster'] == 1].copy() 
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])  

dates = filtered_df.index.get_level_values('date').unique().tolist() 

#create the dictionary with 'date' as key and the list of all the stocks from previous month as value
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
    

#The next step is to define Portfolio Optimization Function
#We'll define the function which optimizes portfolio weights usin PyPortfolioOpt package and EfficientFrontier optimizer to maximize the sharpe ratio
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#There may be cases where the optimize_weights function may assign the weight 0 to many stocks or 1 to a single stock
#to avoid that we'll apply single weight bounds constraint for diversification(minimum half of equal weight and maximum 10% of portfolio)
def optimize_weights(prices, lower_bound=0):
    
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)
    
    ef = EfficientFrontier(expected_returns=returns, 
                           cov_matrix=cov, 
                           weight_bounds=(lower_bound,1), 
                           solver='SCS')
    
    weights = ef.max_sharpe()
    
    return ef.clean_weights()
#download fresh Daily prices Data only for shortlisted stocks
stocks = data.index.get_level_values('ticker').unique().tolist()
new_df = yf.download(tickers=stocks, start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12), 
                     end=data.index.get_level_values('date').unique()[-1])

#Now calculate daily returns for each stock which could land up in our potfolio, then loop over each month start, select the stocks for the month and calculate their weights for the next month
#if the Maximum Sharpe Ratio Optimization fails for a given month, apply equally-weighted weights and calculate each day portfolio return
returns_dataframe = np.log(new_df['Adj Close']).diff()
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]
    
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        optimization_df = new_df['Adj Close'][cols].loc[optimization_start_date:optimization_end_date]

        success = False
        try:
            weights = optimize_weights(prices=optimization_df, lower_bound=round(1/(len(optimization_df.columns)*2), 3))
            weights = pd.DataFrame(weights, index=pd.Series(0))
            success = True
            
        except:
            print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal Weights')
        if success == False:
            weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))],
                               index = optimization_df.columns.to_list(),
                               columns=pd.Series(0)).T
    
        temp_df = returns_dataframe[start_date:end_date]
        temp_df = temp_df.stack().to_frame('return').reset_index(level=0).merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True), left_index=True, right_index=True).reset_index().set_index(['Date','Ticker']).unstack().stack()
        temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
        temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
        portfolio_df = pd.concat([portfolio_df, temp_df], axis = 0)
    
    except Exception as e:
        print(e)

portfolio_df = portfolio_df.drop_duplicates()

#In this step we're going to visulaize the returns of our portfolio and compare them to the benchmark, which in this case is the S&P 500  Index
import matplotlib.ticker as mtick

spy = yf.download(tickers='SPY',
                  start='2015-01-01',
                  end = dt.date.today())
spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':'SPY Buy&Hold'}, axis=1)

portfolio_df = portfolio_df.merge(spy_ret,
                                  left_index=True,
                                  right_index=True)

#the next step is to calculate the cummulative return of both S&P500 and our Strategy, and plot them to compare
portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1
portfolio_cumulative_return.plot(figsize=(16,6))
plt.title('Unsupervised Learning Trading Strategy Returns Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()