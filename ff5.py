import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockMetricsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.trading.client import TradingClient
import warnings
warnings.filterwarnings('ignore')

class FamaFrenchCalculator:
    """
    A class to calculate Fama-French 5 factors using market data from Alpaca API.
    
    Attributes:
        stock_list (List[str]): List of stock tickers to analyze
        start_date (str): Start date for analysis
        end_date (str): End date for analysis
        data_client: Alpaca historical data client
        trading_client: Alpaca trading client
        price_data (pd.DataFrame): Monthly price data for all stocks
        fundamentals (pd.DataFrame): Fundamental data for all stocks
        factors (pd.DataFrame): Calculated Fama-French factors
    """
    
    def __init__(self, stock_list: List[str], start_date: str, end_date: str, api_key: str, secret_key: str):
        """
        Initialize the calculator with stock list, date range, and Alpaca credentials.
        
        Args:
            stock_list: List of stock tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self.stock_list = stock_list
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key)
        self.price_data = None
        self.fundamentals = None
        self.factors = None
        
    def get_price_data(self) -> None:
        """Download monthly price data for all stocks from Alpaca."""
        print("Downloading price data...")
        
        # Request monthly bars for all stocks
        request_params = StockBarsRequest(
            symbol_or_symbols=self.stock_list,
            timeframe=TimeFrame.Month,
            start=self.start_date,
            end=self.end_date,
            adjustment=Adjustment.ALL  # Get adjusted prices
        )
        
        try:
            bars = self.data_client.get_stock_bars(request_params)
            
            # Convert to DataFrame and process
            df = bars.df
            
            # Pivot the data to get tickers as columns
            self.price_data = df['close'].unstack(level=0)
            print(f"Downloaded price data for {len(self.price_data.columns)} stocks")
            
        except Exception as e:
            print(f"Error downloading price data: {str(e)}")
            
    def get_stock_metrics(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get fundamental metrics for a list of stocks using Alpaca API.
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            DataFrame with fundamental metrics
        """
        try:
            # Request metrics for all stocks
            request_params = StockMetricsRequest(
                symbol_or_symbols=tickers,
                start=self.end_date - timedelta(days=7),  # Get most recent metrics
                end=self.end_date
            )
            
            metrics = self.data_client.get_stock_metrics(request_params)
            return metrics.df
            
        except Exception as e:
            print(f"Error getting stock metrics: {str(e)}")
            return pd.DataFrame()
            
    def collect_fundamentals(self) -> None:
        """Collect fundamental data for all stocks using Alpaca API."""
        print("Collecting fundamental data...")
        
        # Get metrics for all stocks
        metrics_df = self.get_stock_metrics(self.stock_list)
        
        # Process metrics and create fundamentals DataFrame
        fundamentals = {}
        
        for ticker in self.stock_list:
            try:
                # Get asset information from trading client
                asset = self.trading_client.get_asset(ticker)
                
                # Get latest metrics for the ticker
                ticker_metrics = metrics_df.xs(ticker, level=0).iloc[-1] if ticker in metrics_df.index else pd.Series()
                
                # Create metrics dictionary
                metrics = pd.DataFrame({
                    'market_cap': ticker_metrics.get('market_cap', np.nan),
                    'book_value': ticker_metrics.get('total_equity', np.nan),
                    'shares_outstanding': ticker_metrics.get('shares_outstanding', np.nan),
                    'operating_income': ticker_metrics.get('operating_income_ttm', np.nan),
                    'total_assets': ticker_metrics.get('total_assets', np.nan),
                    'investment': ticker_metrics.get('capex_ttm', np.nan) / ticker_metrics.get('total_assets', np.nan) if ticker_metrics.get('total_assets', 0) != 0 else np.nan,
                    'sector': asset.sector if hasattr(asset, 'sector') else '',
                    'exchange': asset.exchange,
                }, index=[pd.Timestamp.now()])
                
                fundamentals[ticker] = metrics
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        self.fundamentals = pd.concat(fundamentals, axis=0)
        self.fundamentals.index.names = ['ticker', 'date']
        
        # Print coverage statistics
        coverage = len(fundamentals) / len(self.stock_list) * 100
        print(f"\nData coverage: {coverage:.1f}% ({len(fundamentals)} out of {len(self.stock_list)} stocks)")
        
        if len(fundamentals) > 0:
            sector_counts = self.fundamentals['sector'].value_counts()
            print("\nSector Distribution:")
            print(sector_counts)
    
    def calculate_monthly_factors(self, monthly_data: pd.Series) -> pd.Series:
        """Calculate Fama-French factors for a given month."""
        # Get market return (Mkt-RF)
        mkt_rf = monthly_data.pct_change().mean()
        
        # Merge price data with fundamentals
        data = pd.merge(
            monthly_data.to_frame('price'),
            self.fundamentals,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Calculate ratios
        data['btm'] = data['book_value'] / data['market_cap']
        data['roe'] = data['operating_income'] / data['book_value']
        
        # Calculate factors
        size_median = data['market_cap'].median()
        btm_median = data['btm'].median()
        roe_median = data['roe'].median()
        inv_median = data['investment'].median()
        
        # Size factor (SMB)
        smb = (data[data['market_cap'] <= size_median]['price'].pct_change().mean() - 
               data[data['market_cap'] > size_median]['price'].pct_change().mean())
        
        # Value factor (HML)
        hml = (data[data['btm'] >= btm_median]['price'].pct_change().mean() - 
               data[data['btm'] < btm_median]['price'].pct_change().mean())
        
        # Profitability factor (RMW)
        rmw = (data[data['roe'] >= roe_median]['price'].pct_change().mean() - 
               data[data['roe'] < roe_median]['price'].pct_change().mean())
        
        # Investment factor (CMA)
        cma = (data[data['investment'] <= inv_median]['price'].pct_change().mean() - 
               data[data['investment'] > inv_median]['price'].pct_change().mean())
        
        return pd.Series({
            'Mkt-RF': mkt_rf,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma
        })

    def calculate_factors(self) -> pd.DataFrame:
        """Calculate Fama-French factors for all months."""
        if self.price_data is None:
            self.get_price_data()
        
        if self.fundamentals is None:
            self.collect_fundamentals()
        
        print("Calculating monthly factors...")
        factors = pd.DataFrame()
        
        for date in self.price_data.index:
            monthly_prices = self.price_data.loc[date]
            factors_series = self.calculate_monthly_factors(monthly_prices)
            factors = pd.concat([factors, factors_series.to_frame(date).T])
        
        self.factors = factors
        return factors

    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for the factors."""
        if self.factors is None:
            raise ValueError("Factors have not been calculated yet. Run calculate_factors() first.")
        return self.factors.describe()

    def plot_factor_returns(self, factor: str = None) -> None:
        """Plot cumulative returns for factors."""
        if self.factors is None:
            raise ValueError("Factors have not been calculated yet. Run calculate_factors() first.")
        
        cumulative_returns = (1 + self.factors).cumprod()
        
        plt.figure(figsize=(12, 6))
        if factor:
            cumulative_returns[factor].plot()
            plt.title(f'Cumulative Returns: {factor}')
        else:
            cumulative_returns.plot()
            plt.title('Cumulative Returns: All Factors')
        
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.show()

# Example usage
def main():
    # Get S&P 500 constituents
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    symbols_list = sp500['Symbol'].unique().tolist()

    # Set dates
    end_date = '2024-06-18'
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=8)
    start_date = start_date.strftime('%Y-%m-%d')

    # Initialize calculator with Alpaca credentials
    api_key = "YOUR_ALPACA_API_KEY"
    secret_key = "YOUR_ALPACA_SECRET_KEY"
    
    ff_calculator = FamaFrenchCalculator(symbols_list, start_date, end_date, api_key, secret_key)

    # Calculate factors
    calculated_factors = ff_calculator.calculate_factors()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(ff_calculator.get_summary_statistics())

    # Plot factor returns
    ff_calculator.plot_factor_returns()

if __name__ == "__main__":
    main()