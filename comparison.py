import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict
import pandas_datareader.data as web

class FactorComparison:
    """
    Compare calculated Fama-French factors with official dataset
    """
    def __init__(self, calculated_factors: pd.DataFrame, start_date: str, end_date: str):
        """
        Initialize with calculated factors and date range
        
        Args:
            calculated_factors: DataFrame with calculated FF factors
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.calculated_factors = calculated_factors
        self.start_date = start_date
        self.end_date = end_date
        self.official_factors = None
        self.merged_factors = None
        
    def get_official_factors(self) -> None:
        """Download official Fama-French factors"""
        print("Downloading official Fama-French factors...")
        ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3', 
                                  'famafrench', 
                                  start=self.start_date,
                                  end=self.end_date)[0]
        
        ff_factors.index = ff_factors.index.to_timestamp()
        self.official_factors = ff_factors.drop('RF', axis=1)/100  # Convert to decimal
        
    def merge_factors(self) -> None:
        """Merge calculated and official factors"""
        if self.official_factors is None:
            self.get_official_factors()
            
        # Ensure both DataFrames have the same index frequency
        calc_monthly = self.calculated_factors.resample('M').last()
        off_monthly = self.official_factors.resample('M').last()
        
        # Merge the factors
        self.merged_factors = pd.DataFrame({
            'Mkt-RF_calc': calc_monthly['Mkt-RF'],
            'SMB_calc': calc_monthly['SMB'],
            'HML_calc': calc_monthly['HML'],
            'RMW_calc': calc_monthly['RMW'],
            'CMA_calc': calc_monthly['CMA'],
            'Mkt-RF_off': off_monthly['Mkt-RF'],
            'SMB_off': off_monthly['SMB'],
            'HML_off': off_monthly['HML'],
            'RMW_off': off_monthly['RMW'],
            'CMA_off': off_monthly['CMA']
        })
        
    def calculate_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate comparison statistics for each factor
        
        Returns:
            Dictionary containing statistical measures for each factor
        """
        if self.merged_factors is None:
            self.merge_factors()
            
        stats_dict = {}
        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
        for factor in factors:
            calc_col = f'{factor}_calc'
            off_col = f'{factor}_off'
            
            # Calculate correlation
            correlation = self.merged_factors[calc_col].corr(self.merged_factors[off_col])
            
            # Calculate tracking error
            tracking_error = np.std(self.merged_factors[calc_col] - self.merged_factors[off_col])
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(self.merged_factors[calc_col] - self.merged_factors[off_col]))
            
            # Calculate R-squared
            r_squared = stats.pearsonr(self.merged_factors[calc_col].dropna(), 
                                     self.merged_factors[off_col].dropna())[0] ** 2
            
            stats_dict[factor] = pd.DataFrame({
                'Correlation': correlation,
                'Tracking_Error': tracking_error,
                'MAE': mae,
                'R_Squared': r_squared
            }, index=[0])
            
        return stats_dict
    
    def plot_factor_comparison(self, factor: str) -> None:
        """
        Create comparison plots for a specific factor
        
        Args:
            factor: Factor name to plot (e.g., 'Mkt-RF', 'SMB', etc.)
        """
        if self.merged_factors is None:
            self.merge_factors()
            
        calc_col = f'{factor}_calc'
        off_col = f'{factor}_off'
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Time series plot
        self.merged_factors[[calc_col, off_col]].plot(ax=ax1)
        ax1.set_title(f'{factor} Factor: Calculated vs Official')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Factor Return')
        ax1.legend(['Calculated', 'Official'])
        ax1.grid(True)
        
        # Scatter plot with regression line
        sns.regplot(x=self.merged_factors[off_col], 
                   y=self.merged_factors[calc_col], 
                   ax=ax2,
                   scatter_kws={'alpha':0.5})
        ax2.set_title(f'{factor} Factor: Calculated vs Official (Scatter)')
        ax2.set_xlabel('Official Factor')
        ax2.set_ylabel('Calculated Factor')
        
        # Add correlation coefficient to scatter plot
        corr = self.merged_factors[calc_col].corr(self.merged_factors[off_col])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax2.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generate a comprehensive comparison report
        
        Returns:
            Summary DataFrame and detailed statistics dictionary
        """
        if self.merged_factors is None:
            self.merge_factors()
            
        # Calculate statistics
        stats_dict = self.calculate_statistics()
        
        # Create summary DataFrame
        summary = pd.DataFrame()
        for factor, stats_df in stats_dict.items():
            summary = pd.concat([summary, stats_df], ignore_index=True)
        
        summary.index = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
        # Plot comparisons for each factor
        for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            self.plot_factor_comparison(factor)
            
        return summary, stats_dict