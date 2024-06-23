## S&P 500 Trading Model

This repository contains a Python-based trading model designed to optimize a portfolio of S&P 500 stocks. The model integrates technical indicators, clustering, and portfolio optimization techniques to select and manage a high-performing stock portfolio.

## Features

- **Data Collection**: Downloads historical stock data for S&P 500 constituents.
- **Technical Indicators**: Computes various indicators like RSI, Bollinger Bands, MACD, and ATR.
- **Liquidity Filtering**: Filters the top 150 most liquid stocks based on dollar volume.
- **Feature Calculation**: Calculates monthly returns and rolling factor betas using Fama-French factors.
- **Clustering**: Applies K-means clustering to group stocks based on their technical features.
- **Portfolio Optimization**: Uses the Efficient Frontier to optimize the portfolio for maximum Sharpe ratio.
- **Performance Evaluation**: Compares portfolio performance against the S&P 500 index.

## Installation

1. Clone the repository:
   git clone https://github.com/pradyumaann/Unsupervised-Machine-Learning_Trading-Model.git

2. Navigate to the project directory:
   cd Unsupervised-Machine-Learning_Trading-Model


## Usage

1. Run the main script to execute the trading model:
   python main.py
  
2. The script will download the necessary data, calculate features, perform clustering, optimize the portfolio, and output performance results.

## Dependencies

- Python 3.x
- pandas
- numpy
- yfinance
- pandas_datareader
- scikit-learn
- matplotlib
- statsmodels
- PyPortfolioOpt

## Project Structure

- `trading_model.py`: Main script to run the trading model.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Limitations

- Relies heavily on technical indicators, which may not always predict future performance accurately.
- Limited to S&P 500 stocks, potentially missing broader market trends.
- Fixed initial centroids for K-means clustering might limit adaptability.
- Risk of overfitting due to optimization based on historical data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.

