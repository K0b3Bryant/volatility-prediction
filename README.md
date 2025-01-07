# Monica: Financial Modeling and Strategy Backtesting Framework

Monica is a comprehensive Python-based framework for financial modeling, volatility prediction, and strategy backtesting. It combines advanced data preprocessing, machine learning models, and custom strategy evaluation tools to test and optimize trading strategies.

## Features

- **Data Preprocessing**: Handles CSV inputs with date parsing, duplicate management, and missing value imputation.
- **Volatility Measures**: Implements advanced methods like Garman-Klass, Parkinson, Rogers-Satchell, and more.
- **Machine Learning Models**:
  - Linear Regression
  - Random Forest
  - Gradient Boosting (LightGBM, XGBoost)
  - Artificial Neural Networks (ANNs)
  - Long Short-Term Memory (LSTM) models
- **Markov Switching Models**: Predict state probabilities using Markov Regime Switching and integrates GARCH/MS-GARCH volatility modeling.
- **Feature Engineering**:
  - Technical Analysis indicators (e.g., RSI, Bollinger Bands)
  - Time Series Features (rolling windows, lags)
- **Strategy Backtesting**:
  - Supports threshold-based, momentum, and mean-reversion strategies.
  - Includes benchmarks like buy-and-hold and GARCH/MS-GARCH-based strategies.
- **Performance Metrics**: Calculates Sharpe ratio, Sortino ratio, Alpha, and evaluates against market factors.
- **Robustness Testing**: Expanding and sliding window methodologies for historical robustness testing.

## Installation

### Required Libraries
Install dependencies using pip:
```bash
pip install ta tsfresh torch skorch optuna quantstats ace_tools arch
