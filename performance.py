import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def calculate_metrics(returns, risk_free_rate=0.0):
    """
    Calculates key performance metrics for a given strategy.
    Args:
        returns (pd.Series): Strategy's returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    total_return = (returns + 1).prod() - 1
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(252)
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
    }

def compare_to_benchmark(strategy_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Compares a strategy's performance to a benchmark.
    Args:
        strategy_returns (pd.Series): Strategy returns.
        benchmark_returns (pd.Series): Benchmark returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio.

    Returns:
        pd.DataFrame: Metrics for both the strategy and the benchmark.
    """
    strategy_metrics = calculate_metrics(strategy_returns, risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_returns, risk_free_rate)

    comparison = pd.DataFrame({
        "Metric": strategy_metrics.keys(),
        "Strategy": strategy_metrics.values(),
        "Benchmark": benchmark_metrics.values(),
    })

    return comparison

def factor_analysis(returns, factors, risk_free_rate=0.0):
    """
    Performs factor analysis to calculate alpha and factor sensitivities.
    Args:
        returns (pd.Series): Strategy's excess returns.
        factors (pd.DataFrame): Market factor data.
        risk_free_rate (float): Risk-free rate for calculating excess returns.

    Returns:
        dict: Alpha, p-value, factor coefficients, and factor p-values.
    """
    excess_returns = returns - risk_free_rate
    X = sm.add_constant(factors)
    model = sm.OLS(excess_returns, X).fit()

    return {
        "Alpha": model.params["const"] * 252,  # Annualized alpha
        "Alpha p-value": model.pvalues["const"],
        "Coefficients": model.params.drop("const"),
        "P-values": model.pvalues.drop("const"),
    }

def plot_cumulative_returns(strategy_returns, benchmark_returns=None, title="Cumulative Returns"):
    """
    Plots cumulative returns of a strategy and optional benchmark.
    Args:
        strategy_returns (pd.Series): Strategy returns.
        benchmark_returns (pd.Series, optional): Benchmark returns.
        title (str): Plot title.

    Returns:
        None
    """
    strategy_cum_returns = (strategy_returns + 1).cumprod() - 1
    plt.plot(strategy_cum_returns, label="Strategy")

    if benchmark_returns is not None:
        benchmark_cum_returns = (benchmark_returns + 1).cumprod() - 1
        plt.plot(benchmark_cum_returns, label="Benchmark", linestyle="--")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_factor_exposures(factor_analysis_result, title="Factor Exposures"):
    """
    Plots factor exposures from factor analysis.
    Args:
        factor_analysis_result (dict): Output of factor_analysis.
        title (str): Plot title.

    Returns:
        None
    """
    coefficients = factor_analysis_result["Coefficients"]
    coefficients.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Factors")
    plt.ylabel("Exposure")
    plt.grid(True)
    plt.show()

def generate_performance_report(strategy_returns, benchmark_returns, factors=None, risk_free_rate=0.0):
    """
    Generates a performance report for a strategy.
    Args:
        strategy_returns (pd.Series): Strategy returns.
        benchmark_returns (pd.Series): Benchmark returns.
        factors (pd.DataFrame, optional): Factor data for analysis.
        risk_free_rate (float): Risk-free rate for calculations.

    Returns:
        dict: Performance metrics and visualizations.
    """
    report = {}

    # Compare to Benchmark
    report["Comparison"] = compare_to_benchmark(strategy_returns, benchmark_returns, risk_free_rate)

    # Factor Analysis
    if factors is not None:
        report["Factor Analysis"] = factor_analysis(strategy_returns, factors, risk_free_rate)

    # Visualizations
    print("Plotting cumulative returns...")
    plot_cumulative_returns(strategy_returns, benchmark_returns)

    if factors is not None:
        print("Plotting factor exposures...")
        plot_factor_exposures(report["Factor Analysis"])

    return report
