import pandas as pd
import numpy as np

class Strategy:
    def __init__(self, name: str, signals: pd.Series, risky_returns: pd.Series, risk_free_returns: pd.Series, max_position: float, transaction_costs: float=0.01):
        self.name = name
        self.transaction_costs = transaction_costs
        self.max_position = max_position
        aligned_data = pd.merge(signals, risky_returns, left_index=True, right_index=True, how='inner')
        aligned_data = pd.merge(aligned_data, risk_free_returns, left_index=True, right_index=True, how='inner')
        aligned_data = cleaner(aligned_data, backfill=True, verbose=False)
        self.signals = aligned_data.iloc[:, 0]
        self.risky_returns = aligned_data.iloc[:, 1].fillna(0)
        self.risk_free_returns = aligned_data.iloc[:, 2].fillna(0)


    def calculate_simple_returns(self, lag: int, position_func, *args, **kwargs) -> pd.DataFrame:
        if 'window' in kwargs:
            positions = self.signals.rolling(window=kwargs['window']).apply(lambda x: position_func(x, max_position=self.max_position, **kwargs), raw=False).fillna(0)
        else:
            positions = self.signals.apply(lambda x: position_func(x, *args, max_position=self.max_position, **kwargs)).fillna(0)
        switches = positions.diff().fillna(0).abs() > 0
        simple_returns = (positions.shift(lag) * self.risky_returns + (1 - positions.shift(lag).abs()) * self.risk_free_returns - switches * self.transaction_costs).fillna(0)
        return pd.DataFrame({'positions': positions, 'switches': switches, 'simple_returns': simple_returns})

class StrategyManager:
    def __init__(self, risk_free_returns: pd.Series):
        self.risk_free_returns = risk_free_returns
        self.strategies = {}
        self.benchmarks = {}
        self.calculated_strategies_and_benchmarks = {}

    def add_strategy(self, strategy: Strategy, position_func, lag: int = 1, *args, **kwargs):
        result_df = strategy.calculate_simple_returns(lag, position_func, *args, **kwargs)
        self.strategies[strategy.name] = {
            'strategy_object': strategy,
            'position_func': position_func,
            'lag': lag,
            'signals': strategy.signals,
            'risky_returns': strategy.risky_returns,
            'risk_free_returns': strategy.risk_free_returns,
            'window': kwargs.get('window', None),
            'upper': kwargs.get('upper', None),
            'lower': kwargs.get('lower', None),
            'result_df': result_df
            }

    def add_benchmark(self, benchmark: Strategy, position_func, lag: int = 0, *args, **kwargs):
        result_df = benchmark.calculate_simple_returns(lag, position_func, *args, **kwargs)
        self.benchmarks[benchmark.name] = {
            'strategy_object': benchmark,
            'position_func': position_func,
            'lag': lag,
            'signals': benchmark.signals,
            'risky_returns': benchmark.risky_returns,
            'risk_free_returns': benchmark.risk_free_returns,
            'result_df': result_df
            }

    def calculate_compounded_returns(self, simple_returns: pd.Series) -> pd.Series:
        return (simple_returns + 1).cumprod() - 1

    def calculate_strategy_metrics(self, returns: pd.Series) -> tuple:
        total_return = (returns + 1).prod() - 1
        sharpe_ratio = (returns.mean() - self.risk_free_returns.mean()) / returns.std() * np.sqrt(250)
        sortino_ratio = (returns.mean() - self.risk_free_returns.mean()) / returns[returns < 0].std() * np.sqrt(250)
        return total_return, sharpe_ratio, sortino_ratio

    def calculate_strategies(self, start_date=None, end_date=None, strategy_names: list = None): # shit works
        self.calculated_strategies_and_benchmarks = {'strategies': {}, 'benchmarks': {}}
        strategies_to_calculate = self.strategies
        if strategy_names:
            strategies_to_calculate = {name: strategies_to_calculate[name] for name in strategy_names if name in strategies_to_calculate}
        for name, strategy_data in strategies_to_calculate.items():
            strategy = strategy_data['strategy_object']
            sliced_df = strategy_data['result_df'].loc[start_date:end_date].copy()
            sliced_df['compounded_returns'] = self.calculate_compounded_returns(sliced_df['simple_returns'])
            self.calculated_strategies_and_benchmarks['strategies'][name] = sliced_df

        for name, benchmark_data in self.benchmarks.items():
            sliced_benchmark_df = benchmark_data['result_df'].loc[start_date:end_date].copy()
            sliced_benchmark_df['compounded_returns'] = self.calculate_compounded_returns(sliced_benchmark_df['simple_returns'])
            self.calculated_strategies_and_benchmarks['benchmarks'][name] = sliced_benchmark_df

    def get_calculated_strategies_and_benchmarks(self):
        return self.calculated_strategies_and_benchmarks

    def analyze_factors(self, returns: pd.Series, factors_list: list=['SMB', 'HML', 'RMW', 'CMA', 'UMD']):
        try:
            factors = pd.read_csv('/content/factors.csv', index_col=0) # consider csv_reader()
            factors.index = pd.to_datetime(factors.index, dayfirst=True)
            factors = cleaner(factors, backfill=True, verbose=False)
            factors = factors.loc[:, factors_list]
        except FileNotFoundError:
            print("Error: The file '/content/factors.csv' cannot be found.")
            return None, None, None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None, None

        excess_returns = returns - self.risk_free_returns.reindex(returns.index, fill_value=0)

        if excess_returns.name is None:
            excess_returns.name = 'excess_returns'

        aligned_data = pd.concat([excess_returns, factors], axis=1).dropna()
        y = aligned_data[excess_returns.name]
        X = aligned_data[factors.columns]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            alpha = model.params['const']
            p_value_alpha = model.pvalues['const']
            coefficients = model.params.drop('const')
            p_values = model.pvalues.drop('const')
        except Exception as e:
            print(f"An error occurred while fitting the model: {e}")
            return None, None, None, None

        alpha = alpha * 252  # annualized
        coefficients = coefficients * 252 # annualized

        return alpha, p_value_alpha, coefficients, p_values

    def generate_performance_table(self, factors_list: list=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'VIX', 'BAB'], strategy_names: list = None, include_benchmarks=True) -> pd.DataFrame:
        data = []
        strategies_to_include = self.calculated_strategies_and_benchmarks['strategies']

        if strategy_names:
            strategies_to_include = {name: strategies_to_include[name] for name in strategy_names if name in strategies_to_include}

        for name, df in strategies_to_include.items():
            avg_simple_return = df['simple_returns'].mean()
            switches = df['switches'].count()
            #costs = (switches * (1 + self.transaction_costs))-1
            total_return, sharpe_ratio, sortino_ratio = self.calculate_strategy_metrics(df['simple_returns'])
            alpha, p_value_alpha, coefficients, p_values = self.analyze_factors(df['simple_returns'], factors_list)

            metrics = {
                'Strategy': name,
                'Avg Simple Return': avg_simple_return,
                'Total Return': total_return,
                'switches': switches,
                #'costs': costs,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Alpha': alpha,
                'Alpha p-value': p_value_alpha
            }

            for factor, value in coefficients.items():
                metrics[factor] = value
                metrics[f"{factor} p-value"] = p_values[factor]
            data.append(metrics)

        if include_benchmarks:
            for name, df in self.calculated_strategies_and_benchmarks['benchmarks'].items():
                avg_simple_return = df['simple_returns'].mean()
                total_return, sharpe_ratio, sortino_ratio = self.calculate_strategy_metrics(df['simple_returns'])
                alpha, p_value_alpha, coefficients, p_values = self.analyze_factors(df['simple_returns'])

                metrics = {
                    'Strategy': name,
                    'Avg Simple Return': avg_simple_return,
                    'Total Return': total_return,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Alpha': alpha,
                    'Alpha p-value': p_value_alpha
                }
                for factor, value in coefficients.items():
                    metrics[factor] = value
                    metrics[f"{factor} p-value"] = p_values[factor]
                data.append(metrics)

        performance_df = pd.DataFrame(data)
        performance_df = performance_df.sort_values(by='Alpha', ascending=False)
        return performance_df

    def plot_results(self, plot_name: str, strategy_names: list = None, include_benchmarks=True):
        plt.figure(figsize=(12, 8))
        strategies_to_plot = self.calculated_strategies_and_benchmarks['strategies']
        if strategy_names:
            strategies_to_plot = {name: strategies_to_plot[name] for name in strategy_names if name in strategies_to_plot}

        for name, df in strategies_to_plot.items():
            total_return, sharpe_ratio, sortino_ratio = self.calculate_strategy_metrics(df['simple_returns'])
            alpha, p_value_alpha, _, _ = self.analyze_factors(df['simple_returns'])
            plt.plot(df.index, df['compounded_returns'], label=f"{name} - Total Return: {total_return*100:.2f}%, Sharpe: {sharpe_ratio:.2f}, Alpha: {alpha:.2f}")

        if include_benchmarks:
            for name, df in self.calculated_strategies_and_benchmarks['benchmarks'].items():
                total_return, sharpe_ratio, sortino_ratio = self.calculate_strategy_metrics(df['simple_returns'])
                alpha, p_value_alpha, _, _ = self.analyze_factors(df['simple_returns'])
                plt.plot(df.index, df['compounded_returns'], linestyle='--', label=f"{name} (Benchmark) - Total Return: {total_return*100:.2f}%, Sharpe: {sharpe_ratio:.2f}, Alpha: {alpha:.2f}")

        plt.legend(fontsize=7)
        plt.title(plot_name)
        plt.xlabel("Date")
        plt.ylabel("Compounded Returns")
        plt.grid(True)
        plt.show()


def Threshold(signal, lower, upper, max_position):
    if lower < signal < upper:
        return 0
    elif signal > upper:
        return -1 * max_position
    elif signal < lower:
        return 1 * max_position

def PartialThreshold(signal, max_position):
        return signal * max_position

def MeanMomentum(signal_series, max_position, window=20):
    if len(signal_series) < window:
        return 0
    rolling_mean = signal_series.rolling(window=window).mean().iloc[-1]
    current_signal = signal_series.iloc[-1]
    if current_signal > rolling_mean:
        return -1 * max_position
    else:
        return 1 * max_position

def MeanReversion(signal_series, max_position, window=20):
    if len(signal_series) < window:
        return 0
    rolling_mean = signal_series.rolling(window=window).mean().iloc[-1]
    current_signal = signal_series.iloc[-1]
    if current_signal > rolling_mean:
        return 1 * max_position
    else:
        return -1 * max_position

def long_only(signal_series, max_position):
        return 1 * max_position


from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class GARCH:
    def __init__(self, volatility, p=1, q=1):
        self.volatility = volatility
        self.model = arch_model(self.volatility, vol='Garch', p=p, q=q, rescale=True)
        self.result = None
        self.forecasted_volatility = None

    def fit(self):
        self.result = self.model.fit(disp='off')
        return self.result

    def forecast_volatility(self, horizon=1):
        if self.result is None:
            raise ValueError("GARCH model has not been fitted yet.")

        forecast = self.result.forecast(horizon=horizon, start=0)
        self.forecasted_volatility = forecast.variance.values[-len(self.volatility):]
        return self.forecasted_volatility


class MSGARCH:
    def __init__(self, volatility, p=1, q=1, k_regimes=2):
        self.volatility = volatility
        self.p = p
        self.q = q
        self.k_regimes = k_regimes
        self.msr_model = MarkovRegression(self.volatility, k_regimes=self.k_regimes, trend='c', switching_variance=True)
        self.msr_result = None
        self.garch_models = {}
        self.regime_probs = None
        self.forecasted_volatility = None

    def fit_markov_switching(self):
        self.msr_result = self.msr_model.fit(disp=False)
        self.regime_probs = self.msr_result.predicted_marginal_probabilities
        return self.msr_result, self.regime_probs

    def fit_garch_for_regimes(self):
        if self.regime_probs is None:
            raise ValueError("Markov Switching Regression model has not been fitted yet.")

        for regime in range(self.k_regimes):
            regime_volatility = self.volatility * self.regime_probs.iloc[:, regime]
            regime_volatility_clean = regime_volatility.replace([np.inf, -np.inf], np.nan).dropna()

            if len(regime_volatility_clean) == 0:
                raise ValueError(f"Regime {regime} contains only NaN or inf values after cleaning.")

            garch_model = arch_model(regime_volatility_clean, vol='Garch', p=self.p, q=self.q, rescale=True)
            self.garch_models[regime] = garch_model.fit(disp="off")

        return self.garch_models

    def forecast_volatility(self, horizon=1):
        if not self.garch_models:
            raise ValueError("GARCH models for regimes have not been fitted yet.")

        forecasted_volatility = np.zeros(len(self.regime_probs))

        for regime, model in self.garch_models.items():
            forecast = model.forecast(horizon=horizon, start=0)
            regime_forecast_vol = forecast.variance.values[-len(self.regime_probs):].flatten()

            regime_prob = self.regime_probs.iloc[:, regime]
            min_length = min(len(regime_forecast_vol), len(regime_prob))

            forecasted_volatility[:min_length] += regime_forecast_vol[:min_length] * regime_prob[:min_length]

        self.forecasted_volatility = forecasted_volatility
        return self.forecasted_volatility
