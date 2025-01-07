import optuna

def objective(trial):
    # params
    signal_name = trial.suggest_categorical('signal_name', list(signals.keys()))
    signal_rolling_window = trial.suggest_int('signal_rolling_window', 5, 21)
    strategy_func = trial.suggest_categorical('strategy_func', [mean_reversion, mean_momentum])

    strategy = Strategy('Strategy', signals[signal_name], risky_returns, risk_free_returns, transaction_costs, max_position)
    manager = StrategyManager(index=index, risk_free_returns=risk_free_returns)
    manager.add_strategy(strategy, strategy_func, lag=lag, window=signal_rolling_window)
    strategy_returns = manager.strategies['Strategy']['simple_returns']

    excess_returns = strategy_returns - risk_free_returns
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(250)

    return sharpe_ratio

# Create and run study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best Sharpe Ratio: {study.best_value}")
print(f"Best Parameters: {study.best_params}")
