import optuna

# Parameters
split_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] # years to include
train_years_num = 3                                                                   # size of train window
test_years_num = 3                                                                    # size of test window
robustness_test = 'sliding window'                                                    # type of window either 'expanding window' or 'sliding window'
window_size = 3                                                                       # size of measures generation window
high, low, close, open, = 'high_SPX', 'low_SPX', 'close_SPX', 'open_SPX'              # Define high, low, open, and close prices
probability = 'filtered_1'                                                             # Type of probabilitites in a Hidden Markov Model or Markow Switching Model
labels = ['p_3d']                                                                      # Which probabiltites to include
trends = ['c']                                                                         # Types of trends
exogs = [['vix_SPX', 'iv_call_SPX']]                                                   # Exogenous variables
switching_variance = True                                                              # ...
scale_endog = False
rescale_endog= True # multiply by 100
scale_exog = True
em_iter = 1000
search_reps = 100
regimes = 2                                                                            # Number of underlying regimes
show_model = True
lags = [1, 2, 3, 4, 5, 10, 21, 63]                                                     # Number of parameter lags
all_position_funcs = [Threshold]                                                       # Type of strategy either MeanMomentum, MeanReversion, or Threshold
bound_sets = [
    {'upper': 0.99, 'lower': 0.99},
    ]                                                                                  # Boundries of strategy switches
rolling_windows = [15]                                                                 # Size of window for stratgy switches
params = {'transaction_costs': 0.001, 'lag': 1, 'max_position': 1}                     # Trading parameters including trading costs
factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'VIX', 'BAB']             # Factors for return analysis


# Executetion - Loop over defined years for average results
main_msr_probs = []
main_msr_metric = []
main_feature_importance = []
main_prediction_metrics = []
main_returns = []
main_performance = []

for split_year in split_years:
    from datetime import datetime
    split_date = datetime(split_year, 1, 1)
    end_date = datetime(split_year+test_years_num-1, 12, 31)
    if robustness_test == 'expanding window':
        start_date = '2000-01-01'
    if robustness_test == 'sliding window':
        start_date = datetime(split_year-train_years_num, 1, 1)
    if isinstance(start_date, str):
        start_year = datetime.strptime(start_date, '%Y-%m-%d').year
    else:
        start_year = start_date.year
    end_year = end_date.year+1

    print('Testing on:')
    print(start_date, start_year, split_date, split_year, end_date, end_year)

    # run main
    main() # insert params ...

    # concat and export results
    main_msr_probs_merged = pd.concat(main_msr_probs, ignore_index=True)
    main_msr_metric_merged = pd.concat(main_msr_metric, ignore_index=True)
    main_feature_importance_merged = pd.concat(main_feature_importance, ignore_index=True)
    main_prediction_metrics_merged = pd.concat(main_prediction_metrics, ignore_index=True)
    main_returns_merged = pd.concat(main_returns, ignore_index=True)
    main_performance_merged = pd.concat(main_performance, ignore_index=True)
    main_msr_probs_merged.to_csv('main_msr_probs_merged.csv', index=False)
    main_msr_metric_merged.to_csv('main_msr_metric_merged.csv', index=False)
    main_feature_importance_merged.to_csv('main_feature_importance_merged.csv', index=False)
    main_prediction_metrics_merged.to_csv('main_prediction_metrics_merged.csv', index=False)
    main_returns_merged.to_csv('main_returns_merged.csv', index=False)
    main_performance_merged.to_csv('main_performance_merged.csv', index=False)
    print("DataFrames exported to CSV files successfully.")


# average performance over all test periods
performance = main_performance_merged.groupby('Strategy').mean()
performance = performance.reset_index()
performance.to_csv('performance.csv', index=False)

prediction_metrics = main_prediction_metrics_merged.groupby(['underlying', 'Model']).mean()
prediction_metrics = prediction_metrics.reset_index()
prediction_metrics.to_csv('prediction_metrics.csv', index=False)

msr_metrics = main_msr_metric_merged.groupby('model_name').mean()
msr_metrics = msr_metrics.reset_index()
msr_metrics.to_csv('msr_metrics.csv', index=False)

# extract buy-and-hold benchmark
buy_and_hold_df = performance[performance['Strategy'] == 'Buy and Hold']
print(buy_and_hold_df)

# add filters
#filtered_df = performance[performance['Alpha'] > 0]
#filtered_df = filtered_df[filtered_df['Alpha p-value'] < 0.05]

# add sort
#sorted_df = filtered_df.sort_values(by='Alpha', ascending=False)
#print(sorted_df)


# Visualize strategies
import matplotlib.pyplot as plt
import quantstats as qs

# Plot best strategy return
best_strategy = main_returns_merged.loc[:,'Pred_p_3d_vix_SPX_iv_call_SPX_c_lightgbm_Threshold_0.999-0.999']
print(len(best_strategy), len(test_index))
best_strategy.index = test_index
print(best_strategy)

# Generate the report
qs.extend_pandas()
report = qs.reports.full(best_strategy, benchmark='SPY')
print(report)

# Generate states of volatility
states = main_msr_probs_merged.loc[:,'p_3d_vix_SPX_iv_call_SPX_c']
states.index = data_index
print(states)

vol = data_with_variables_cleaned.loc[:,'p_3d']
print(vol)

# Define time periods
start_date = '2007-01-01'
end_date = '2008-01-01'

# import states and volatilities
states = states.loc[start_date:end_date]
vol = vol.loc[start_date:end_date]

# scaling
mean_value = vol.mean()
amplitude_factor = 10000
centered_amplified_series = (vol - mean_value) * amplitude_factor + 50

max_high = 1.5
vol_centered = centered_amplified_series / centered_amplified_series.max() * max_high

# Plot states and volatilities
plt.figure(figsize=(10, 6))
plt.plot(states.index, states, label='True Probability of High Volatility State', alpha=0.95)
plt.plot(states.index, vol_centered, label='Parkinson Vol. scaled', linewidth=0.80, color='black')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Trained State-Probabilities')
plt.legend(loc='upper right')
plt.grid(True)


# Find optimum SR
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

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(f"Best Sharpe Ratio: {study.best_value}")
print(f"Best Parameters: {study.best_params}")
