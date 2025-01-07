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
