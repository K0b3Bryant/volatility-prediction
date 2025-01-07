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
