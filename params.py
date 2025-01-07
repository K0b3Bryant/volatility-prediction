
# params
split_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] # 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
train_years_num = 3
test_years_num = 3
robustness_test = 'sliding window' # 'expanding window', 'sliding window'

# Measures params
window_size = 3
high, low, close, open, = 'high_SPX', 'low_SPX', 'close_SPX', 'open_SPX'

# MSR params
probability = 'filtered_1'  # pred_smooth, filtered_1, ...  - try with filtered
labels = ['p_3d'] # 'vol_3d', 'gk_3d', 'p_3d', 'gkyz_3d', 'diff_vol_3d', 'vov_3d'
trends = ['c'] # 'ct', 'n',
exogs = [['vix_SPX', 'iv_call_SPX']] # None, ['vix_SPX', 'iv_call_SPX']
switching_variance = True
scale_endog = False
rescale_endog= True # multiply by 100
scale_exog = True
em_iter = 1000
search_reps = 100
regimes = 2
show_model = True

# prediciton
lags = [1, 2, 3, 4, 5, 10, 21, 63]

# Strategies params
all_position_funcs = [Threshold] #MeanMomentum, MeanReversion, Threshold
bound_sets = [
    {'upper': 0.99, 'lower': 0.99},
    ]
rolling_windows = [15]
params = {'transaction_costs': 0.001, 'lag': 1, 'max_position': 1} # neutralized trading costs until finding alpha

# Performance params
factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'VIX', 'BAB'] # cinsider cutting until finding alpha


############################### Execution ######################################
# store data
main_msr_probs = []
main_msr_metric = []

main_feature_importance = []
main_prediction_metrics = []

main_returns = []
main_performance = []

# date loop
for split_year in split_years:
    from datetime import datetime
    split_date = datetime(split_year, 1, 1)
    end_date = datetime(split_year+test_years_num-1, 12, 31)
    if robustness_test == 'expanding window':
        start_date = '2000-01-01'
    if robustness_test == 'sliding window':
        start_date = datetime(split_year-train_years_num, 1, 1)

    # extract years
    if isinstance(start_date, str):
        start_year = datetime.strptime(start_date, '%Y-%m-%d').year
    else:
        start_year = start_date.year

    end_year = end_date.year+1

    print('Testing on:')
    print(start_date, start_year, split_date, split_year, end_date, end_year)
