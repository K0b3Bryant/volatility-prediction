import numpy as np
import pandas as pd

def meassures(data: pd.DataFrame, high, low, close, open, window_size, frac_diff_order=0.5) -> pd.DataFrame:
    """ Creates meassures of volatility """
    def rolling_window(array, window):
        shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
        strides = array.strides + (array.strides[-1],)
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    def close_to_close_volatility(close_prices):
        log_returns = np.log(close_prices[:, 1:] / close_prices[:, :-1])
        return np.std(log_returns, axis=1)

    def volatility_direction(volatility):
        return np.sign(np.diff(volatility, prepend=volatility[0]))

    def parkinson(hi, li):
        return np.sqrt((1 / window_size) * (1 / (4 * np.log(2))) * np.sum((np.log(hi) - np.log(li))**2, axis=1))

    def garman_klass(hi, li, ci, oi):
        return np.sqrt((1 / window_size) * (0.5 * np.sum((np.log(hi) - np.log(li))**2, axis=1) - (2 * np.log(2) - 1) * np.sum((np.log(ci) - np.log(oi))**2, axis=1)))

    def rogers_satchell(hi, li, ci, oi):
        return np.sqrt((1 / window_size) * np.sum((np.log(hi) - np.log(ci)) * (np.log(hi) - np.log(oi)) + (np.log(li) - np.log(ci)) * (np.log(li) - np.log(oi)), axis=1))

    def yang_zhang(hi, li, ci, oi):
        sigma_oc = np.sqrt((1 / (window_size - 1)) * np.sum((np.log(oi[:, 1:]) - np.log(ci[:, :-1]))**2, axis=1))
        sigma_co = np.sqrt((1 / window_size) * np.sum((np.log(ci) - np.log(oi))**2, axis=1))
        k = 0.34 / (1.34 + (window_size + 1) / (window_size - 1))
        sigma_rs = rogers_satchell(hi, li, ci, oi)
        return np.sqrt(sigma_oc**2 + k * sigma_co**2 + (1 - k) * sigma_rs**2)

    def garman_klass_yang_zhang(hi, li, ci, oi):
        term1 = (1 / window_size) * 0.5 * np.sum((np.log(oi[:, 1:]) - np.log(ci[:, :-1]))**2, axis=1)
        term2 = (1 / window_size) * 0.5 * np.sum((np.log(hi) - np.log(li))**2, axis=1)
        term3 = (1 / window_size) * (2 * np.log(2) - 1) * np.sum((np.log(ci) - np.log(oi))**2, axis=1)
        return np.sqrt(term1 + term2 - term3)

    def volatility_of_volatility(volatility_series):
        vol_windowed = rolling_window(volatility_series, window_size)
        return np.std(vol_windowed, axis=1)

    def volatility_difference(volatility_series):
        # Return difference and pad with NaN at the front to match the length
        return np.append(np.nan, np.diff(volatility_series))

    # Rolling windows
    high_windowed = rolling_window(data[high].values, window_size)
    low_windowed = rolling_window(data[low].values, window_size)
    close_windowed = rolling_window(data[close].values, window_size)
    open_windowed = rolling_window(data[open].values, window_size)

    # Initialize columns and generate volatility variables
    variable_names = [f'vol_{window_size}d', f'p_{window_size}d', f'gk_{window_size}d', f'rs_{window_size}d', f'yz_{window_size}d', f'gkyz_{window_size}d', f'vov_{window_size}d', f'diff_vol_{window_size}d']
    for name in variable_names:
        data[name] = np.nan

    data.iloc[window_size-1:, data.columns.get_loc(f'vol_{window_size}d')] = close_to_close_volatility(close_windowed)
    data.iloc[window_size-1:, data.columns.get_loc(f'p_{window_size}d')] = parkinson(high_windowed, low_windowed)
    data.iloc[window_size-1:, data.columns.get_loc(f'gk_{window_size}d')] = garman_klass(high_windowed, low_windowed, close_windowed, open_windowed)
    data.iloc[window_size-1:, data.columns.get_loc(f'rs_{window_size}d')] = rogers_satchell(high_windowed, low_windowed, close_windowed, open_windowed)
    data.iloc[window_size-1:, data.columns.get_loc(f'yz_{window_size}d')] = yang_zhang(high_windowed, low_windowed, close_windowed, open_windowed)
    data.iloc[window_size-1:, data.columns.get_loc(f'gkyz_{window_size}d')] = garman_klass_yang_zhang(high_windowed, low_windowed, close_windowed, open_windowed)

    # Calculate volatility of volatility (vol of vol)
    vol_of_vol = volatility_of_volatility(data[f'vol_{window_size}d'].dropna().values)
    data.iloc[window_size*2-2:len(vol_of_vol) + (window_size*2-2), data.columns.get_loc(f'vov_{window_size}d')] = vol_of_vol

    # Calculate difference in volatility (pad with NaN to ensure same length)
    diff_vol = volatility_difference(data[f'vol_{window_size}d'].values)
    data.iloc[:, data.columns.get_loc(f'diff_vol_{window_size}d')] = diff_vol

    # Other variables
    data['r'] = data[close].pct_change()
    data['log_r'] = np.log(data[close] / data[close].shift(1))

    return data
