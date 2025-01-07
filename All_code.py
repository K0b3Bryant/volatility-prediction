# -*- coding: utf-8 -*-

!pip install ta tsfresh torch skorch optuna quantstats ace_tools arch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.simplefilter(action='ignore', category=ValueWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

########################## Preprocessing #######################################

def csv_reader(file_path: str):
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path, nrows=0)
        date_column = None

        for col in temp_df.columns:
            if col.lower() == 'date':
                date_column = col
                break

        if date_column:
            return pd.read_csv(file_path, parse_dates=[date_column], index_col=date_column)
            print('file read with date column as index')
        else:
            print('file read raw')
            return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


def export_csv(dataframe: pd.DataFrame, filename: str):
    dataframe.to_csv(f"{filename}.csv", index=False)


import pandas as pd
import numpy as np
def cleaner(data: pd.Series or pd.DataFrame, backfill=False, verbose=False) -> pd.DataFrame or pd.Series:
    data.sort_index(inplace=True)
    is_dataframe = isinstance(data, pd.DataFrame)

    # Check and print for duplicates
    duplicates = data[data.index.duplicated(keep=False)]
    if not duplicates.empty:
        print(f"The {'DataFrame' if is_dataframe else 'Series'} contains duplicates.")
        if verbose:
            print("Duplicates:\n", duplicates)

    # Remove duplicates by taking the mean of duplicated rows
    data = data.groupby(data.index).mean()

    # Replace inf with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check for infinite or excessively large values
    if is_dataframe:
        if (data.abs() > 1e10).any().any():
            print("Data contains excessively large values.")
            data[data.abs() > 1e10] = np.nan
    else:
        if (data.abs() > 1e10).any():
            print("Series contains excessively large values.")
            data[data.abs() > 1e10] = np.nan

    # Forward fill NaN values
    data = data.ffill()
    if backfill:
        data = data.bfill()

    # Remove columns and rows that are all NaN
    if is_dataframe:
        data.dropna(how='all', axis=1, inplace=True)
        data.dropna(how='all', axis=0, inplace=True)
    else:
        data.dropna(inplace=True)

    return data

def check_and_print_duplicate_indices(data: pd.DataFrame):
    duplicate_indices = data.index[data.index.duplicated(keep=False)]
    if not duplicate_indices.empty:
        print("Duplicate indices found:")
        print(data.loc[duplicate_indices])
    else:
        print("No duplicate indices found.")


def meassures(data: pd.DataFrame, high, low, close, open, window_size, frac_diff_order=0.5) -> pd.DataFrame:
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



################################ Labels ########################################

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from numpy.linalg import LinAlgError

# Scaling function
def scale_any_data(data, trained_scaler=None):
    """ Scales pandas Series, DataFrames, and numpy arrays """

    if trained_scaler is None:
        scaler = StandardScaler()  # Use StandardScaler if no scaler is passed
    else:
        scaler = trained_scaler  # Use the provided scaler

    if isinstance(data, pd.Series):
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_data, index=data.index, name=data.name)
        return scaled_series, scaler

    elif isinstance(data, pd.DataFrame):
        scaled_data = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
        return scaled_df, scaler

    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    else:
        raise TypeError("Input data must be a pandas Series, DataFrame, or numpy array")


def probabilities(data, endog, selected_model, train_index, test_index, scale_endog=True, scale_exog=True, exog=None, msr_params=None, show_model=False, rescale_endog=False):

    def clean_and_scale(data, index, variable, scale_data, trained_scaler=None, rescale=False):
        cleaned = cleaner(data.loc[index, variable], backfill=True)  # Assume `cleaner` is defined elsewhere

        # Apply rescaling if needed
        if rescale:
            cleaned = cleaned * 100  # Rescale the endogenous variable by 100

        if scale_data:
            scaled, fitted_scaler = scale_any_data(cleaned, trained_scaler)
        else:
            scaled = cleaned
            fitted_scaler = None
        return scaled, fitted_scaler

    # Endogenous variable
    endog_train_scaled, endog_train_scaler = clean_and_scale(data, train_index, endog, scale_endog, rescale=rescale_endog)
    endog_test_scaled, _ = clean_and_scale(data, test_index, endog, scale_endog, endog_train_scaler, rescale=rescale_endog)

    # Exogenous variable, if provided
    if exog:
        exog_train_scaled, exog_train_scaler = clean_and_scale(data, train_index, exog, scale_exog)
        exog_test_scaled, _ = clean_and_scale(data, test_index, exog, scale_exog, exog_train_scaler)
    else:
        exog_train_scaled = exog_test_scaled = None

    results = pd.DataFrame()
    aic = bic = log_likelihood = None

    if selected_model == 'msr':
        # Set default MSR parameters if none are provided
        if msr_params is None:
            msr_params = {'k_regimes': 2, 'switching_variance': True, 'trend': 'c'}

        # Unpack the MSR parameters
        k_regimes = msr_params.get('k_regimes', 2)
        switching_variance = msr_params.get('switching_variance', True)
        trend = msr_params.get('trend', 'c')

        # Create and fit the model with custom parameters
        model_train = MarkovRegression(
            endog_train_scaled,
            exog=exog_train_scaled,
            k_regimes=k_regimes,
            switching_variance=switching_variance,
            trend=trend
        )

        try:
            model_train_fitted = model_train.fit(em_iter=500, search_reps=20)

            # Check if the model converged
            if not model_train_fitted.mle_retvals.get('converged', False):
                print(f"Warning: Maximum likelihood estimation did not converge for model {endog}_{exog}. Skipping...")
                return results, False, aic, bic, log_likelihood

            # Extract AIC, BIC, and Log Likelihood
            aic = model_train_fitted.aic
            bic = model_train_fitted.bic
            log_likelihood = model_train_fitted.llf

            if show_model:
                print(model_train_fitted.summary())

            test_model = MarkovRegression(
                endog_test_scaled,
                exog=exog_test_scaled,
                k_regimes=k_regimes,
                switching_variance=switching_variance,
                trend=trend
            )
            test_model_fitted = test_model.smooth(model_train_fitted.params)

            for regime in range(k_regimes):
                results[f'predicted_{regime}'] = pd.concat([
                    pd.Series(model_train_fitted.predicted_marginal_probabilities[regime], index=train_index),
                    pd.Series(test_model_fitted.predicted_marginal_probabilities[regime], index=test_index)
                ])
                results[f'filtered_{regime}'] = pd.concat([
                    pd.Series(model_train_fitted.filtered_marginal_probabilities[regime], index=train_index),
                    pd.Series(test_model_fitted.filtered_marginal_probabilities[regime], index=test_index)
                ])
                results[f'smoothed_{regime}'] = pd.concat([
                    pd.Series(model_train_fitted.smoothed_marginal_probabilities[regime], index=train_index),
                    pd.Series(test_model_fitted.smoothed_marginal_probabilities[regime], index=test_index)
                ])
                results[f'smoothed_pred_{regime}'] = pd.concat([
                    pd.Series(model_train_fitted.smoothed_marginal_probabilities[regime], index=train_index),
                    pd.Series(test_model_fitted.predicted_marginal_probabilities[regime], index=test_index)
                ])
                results[f'smoothed_filt_{regime}'] = pd.concat([
                    pd.Series(model_train_fitted.smoothed_marginal_probabilities[regime], index=train_index),
                    pd.Series(test_model_fitted.filtered_marginal_probabilities[regime], index=test_index)
                ])

        except (LinAlgError, RuntimeError) as e:
            print(f"Skipping {endog}_{exog} due to an error: {e}")
            return results, False, aic, bic, log_likelihood

    else:
        raise ValueError("Currently, only 'msr' models are supported.")

    return results, True, aic, bic, log_likelihood


################################################################################
################################ New Predictor #################################
################################################################################
import os
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging

class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1, activation_function=nn.ReLU):
        super(ANN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_function())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

class UniversalRegressor:
    def __init__(self, model_name, use_pca=False, use_scaler=False, pca_n_components=None, **kwargs):
        self.model_name = model_name.lower()
        self.model = None
        self.kwargs = kwargs
        self.pipeline_steps = []
        if use_scaler:
            self.pipeline_steps.append(('scaler', StandardScaler()))
        if use_pca:
            self.pipeline_steps.append(('pca', PCA(n_components=pca_n_components)))

        self._initialize_model()
        self.pipeline = Pipeline(self.pipeline_steps)

    def _initialize_model(self):
        if self.model_name == 'linreg' or self.model_name.startswith('linreg_'):
            self.model = LinearRegression(**self.kwargs)
        elif self.model_name == 'ridge' or self.model_name.startswith('ridge_'):
            self.model = Ridge(**self.kwargs)
        elif self.model_name == 'svr' or self.model_name.startswith('svr_'):
            self.model = SVR(**self.kwargs)
        elif self.model_name == 'rf' or self.model_name.startswith('rf_'):
            self.model = RandomForestRegressor(**self.kwargs)
        elif self.model_name == 'adaboost' or self.model_name.startswith('adaboost_'):
            self.model = AdaBoostRegressor(**self.kwargs)
        elif self.model_name == 'xgb' or self.model_name.startswith('xgb_'):
            self.model = XGBRegressor(**self.kwargs)
        elif self.model_name == 'lightgbm' or self.model_name.startswith('lightgbm_'):
            self.model = LGBMRegressor(**self.kwargs)
        elif self.model_name == 'ann' or self.model_name.startswith('ann_'):
            activation_function = self.kwargs.get('activation_function', nn.ReLU)
            self.model = NeuralNetRegressor(
                ANN(input_dim=self.kwargs['input_dim'], hidden_dims=self.kwargs['hidden_dims'], activation_function=activation_function),
                max_epochs=self.kwargs.get('epochs', 100),
                lr=self.kwargs.get('lr', 0.001),
                optimizer=optim.Adam,
                criterion=nn.MSELoss,
                train_split=None,
                verbose=0
            )
        elif self.model_name == 'lstm':
            self.model = NeuralNetRegressor(
                LSTMModel(input_dim=self.kwargs['input_dim'], hidden_dim=self.kwargs['hidden_dim'],
                          num_layers=self.kwargs['num_layers'], output_dim=self.kwargs.get('output_dim', 1)),
                max_epochs=self.kwargs.get('epochs', 100),
                lr=self.kwargs.get('lr', 0.001),
                optimizer=optim.Adam,
                criterion=nn.MSELoss,
                train_split=None,
                verbose=0
            )
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

        self.pipeline_steps.append(('model', self.model))

    def fit(self, X, y):
        if isinstance(self.model, NeuralNetRegressor):
            X = X.astype(np.float32)
            y = y.values.astype(np.float32).reshape(-1, 1)
        self.pipeline.fit(X, y)

    def predict(self, X):
        if isinstance(self.model, NeuralNetRegressor):
            X = X.astype(np.float32)
        return self.pipeline.predict(X)

class PredictionManager:
    def __init__(self):
        self.models = {}
        self.predicted_signals = pd.DataFrame()
        self.scores = pd.DataFrame(columns=['Model', 'RMSE', 'Efron’s R²'])
        self.scores = pd.DataFrame(columns=['Model', 'RMSE', 'Efron’s R²'])
        self.feature_importance_df = pd.DataFrame()

    def generate_ta_features(self, data: pd.DataFrame, high, low, close, open, volume, cmo_window: int=14):
        def cmo(close_prices, window=14):
            delta = close_prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            sum_gain = gain.rolling(window=window, min_periods=1).sum()
            sum_loss = loss.rolling(window=window, min_periods=1).sum()
            return ((sum_gain - sum_loss) / (sum_gain + sum_loss)).abs() * 100

        features = data.loc[:, [close, open, high, low, volume]]
        features.sort_index(inplace=True)
        calculations = {
            'kama': lambda: ta.momentum.KAMAIndicator(features[close]).kama(),
            'ppo': lambda: ta.momentum.PercentagePriceOscillator(features[close]).ppo(),
            'pvo': lambda: ta.momentum.PercentageVolumeOscillator(features[volume]).pvo(),
            'roc': lambda: ta.momentum.ROCIndicator(features[close]).roc(),
            'rsi': lambda: ta.momentum.RSIIndicator(features[close]).rsi(),
            'srsi': lambda: ta.momentum.StochRSIIndicator(features[close]).stochrsi(),
            'stoch': lambda: ta.momentum.StochasticOscillator(features[high], features[low], features[close]).stoch(),
            'tsi': lambda: ta.momentum.TSIIndicator(features[close]).tsi(),
            'uo': lambda: ta.momentum.UltimateOscillator(features[high], features[low], features[close]).ultimate_oscillator(),
            'willr': lambda: ta.momentum.WilliamsRIndicator(features[high], features[low], features[close]).williams_r(),
            'ao': lambda: ta.momentum.AwesomeOscillatorIndicator(features[high], features[low]).awesome_oscillator(),
            'ppo_hist': lambda: ta.momentum.PercentagePriceOscillator(features[close]).ppo_hist(),
            'ppo_signal': lambda: ta.momentum.PercentagePriceOscillator(features[close]).ppo_signal(),
            'aci': lambda: ta.volume.AccDistIndexIndicator(features[high], features[low], features[close], features[volume]).acc_dist_index(),
            'chaikin': lambda: ta.volume.ChaikinMoneyFlowIndicator(features[high], features[low], features[close], features[volume]).chaikin_money_flow(),
            'fi': lambda: ta.volume.ForceIndexIndicator(features[close], features[volume]).force_index(),
            'mfi': lambda: ta.volume.MFIIndicator(features[high], features[low], features[close], features[volume]).money_flow_index(),
            'nvi': lambda: ta.volume.NegativeVolumeIndexIndicator(features[close], features[volume]).negative_volume_index(),
            'obv': lambda: ta.volume.OnBalanceVolumeIndicator(features[close], features[volume]).on_balance_volume(),
            'vpt': lambda: ta.volume.VolumePriceTrendIndicator(features[close], features[volume]).volume_price_trend(),
            'vi': lambda: ta.volume.VolumeWeightedAveragePrice(features[high], features[low], features[close], features[volume]).volume_weighted_average_price(),
            'atr': lambda: ta.volatility.AverageTrueRange(features[high], features[low], features[close]).average_true_range(),
            'bb_bbm': lambda: ta.volatility.BollingerBands(features[close]).bollinger_mavg(),
            'bb_bbh': lambda: ta.volatility.BollingerBands(features[close]).bollinger_hband(),
            'bb_bbl': lambda: ta.volatility.BollingerBands(features[close]).bollinger_lband(),
            'dc': lambda: ta.volatility.DonchianChannel(features[high], features[low], features[close]).donchian_channel_lband(),
            'kc': lambda: ta.volatility.KeltnerChannel(features[high], features[low], features[close]).keltner_channel_lband(),
            'ui': lambda: ta.volatility.UlcerIndex(features[close]).ulcer_index(),
            'adx': lambda: ta.trend.ADXIndicator(features[high], features[low], features[close]).adx(),
            'ai': lambda: ta.trend.AroonIndicator(features[high], features[low]).aroon_indicator(),
            'cci': lambda: ta.trend.CCIIndicator(features[high], features[low], features[close]).cci(),
            'kst': lambda: ta.trend.KSTIndicator(features[close]).kst(),
            'kst_sig': lambda: ta.trend.KSTIndicator(features[close]).kst_sig(),
            'macd': lambda: ta.trend.MACD(features[close]).macd(),
            'macd_signal': lambda: ta.trend.MACD(features[close]).macd_signal(),
            'macd_diff': lambda: ta.trend.MACD(features[close]).macd_diff(),
            'psar': lambda: ta.trend.PSARIndicator(features[high], features[low], features[close]).psar(),
            'sma_30': lambda: ta.trend.SMAIndicator(features[close], window=30).sma_indicator(),
            'sma_50': lambda: ta.trend.SMAIndicator(features[close], window=50).sma_indicator(),
            'sma_100': lambda: ta.trend.SMAIndicator(features[close], window=100).sma_indicator(),
            'sma_200': lambda: ta.trend.SMAIndicator(features[close], window=200).sma_indicator(),
            'trix': lambda: ta.trend.TRIXIndicator(features[close]).trix(),
            'ema': lambda: ta.trend.EMAIndicator(features[close]).ema_indicator(),
            'ii': lambda: ta.trend.IchimokuIndicator(features[high], features[low]).ichimoku_a(),
            'mi': lambda: ta.trend.MassIndex(features[high], features[low]).mass_index(),
            'cmo': lambda: cmo(features[close], window=cmo_window)
        }

        for name, calc in calculations.items():
            try:
                features[name] = calc()
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")

        features.drop(columns=[close, open, high, low, volume], inplace=True)
        return features

    def generate_ts_features(self, data: pd.DataFrame, underlying: str, max_timeshift: int = 3, min_timeshift: int = 3):
        file_name = f'features_{underlying}_{max_timeshift}d'
        path = f'{file_name}.csv'
        if os.path.exists(path):
            print(f'Found precomputed features at: {path}')
            return csv_reader(path)

        if underlying not in data.columns:
            raise ValueError(f"Column '{underlying}' not found in DataFrame.")

        print('Generating TS features...')
        features = pd.DataFrame({'date': data.index, f'{underlying}': data[underlying], 'id': 1}).reset_index(drop=True)
        roller = roll_time_series(features, column_id="id", column_sort="date", min_timeshift=min_timeshift, max_timeshift=max_timeshift, rolling_direction=1)
        roller = roller.fillna(0)
        logging.basicConfig(level=logging.INFO)
        extracted_features = extract_features(roller, column_id='id', column_sort='date', column_value=underlying, impute_function=impute, show_warnings=False, n_jobs=1) #n_jobs=1
        extracted_features.index = extracted_features.index.droplevel(0)
        extracted_features.index.name = 'date'
        #extracted_features.to_csv(path) # don't export csv
        return extracted_features

    def lag_features(self, df: pd.DataFrame, lags: list):
        lagged_dfs = []
        for lag in lags:
            lagged_df = df.shift(lag)
            lagged_df.columns = [f"{col}_lag({lag})" for col in df.columns]
            lagged_dfs.append(lagged_df)
        return pd.concat(lagged_dfs, axis=1)

    def prepare_features(self, X: pd.DataFrame, ta_params=None, ts_params=None, lags: list = None):
        features = X.copy()

        if ta_params:
            print("Generating TA features...")
            ta_features = self.generate_ta_features(X, **ta_params)
            features = pd.merge(features, ta_features, left_index=True, right_index=True, how='outer')

        if ts_params and ts_params['underlying'] in X.columns:
            print("Generating TS features...")
            ts_features = self.generate_ts_features(X, **ts_params)
            features = pd.merge(features, ts_features, left_index=True, right_index=True, how='outer')

        if lags:
            print("Applying lag features...")
            features = self.lag_features(features, lags)

        features = features.reindex(X.index)
        features.fillna(method='ffill', inplace=True)
        features.fillna(method='bfill', inplace=True)
        return features

    def add_model(self, model_name, **kwargs):
        self.models[model_name] = UniversalRegressor(model_name, **kwargs)

    def reshape_for_lstm(X, sequence_length):
        X_lstm = []
        for i in range(sequence_length, len(X)):
            X_lstm.append(X[i-sequence_length:i].values)
        return np.array(X_lstm)

    def train(self, X, y, train_index):
        X_train = X.loc[train_index]
        y_train = y.loc[X_train.index].copy()

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            if model_name == 'lstm':
                sequence_length = 21 # trading month
                X_train_lstm = reshape_for_lstm(X_train, sequence_length)
                y_train_lstm = y_train[sequence_length:].values.astype(np.float32)
                model.fit(X_train_lstm.astype(np.float32), y_train_lstm)
            else:
                model.fit(X_train, y_train)

    def predict(self, X, test_index):
        X_test = X.loc[test_index]

        for model_name, model in self.models.items():
            print(f"Predicting with {model_name}...")
            if model_name == 'lstm':
                sequence_length = 21  # trading month
                X_test_lstm = reshape_for_lstm(X_test, sequence_length)
                predictions = model.predict(X_test_lstm.astype(np.float32))
                predictions = pd.Series(predictions.ravel(), index=X_test.index[sequence_length:])
            else:
                predictions = pd.Series(model.predict(X_test).ravel(), index=X_test.index)

            self.predicted_signals[model_name] = predictions

    def extract_predicted_signals(self, model_name=None):
        if model_name:
            if model_name in self.predicted_signals.columns:
                return self.predicted_signals[[model_name]]
            else:
                raise ValueError(f"Model '{model_name}' not found in predicted signals.")
        else:
            return self.predicted_signals

    def evaluate(self, y, test_index):
        y_test = y.loc[test_index]

        for model_name in self.predicted_signals.columns:
            predictions = self.predicted_signals[model_name]
            predictions = predictions.loc[test_index]

            rmse = mean_squared_error(y_test, predictions, squared=False)
            efrons_r2 = self.efrons_r2(y_test, predictions)

            self.scores = pd.concat([self.scores, pd.DataFrame({
                'Model': [model_name],
                'RMSE': [rmse],
                'Efron’s R²': [efrons_r2]
            })], ignore_index=True)

    def evaluate_train(self, y, train_index):
        y_test = y.loc[train_index]

        for model_name in self.predicted_signals.columns:
            predictions = self.predicted_signals[model_name]
            predictions = predictions.loc[train_index]

            rmse = mean_squared_error(y_test, predictions, squared=False)
            efrons_r2 = self.efrons_r2(y_test, predictions)

            self.scores = pd.concat([self.scores, pd.DataFrame({
                'Model': [model_name],
                'RMSE': [rmse],
                'Efron’s R²': [efrons_r2]
            })], ignore_index=True)

    def efrons_r2(self, y, y_pred):
        y_mean = np.mean(y)
        numerator = np.sum((y - y_pred) ** 2)
        denominator = np.sum((y - y_mean) ** 2)
        r2 = 1 - (numerator / denominator)
        return r2

    def plot_residuals(self, y, test_index):
        y_test = y.loc[test_index]

        for model_name in self.predicted_signals.columns:
            predictions = self.predicted_signals[model_name]
            predictions = predictions.loc[test_index]

            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, predictions, alpha=0.5, label='Predicted vs True')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='y = x')  # Line y = x
            plt.title(f'Predicted vs True Values for {model_name}')
            plt.ylabel('Predicted Values')
            plt.xlabel('True Values')
            plt.legend()
            plt.show()

    def feature_importance(self, X):
        importance_data = []
        for model_name, model in self.models.items():
            if hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                features = X.columns
                sorted_indices = importances.argsort()[::-1][:5]
                for idx in sorted_indices:
                    importance_data.append({
                        'Model': model_name,
                        'Feature': features[idx],
                        'Importance': importances[idx]
                    })
            elif hasattr(model.model, 'coef_'):
                importances = np.abs(model.model.coef_).ravel()
                features = X.columns
                sorted_indices = importances.argsort()[::-1][:5]
                for idx in sorted_indices:
                    importance_data.append({
                        'Model': model_name,
                        'Feature': features[idx],
                        'Importance': importances[idx]
                    })
            else:
                print(f"{model_name} does not support feature importance.")

        self.feature_importance_df = pd.DataFrame(importance_data)

################################ New Strategies ################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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



################################################################################

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


    ############################ Preprocess ####################################
    data = csv_reader('/content/data.csv')
    data_cleaned = cleaner(data, backfill=False, verbose=False)
    #print(data_cleaned)

    data_index = data_cleaned[start_date:end_date].index
    train_index = data_cleaned[start_date:(pd.to_datetime(split_date) - pd.Timedelta(days=1))].index
    test_index = data_cleaned[split_date:end_date].index
    #print(data_index, train_index, test_index)

    # generate volatility meassures
    data_with_variables = meassures(data_cleaned.loc[data_index,:], high, low, close, open, window_size)
    data_with_variables_cleaned = cleaner(data_with_variables, backfill=True, verbose=False) # backfilling applied
    #print(data_with_variables_cleaned)### Model Volatility with MSR


    ############################# Probabilities ################################
    # store data
    state_probabilities = pd.DataFrame()
    msr_metrics = []

    # loop over models
    for trend in trends:
        msr_params = {
            'k_regimes': regimes,
            'switching_variance': switching_variance,
            'trend': trend,
            'em_iter': em_iter,
            'search_reps': search_reps}

        for label in labels:
            for exog in exogs:
                exog_name = '_'.join(exog) if exog else 'no_exog'
                name = f'{label}_{exog_name}_{trend}'

                print(f'Predicting Probabilities for {name}')
                probs, converged, aic, bic, log_likelihood = probabilities(data_with_variables_cleaned, label, 'msr', train_index, test_index, scale_endog, scale_exog, exog, msr_params, show_model, rescale_endog)

                if not converged:
                    print(f"Warning: Model did not converge for {label} with {exog}")

                if converged and probability in probs:
                    state_probabilities[name] = probs[probability]
                    msr_metrics.append({'model_name': name, 'aic': aic, 'bic': bic, 'log_likelihood': log_likelihood})
                elif not converged:
                    print(f"Skipping {name} due to lack of convergence.")
                else:
                    print(f"Warning: {probability} not found in probs")

    # append to main
    state_probabilities = cleaner(state_probabilities, backfill=True, verbose=False)
    main_msr_probs.append(state_probabilities)
    #print(filtered_state_probabilities)

    msr_metrics = pd.DataFrame(msr_metrics)
    main_msr_metric.append(msr_metrics)
    #pritn(model_metrics_df)


    ########################### Prediction #####################################

    # extract data
    underlyings =  list(state_probabilities.columns) #+['vol_3d', 'p_3d', 'rs_3d', 'gk_3d', 'yz_3d', 'gkyz_3d'] apply for predicted RV

    # store data
    all_predictions = pd.DataFrame()
    all_metrics = pd.DataFrame()
    all_feature_importance = pd.DataFrame()

    # loop over models
    for underlying in underlyings:
        print(f'Initiating prediction for {underlying}')

        X = pd.merge(data_with_variables_cleaned, state_probabilities, left_index=True, right_index=True, how='inner') #X = data_with_variables_cleaned for RV
        y = X.loc[:, underlying]

        manager = PredictionManager()

        # features
        ta_params = {'high': 'high_SPX', 'low': 'low_SPX', 'close': 'close_SPX', 'open': 'open_SPX', 'volume': 'volume_SPX'}
        ts_params = {'underlying': underlying, 'max_timeshift': 3, 'min_timeshift': 3}
        X_transformed = manager.prepare_features(X, lags=lags) # no ta or ts params given thus no feature generated
        X_transformed =  cleaner(X_transformed, backfill=True, verbose=False)
        input_dim = X_transformed.shape[1]

        # add models
        #manager.add_model('linreg', use_scaler=True, use_pca=False)
        #manager.add_model('ridge', use_scaler=True, use_pca=False)
        #manager.add_model('rf', use_scaler=False, use_pca=False)
        #manager.add_model('adaboost', use_scaler=False, use_pca=False)
        #manager.add_model('xgb', use_scaler=False, use_pca=False)
        manager.add_model('lightgbm', use_scaler=False, use_pca=False, n_estimators=100, learning_rate=0.1)
        #manager.add_model('lightgbm_high_lr', use_scaler=False, use_pca=False, n_estimators=100, learning_rate=0.5)
        #manager.add_model('lightgbm_low_lr', use_scaler=False, use_pca=False, n_estimators=100, learning_rate=0.05)
        manager.add_model('ann', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.001, use_scaler=True)
        #manager.add_model('ann_high_lr', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.005, use_scaler=True)
        #manager.add_model('ann_low_lr', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.0005, use_scaler=True)
        #manager.add_model('ann_sigmoid', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.001, use_scaler=True, activation_function=nn.Sigmoid)
        #manager.add_model('ann_sigmoid_high_lr', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.005, use_scaler=True, activation_function=nn.Sigmoid)
        #manager.add_model('ann_sigmoid_low_lr', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, lr=0.0005, use_scaler=True, activation_function=nn.Sigmoid)
        #manager.add_model('lstm', input_dim=input_dim, hidden_dim=64, num_layers=2, epochs=100, lr=0.001)

        # train
        manager.train(X_transformed, y, train_index)

        # predict
        manager.predict(X_transformed, data_index)

        # metrics
        manager.evaluate_train(y, train_index)
        metrics = manager.scores.copy()
        metrics['underlying'] = underlying
        all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)

        # residuals
        manager.plot_residuals(y, test_index)

        # feature importance
        manager.feature_importance(X_transformed)
        feauture_importance = manager.feature_importance_df.copy()
        feauture_importance['underlying'] = underlying
        all_feature_importance = pd.concat([all_feature_importance, feauture_importance], ignore_index=True)

        # predictions
        predictions = manager.extract_predicted_signals()
        predictions.columns = [f'{underlying}_{col}' for col in predictions.columns]
        all_predictions = pd.concat([all_predictions, predictions], axis=1)

    # append to main
    main_feature_importance.append(all_feature_importance)
    main_prediction_metrics.append(all_metrics)

    # export
    #all_metrics.to_csv("model_performance_metrics.csv", index=False)
    #all_feature_importance.to_csv("feature_importance.csv", index=False)


    ################################ Strategies ################################

    # extract data
    risky_returns = data_with_variables_cleaned.loc[data_index, ['r']]
    risk_free_returns = pd.Series(0, index=data_index, name='risk_free_returns')

    signals = {}
    #signals.update({f'RV_{key}': data_with_variables_cleaned.loc[data_index, [key]] for key in ['vol_3d', 'p_3d', 'rs_3d', 'gk_3d', 'yz_3d', 'gkyz_3d', 'vix_SPX']})
    #signals.update({f'Prob_{col}': filtered_state_probabilities.loc[data_index, [col]] for col in filtered_state_probabilities.columns})
    signals.update({f'Pred_{col}': all_predictions.loc[data_index, [col]] for col in all_predictions.columns})

    # loop over strategies
    manager = StrategyManager(risk_free_returns=risk_free_returns.loc[data_index])

    for signal_name, signal_data in signals.items():
        position_funcs = [MeanReversion, MeanMomentum] if 'RV' in signal_name else all_position_funcs

        for func in position_funcs:
            if func in [MeanReversion, MeanMomentum]:
                for window in rolling_windows:

                    strategy_name = f"{signal_name}_{func.__name__}_{window}d"
                    print(f'Adding Strategy: {strategy_name}')

                    strategy = Strategy(
                        name=strategy_name,
                        signals=signal_data.loc[data_index],
                        risky_returns=risky_returns.loc[data_index],
                        risk_free_returns=risk_free_returns.loc[data_index],
                        transaction_costs=params['transaction_costs'],
                        max_position=params['max_position'])
                    manager.add_strategy(strategy, position_func=func, lag=params['lag'], window=window)

            else:
                for bounds in bound_sets:
                    upper = bounds['upper']
                    lower = bounds['lower']

                    strategy_name = f"{signal_name}_{func.__name__}_{lower}-{upper}"
                    print(f'Adding Strategy: {strategy_name}')

                    strategy = Strategy(
                        name=strategy_name,
                        signals=signal_data.loc[data_index],
                        risky_returns=risky_returns.loc[data_index],
                        risk_free_returns=risk_free_returns.loc[data_index],
                        transaction_costs=params['transaction_costs'],
                        max_position=params['max_position'])
                    manager.add_strategy(strategy, position_func=func, lag=params['lag'], upper=upper, lower=lower)

    # Buy and Hold
    buy_and_hold_strategy = Strategy(
        name='Buy and Hold',
        signals=pd.Series(1, index=data_index, name='Buy and Hold Signal'),
        risky_returns=risky_returns.loc[data_index],
        risk_free_returns=risk_free_returns.loc[data_index],
        transaction_costs=params['transaction_costs'],
        max_position=params['max_position']
    )
    print(f'Adding Benchmark: Buy and Hold')
    manager.add_benchmark(buy_and_hold_strategy, position_func=long_only)

    # GARCH
    garch_variable = cleaner(risky_returns['r'], backfill=True)
    garch_strategy = GARCH(garch_variable)
    garch_strategy.fit()
    garch_forecasted_vol = garch_strategy.forecast_volatility(horizon=1)
    garch_signal = pd.Series(garch_forecasted_vol.flatten(), index=data_index[-len(garch_forecasted_vol):], name='GARCH Signal')
    print(f'Adding Benchmark: GARCH')
    for window in rolling_windows:
        #garch_strategy_reversion = Strategy(
        #    name=f'GARCH_Volatility_MeanReversion_{window}d',
        #    signals=garch_signal,
        #    risky_returns=risky_returns.loc[data_index],
        #    risk_free_returns=risk_free_returns.loc[data_index],
        #    transaction_costs=params['transaction_costs'],
        #    max_position=params['max_position']
        #)
        #manager.add_benchmark(garch_strategy_reversion, position_func=MeanReversion, lag=params['lag'], window=window)

        garch_strategy_momentum = Strategy(
            name=f'GARCH_Volatility_MeanMomentum_{window}d',
            signals=garch_signal,
            risky_returns=risky_returns.loc[data_index],
            risk_free_returns=risk_free_returns.loc[data_index],
            transaction_costs=params['transaction_costs'],
            max_position=params['max_position']
        )
        manager.add_benchmark(garch_strategy_momentum, position_func=MeanMomentum, lag=params['lag'], window=window)

    # MS-GARCH
    ms_garch_variable = cleaner(risky_returns['r'], backfill=True)
    ms_garch = MSGARCH(ms_garch_variable, p=1, q=1, k_regimes=2)
    ms_res, regime_probs = ms_garch.fit_markov_switching()
    garch_fits = ms_garch.fit_garch_for_regimes()
    forecasted_volatility = ms_garch.forecast_volatility(horizon=1)
    ms_garch_signal = pd.Series(garch_forecasted_vol.flatten(), index=data_index[-len(garch_forecasted_vol):], name='MS-GARCH Signal')
    print(f'Adding Benchmark: MS-GARCH')
    for window in rolling_windows:
        #ms_garch_strategy_reversion = Strategy(
        #    name=f'MS-GARCH_Volatility_MeanReversion_{window}d',
        #    signals=ms_garch_signal,
        #    risky_returns=risky_returns.loc[data_index],
        #    risk_free_returns=risk_free_returns.loc[data_index],
        #    transaction_costs=params['transaction_costs'],
        #    max_position=params['max_position']
        #)
        #manager.add_benchmark(ms_garch_strategy_reversion, position_func=MeanReversion, lag=params['lag'], window=window)

        ms_garch_strategy_momentum = Strategy(
            name=f'MS-GARCH_Volatility_MeanMomentum_{window}d',
            signals=ms_garch_signal,
            risky_returns=risky_returns.loc[data_index],
            risk_free_returns=risk_free_returns.loc[data_index],
            transaction_costs=params['transaction_costs'],
            max_position=params['max_position']
        )
        manager.add_benchmark(ms_garch_strategy_momentum, position_func=MeanMomentum, lag=params['lag'], window=window)

    # VIX scaled
    vix_spx = data_with_variables_cleaned['vix_SPX']
    vix_spx_scaled, vix_scaler = scale_any_data(vix_spx)

    for window in rolling_windows:
        vix_reversion_strategy = Strategy(
            name=f'Scaled_VIX_MeanReversion_{window}d',
            signals=vix_spx_scaled,
            risky_returns=risky_returns.loc[data_index],
            risk_free_returns=risk_free_returns.loc[data_index],
            transaction_costs=params['transaction_costs'],
            max_position=params['max_position']
        )
        manager.add_benchmark(vix_reversion_strategy, position_func=MeanReversion, lag=params['lag'], window=window)

        #vix_momentum_strategy = Strategy(
        #    name=f'Scaled_VIX_MeanMomentum_{window}d',
        #    signals=vix_spx_scaled,
        #    risky_returns=risky_returns.loc[data_index],
        #    risk_free_returns=risk_free_returns.loc[data_index],
        #    transaction_costs=params['transaction_costs'],
        #    max_position=params['max_position']
        #)
        #manager.add_benchmark(vix_momentum_strategy, position_func=MeanMomentum, lag=params['lag'], window=window)

    ############################## Performance #################################

    # recalculate
    manager.calculate_strategies(start_date=split_date, end_date=end_date)

    # performance
    performance_df = manager.generate_performance_table(factors_list=factors_list, strategy_names=None, include_benchmarks=True)
    #print(performance_df)

    # out-performance
    buy_and_hold_return = performance_df.loc[performance_df['Strategy'] == 'Buy and Hold', 'Total Return'].values[0]
    performance_df['Outperform_BuyAndHold'] = np.where(performance_df['Total Return'] > buy_and_hold_return, 1, 0)
    performance_df['Converged'] = 1

    # plot
    manager.plot_results(plot_name=f"Performance from {split_year} to {end_year}", include_benchmarks=True)

    # extract strategies
    strategies_and_benches = manager.get_calculated_strategies_and_benchmarks()

   # simple returns
    simple_returns_df = pd.DataFrame()
    for strategy, data in strategies_and_benches['strategies'].items():
        simple_returns_df[strategy] = data['simple_returns']
    for benchmark, benchmark_data in strategies_and_benches['benchmarks'].items():
        simple_returns_df[benchmark] = benchmark_data['simple_returns']
    #print(simple_returns_df)

    # append to main
    main_returns.append(simple_returns_df)
    main_performance.append(performance_df)

################################# Data #########################################
# concat
main_msr_probs_merged = pd.concat(main_msr_probs, ignore_index=True)
main_msr_metric_merged = pd.concat(main_msr_metric, ignore_index=True)
main_feature_importance_merged = pd.concat(main_feature_importance, ignore_index=True)
main_prediction_metrics_merged = pd.concat(main_prediction_metrics, ignore_index=True)
main_returns_merged = pd.concat(main_returns, ignore_index=True)
main_performance_merged = pd.concat(main_performance, ignore_index=True)


# export
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

"""

---

Plots


---

"""

best_strategy = main_returns_merged.loc[:,'Pred_p_3d_vix_SPX_iv_call_SPX_c_lightgbm_Threshold_0.999-0.999']
print(len(best_strategy), len(test_index))
best_strategy.index = test_index
print(best_strategy)

import quantstats as qs

# Generate the report
qs.extend_pandas()
report = qs.reports.full(best_strategy, benchmark='SPY')
print(report)

states = main_msr_probs_merged.loc[:,'p_3d_vix_SPX_iv_call_SPX_c']
states.index = data_index
print(states)

vol = data_with_variables_cleaned.loc[:,'p_3d']
print(vol)

import matplotlib.pyplot as plt

start_date = '2007-01-01'
end_date = '2008-01-01'

states = states.loc[start_date:end_date]
vol = vol.loc[start_date:end_date]


# scaling
mean_value = vol.mean()
amplitude_factor = 10000
centered_amplified_series = (vol - mean_value) * amplitude_factor + 50

max_high = 1.5
vol_centered = centered_amplified_series / centered_amplified_series.max() * max_high

# Plot
plt.figure(figsize=(10, 6))
plt.plot(states.index, states, label='True Probability of High Volatility State', alpha=0.95)
plt.plot(states.index, vol_centered, label='Parkinson Vol. scaled', linewidth=0.80, color='black')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Trained State-Probabilities')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

"""

---
Hyperparams optimization


---


"""

########################## Find optimal hyperparams ############################
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

# find profitables strategy
import optuna
def objective(trial):
    # search over all models
    model = trial.suggest_categorical('model_name', list(test_predictions.keys()))

    # params
    lower_bound = trial.suggest_float('lower_bound', -1.00, 1.50)
    upper_bound = trial.suggest_float('upper_bound', lower_bound, 1.50)
    if upper_bound <= lower_bound:
        raise optuna.TrialPruned()

    # fixed
    transaction_cost = 0.001
    max_position = 1.0
    risk_free_rate = 0.00

    # Generate strategy
    strategy_name = f'optimized_strategy_{int(lower_bound*100)}_{int(upper_bound*100)}'
    strategy_params = {
        strategy_name: {
            'signals': test_predictions[model], # chose model
            'returns': data_with_variables_cleaned['r'],
            'lower': lower_bound,
            'upper': upper_bound,
            'transaction_cost': transaction_cost,
            'max_position': max_position
        }
    }
    results = {}
    for strategy_name, params in strategy_params.items():
        strategy = classic_long_short(strategy_name, params, test_index, risk_free_rate)
        total_return, sharpe_ratio, sortino_ratio = calculate_strategy_metrics(strategy['returns'], risk_free_rate)
        results[strategy_name] = (strategy, total_return, sharpe_ratio, sortino_ratio)

    # target
    sharpe_ratio = results[strategy_name][2]
    return sharpe_ratio

# Execute the optimization and print best params
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # chose trials
print("Best parameters:", study.best_params)
print("Best Sharpe ratio:", study.best_value)

"""

---
ANN


---






"""

############################# Execution OLD  ########################################
# partition indexes
start_date = '2000-01-01'
split_date = '2005-01-01'
test_years = 1


### preprocess data
data = csv_reader('/content/data.csv')
data_cleaned = cleaner(data, backfill=False, verbose=False)
#print(data_cleaned)

# calculate end_date and extract indexes for partitioning
from datetime import datetime
split_date_dt = datetime.strptime(split_date, '%Y-%m-%d')
end_date_dt = datetime(split_date_dt.year + test_years-1, 12, 31)
end_date = end_date_dt.strftime('%Y-%m-%d')

data_index = data_cleaned[start_date:end_date].index
train_index = data_cleaned[start_date:(pd.to_datetime(split_date) - pd.Timedelta(days=1))].index
test_index = data_cleaned[split_date:end_date].index
#print(data_index, train_index, test_index)

# generate volatility meassures
window_size=3
data_with_variables = meassures(data_cleaned.loc[data_index,:], 'high_SPX', 'low_SPX', 'close_SPX', 'low_SPX', window_size)
data_with_variables_cleaned = cleaner(data_with_variables, backfill=True, verbose=False) # backfilling applied
#print(data_with_variables_cleaned)### Model Volatility with MSR

### generate regime probabilitites
target = 'predicted_1'  # pred_smooth, filtered_1, ... try with filtered
labels = [ 'gkyz_3d'] #'vol_3d', 'p_3d', 'gk_3d',
trends = ['c'] #'ct', 'n',
exogs = [['iv_call_SPX']] # add ['vix_SPX', 'iv_call_SPX', 'log_r']

# generate state probs
switching_variance = True
scale_endog = False
scale_exog = True
search_reps = 50
em_iter = 1000
regimes = 2
show_model = False
rescale_endog = True

state_probabilities = pd.DataFrame()
model_metrics = []
for trend in trends:
    msr_params = {
        'k_regimes': regimes,
        'switching_variance': switching_variance,
        'trend': trend,
        'em_iter': em_iter,
        'search_reps': search_reps}
    for label in labels:
        for exog in exogs:
            exog_name = '_'.join(exog) if exog else 'no_exog'
            name = f'{label}_{exog_name}_{trend}'

            print(f'Predicting Probabilities for {name}')
            probs, converged, aic, bic, log_likelihood = probabilities(data_with_variables_cleaned, label, 'msr', train_index, test_index, scale_endog, scale_exog, exog, msr_params, show_model, rescale_endog)

            if converged and target in probs:
                state_probabilities[name] = probs[target]
                model_metrics.append({
                    'model_name': name,
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood
                })
            elif not converged:
                print(f"Skipping {name} due to lack of convergence.")
            else:
                print(f"Warning: {target} not found in probs")

# clean stat probs
state_probabilities_cleaned = cleaner(state_probabilities, backfill=False, verbose=False)
#print(state_probabilities_cleaned)

# filter top 10 msr models with best AIC
model_metrics_df = pd.DataFrame(model_metrics)
if not model_metrics_df.empty:
    best_model_aic = model_metrics_df.loc[model_metrics_df['aic'].idxmin()]
    best_model_bic = model_metrics_df.loc[model_metrics_df['bic'].idxmin()]
    best_model_log_likelihood = model_metrics_df.loc[model_metrics_df['log_likelihood'].idxmax()]
else:
    print("No models converged.")
sorted_model_metrics_df = model_metrics_df.sort_values(by='aic', ascending=True)
#print(sorted_model_metrics_df)

# extract top 10 models' probs for predicting
top_10_models = model_metrics_df.nsmallest(5, 'aic')['model_name']
filtered_state_probabilities = state_probabilities_cleaned[top_10_models]
#print(filtered_state_probabilities)


### Predict State Probabilities
prediction_targets =  list(filtered_state_probabilities.columns) #+['vol_3d', 'p_3d', 'rs_3d', 'gk_3d', 'yz_3d', 'gkyz_3d'] apply for predicted RV
all_predictions = pd.DataFrame() # to store predictions
all_metrics = pd.DataFrame()  # To store MSE, Efron's R², etc.
all_feature_importance = pd.DataFrame()  # To store feature importance

for target in prediction_targets:
    print(f'Searching for {target}...')
    path = f'features_{target}.csv'
    try:
        all_targets_predictions = csv_reader(path)
        print(f'File found for {target}')
    except FileNotFoundError:
        X = pd.merge(data_with_variables_cleaned, state_probabilities_cleaned, left_index=True, right_index=True, how='inner') #X = data_with_variables_cleaned for RV
        y = X.loc[:, target]

        # Initialize PredictionManager
        manager = PredictionManager()
        ta_params = {'high': 'high_SPX', 'low': 'low_SPX', 'close': 'close_SPX', 'open': 'open_SPX', 'volume': 'volume_SPX'}
        ts_params = {'underlying': target, 'max_timeshift': 3, 'min_timeshift': 3}

        # Prepare features
        X_transformed = manager.prepare_features(X, ta_params, ts_params=ts_params, lags=[1])
        X_transformed =  cleaner(X_transformed, backfill=False, verbose=False)
        input_dim = X_transformed.shape[1]

        # Add models - (lightgbm, catboost, knn, lasso, elasticnet)
        manager.add_model('linreg', use_scaler=True, use_pca=False)
        #manager.add_model('ridge', use_scaler=True, use_pca=False)
        #manager.add_model('rf', use_scaler=False, use_pca=False)
        #manager.add_model('adaboost', use_scaler=False, use_pca=False)
        #manager.add_model('ann', input_dim=input_dim, hidden_dims=[64, 32], epochs=100, use_scaler=True)

        # Train and predict
        manager.train(X_transformed, y, train_index)
        manager.predict(X_transformed, data_index)

        # Extract evaluation metrics (MSE and Efron’s R²) and store them
        print('Train & Test Evaluation')
        manager.evaluate(y, test_index)
        target_metrics = manager.scores.copy()
        target_metrics['Target'] = target
        all_metrics = pd.concat([all_metrics, target_metrics], ignore_index=True)

        # Plot residuals
        manager.plot_residuals(y, test_index)

        # Extract and store feature importance
        print(f'Feature Importance for {target}')
        manager.feature_importance(X_transformed)
        target_feature_importance = manager.feature_importance_df.copy()
        target_feature_importance['Target'] = target
        all_feature_importance = pd.concat([all_feature_importance, target_feature_importance], ignore_index=True)

        # Extract the predictions
        all_targets_predictions = manager.extract_predicted_signals()
        all_targets_predictions.columns = [f'{target}_{col}' for col in all_targets_predictions.columns]

    # Concatenate to all_predictions
    if all_predictions.empty:
        all_predictions = all_targets_predictions
    else:
        all_predictions = pd.concat([all_predictions, all_targets_predictions], axis=1)

# Save evaluation metrics to CSV or display
print("Model Performance Metrics (RMSE, Efron’s R²): \n", all_metrics)
all_metrics.to_csv("model_performance_metrics.csv", index=False)

# Save feature importance to CSV or display
print("Feature Importance: \n", all_feature_importance)
all_feature_importance.to_csv("feature_importance.csv", index=False)


#### Generate Strategies
# chose predicted vaues to generate strategies from
signals = {}
#signals.update({f'RV_{key}': data_with_variables_cleaned.loc[data_index, [key]] for key in ['vol_3d', 'p_3d', 'rs_3d', 'gk_3d', 'yz_3d', 'gkyz_3d', 'vix_SPX']})
#signals.update({f'Prob_{col}': filtered_state_probabilities.loc[data_index, [col]] for col in filtered_state_probabilities.columns})
signals.update({f'Pred_{col}': all_predictions.loc[data_index, [col]] for col in all_predictions.columns})

# Extract returns of assets
risky_returns = data_with_variables_cleaned.loc[data_index, ['r']]
risk_free_returns = pd.Series(0, index=data_index, name='risk_free_returns')
if (risky_returns['r'] > 0.5).any():
    raise ValueError("Error: There are values in 'risky_returns' that exceed 0.5")
if (risk_free_returns > 0.5).any():
    raise ValueError("Error: There are values in 'risk_free_returns' that exceed 0.5")

# set params
all_position_funcs = [MeanReversion, MeanMomentum, ThresholdReversion, ThresholdMomentum]
bound_sets = [
    {'upper': 0.05, 'lower': 0.05},
    {'upper': 0.025, 'lower': 0.025},
    {'upper': 0.02, 'lower': 0.02},
    {'upper': 0.015, 'lower': 0.015},
    {'upper': 0.01, 'lower': 0.01},
    {'upper': 0.05, 'lower': 0.005},
]
rolling_windows = [5, 10, 15, 21, 63, 126, 252]
params = {
    'transaction_costs': 0.001,
    'lag': 1,
    'max_position': 1
}

# Initialize the manager
manager = StrategyManager(risk_free_returns=risk_free_returns.loc[data_index])

# Add Strategies
for signal_name, signal_data in signals.items():
    position_funcs = [MeanReversion, MeanMomentum] if 'RV' in signal_name else all_position_funcs

    for func in position_funcs:
        if func in [MeanReversion, MeanMomentum]:
            for window in rolling_windows:

                strategy_name = f"{signal_name}_{func.__name__}_{window}d"
                print(f'Adding Strategy: {strategy_name}')

                strategy = Strategy(
                    name=strategy_name,
                    signals=signal_data.loc[data_index],
                    risky_returns=risky_returns.loc[data_index],
                    risk_free_returns=risk_free_returns.loc[data_index],
                    transaction_costs=params['transaction_costs'],
                    max_position=params['max_position'])
                manager.add_strategy(strategy, position_func=func, lag=params['lag'], window=window)

        else:
            for bounds in bound_sets:
                upper = bounds['upper']
                lower = bounds['lower']

                strategy_name = f"{signal_name}_{func.__name__}_{lower}-{upper}"
                print(f'Adding Strategy: {strategy_name}')

                strategy = Strategy(
                    name=strategy_name,
                    signals=signal_data.loc[data_index],
                    risky_returns=risky_returns.loc[data_index],
                    risk_free_returns=risk_free_returns.loc[data_index],
                    transaction_costs=params['transaction_costs'],
                    max_position=params['max_position'])
                manager.add_strategy(strategy, position_func=func, lag=params['lag'], upper=upper, lower=lower)

# Add benchmark: Buy and Hold
buy_and_hold_strategy = Strategy(
    name='Buy and Hold',
    signals=pd.Series(1, index=data_index, name='Buy and Hold Signal'),
    risky_returns=risky_returns.loc[data_index],
    risk_free_returns=risk_free_returns.loc[data_index],
    transaction_costs=params['transaction_costs'],
    max_position=params['max_position']
)
manager.add_benchmark(buy_and_hold_strategy, position_func=long_only)

# Add benchmark: GARCH
from arch import arch_model

risky_returns_series = risky_returns['r']
garch_strategy = GARCHStrategy(returns=risky_returns_series)
garch_strategy.fit()
garch_forecasted_vol = garch_strategy.forecast_volatility(horizon=1)
garch_signal = pd.Series(garch_forecasted_vol.flatten(), index=data_index[-len(garch_forecasted_vol):], name='GARCH Signal')

for window in rolling_windows:
    # For mean_reversion
    garch_strategy_reversion = Strategy(
        name=f'GARCH_Volatility_MeanReversion_{window}d',
        signals=garch_signal,
        risky_returns=risky_returns.loc[data_index],
        risk_free_returns=risk_free_returns.loc[data_index],
        transaction_costs=params['transaction_costs'],
        max_position=params['max_position']
    )
    manager.add_benchmark(garch_strategy_reversion, position_func=MeanReversion, lag=params['lag'], window=window)

    # For mean_momentum
    garch_strategy_momentum = Strategy(
        name=f'GARCH_Volatility_MeanMomentum_{window}d',
        signals=garch_signal,
        risky_returns=risky_returns.loc[data_index],
        risk_free_returns=risk_free_returns.loc[data_index],
        transaction_costs=params['transaction_costs'],
        max_position=params['max_position']
    )
    manager.add_benchmark(garch_strategy_momentum, position_func=MeanMomentum, lag=params['lag'], window=window)

# Add a benchmark: VIX scaled
vix_spx = data_with_variables_cleaned['vix_SPX']
vix_spx_scaled, vix_scaler = scale_any_data(vix_spx)

for window in rolling_windows:
    # For mean_reversion
    vix_reversion_strategy = Strategy(
        name=f'Scaled_VIX_MeanReversion_{window}d',
        signals=vix_spx_scaled,
        risky_returns=risky_returns.loc[data_index],
        risk_free_returns=risk_free_returns.loc[data_index],
        transaction_costs=params['transaction_costs'],
        max_position=params['max_position']
    )
    manager.add_benchmark(vix_reversion_strategy, position_func=MeanReversion, lag=params['lag'], window=window)

    # For mean_momentum
    vix_momentum_strategy = Strategy(
        name=f'Scaled_VIX_MeanMomentum_{window}d',
        signals=vix_spx_scaled,
        risky_returns=risky_returns.loc[data_index],
        risk_free_returns=risk_free_returns.loc[data_index],
        transaction_costs=params['transaction_costs'],
        max_position=params['max_position']
    )
    manager.add_benchmark(vix_momentum_strategy, position_func=MeanMomentum, lag=params['lag'], window=window)

################################# Testset ######################################

# Recalculate strategies and get performance
manager.calculate_strategies(start_date=split_date, end_date=end_date)
performance_df = manager.generate_performance_table(factors_list=['SMB', 'HML', 'RMW', 'CMA', 'UMD', 'VIX', 'BAB'], include_benchmarks=True)
print(performance_df)

# Plot performance results
manager.plot_results(plot_name=f"Performance from {split_date} to {end_date}", include_benchmarks=True)

# Get calculated strategies and benchmarks
strategies_and_benches = manager.get_calculated_strategies_and_benchmarks()

# Create DataFrame for simple returns
simple_returns_df = pd.DataFrame()

# Extract simple returns for each strategy and benchmark
for strategy in performance_df['Strategy']:
    if strategy in strategies_and_benches['strategies']:
        simple_returns_df[strategy] = strategies_and_benches['strategies'][strategy]['simple_returns']
    else:
        print(f"Strategy {strategy} not found")

for benchmark, benchmark_data in strategies_and_benches['benchmarks'].items():
    simple_returns_df[benchmark] = benchmark_data['simple_returns']
print(simple_returns_df)


split_year = split_date[:4]
end_year = end_date[:4]

# export
#simple_returns_df.to_csv(f'returns_{split_year}_{end_year}.csv', index=True)
#performance_df.to_csv(f'performance_{split_year}_{end_year}.csv', index=True)

"""

---
GARCH


---


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Example data: returns of a risky asset and risk-free rate
np.random.seed(42)
n_periods = 252 * 10  # 10 years of daily data
risky_returns = np.random.normal(0.0005, 0.02, n_periods)  # Simulate daily returns
risk_free_rate = 0.0001  # Daily risk-free rate (e.g., 0.01% per day)

# Convert to pandas DataFrame
data = pd.DataFrame({
    'risky_returns': risky_returns,
    'risk_free_rate': risk_free_rate
})

# Fit a GARCH(1,1) model to predict volatility
# Initialize a list to store predicted variances
predicted_variances = []

# Loop through each day and fit the GARCH model using past returns up to that day
for i in range(len(data)):
    if i < 21:  # Minimum data points to fit the model
        predicted_variances.append(np.nan)
    else:
        model = arch_model(data['risky_returns'][:i], vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=1)
        predicted_variances.append(forecast.variance.values[-1][0])

# Add predicted variances to the DataFrame
data['predicted_variance'] = predicted_variances

# Set a target volatility (e.g., 10% annualized, which is ~0.0062 daily)
target_volatility = 0.10 / np.sqrt(252)

# Compute the allocation to the risky asset based on predicted volatility
# More allocation to risky asset when volatility is low and vice versa
data['risky_allocation'] = target_volatility / np.sqrt(data['predicted_variance'])

# Apply caps to avoid shorting (no negative allocation) and leveraging (allocation > 100%)
data['risky_allocation'] = data['risky_allocation'].clip(lower=0.0, upper=1.0)

# Allocation to risk-free asset is the remainder
data['risk_free_allocation'] = 1.0 - data['risky_allocation']

# Compute the portfolio returns based on dynamic allocation
data['portfolio_returns'] = (data['risky_allocation'] * data['risky_returns']) + \
                            (data['risk_free_allocation'] * data['risk_free_rate'])

# Compute the cumulative returns of the dynamically allocated strategy
data['cumulative_returns'] = (1 + data['portfolio_returns']).cumprod()

# Compute the cumulative returns of the original risky asset for comparison
data['cumulative_risky'] = (1 + data['risky_returns']).cumprod()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['cumulative_returns'], label='Dynamic Allocation Strategy (No Short, No Leverage)')
plt.plot(data['cumulative_risky'], label='Original Risky Asset')
plt.title('Cumulative Returns: Dynamic Allocation vs. Risky Asset')
plt.legend()
plt.show()

######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Example data: returns of a risky asset and risk-free rate
np.random.seed(42)
n_periods = 252 * 10  # 10 years of daily data
risky_returns = np.random.normal(0.0005, 0.02, n_periods)  # Simulate daily returns
risk_free_rate = 0.0001  # Daily risk-free rate (e.g., 0.01% per day)

# Convert to pandas DataFrame
data = pd.DataFrame({
    'risky_returns': risky_returns,
    'risk_free_rate': risk_free_rate
})

# Fit a GARCH(1,1) model to predict volatility
# Initialize a list to store predicted variances
predicted_variances = []

# Loop through each day and fit the GARCH model using past returns up to that day
for i in range(len(data)):
    if i < 21:  # Minimum data points to fit the model
        predicted_variances.append(np.nan)
    else:
        model = arch_model(data['risky_returns'][:i], vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=1)
        predicted_variances.append(forecast.variance.values[-1][0])

# Add predicted variances to the DataFrame
data['predicted_variance'] = predicted_variances

# Set a target volatility (e.g., 10% annualized, which is ~0.0062 daily)
target_volatility = 0.10 / np.sqrt(252)

# Compute the allocation to the risky asset based on predicted volatility
# More allocation to risky asset when volatility is low and vice versa
data['risky_allocation'] = target_volatility / np.sqrt(data['predicted_variance'])
data['risky_allocation'] = data['risky_allocation'].clip(upper=1.0)  # Cap allocation at 100%

# Allocation to risk-free asset is the remainder
data['risk_free_allocation'] = 1.0 - data['risky_allocation']

# Compute the portfolio returns based on dynamic allocation
data['portfolio_returns'] = (data['risky_allocation'] * data['risky_returns']) + \
                            (data['risk_free_allocation'] * data['risk_free_rate'])

# Compute the cumulative returns of the dynamically allocated strategy
data['cumulative_returns'] = (1 + data['portfolio_returns']).cumprod()

# Compute the cumulative returns of the original risky asset for comparison
data['cumulative_risky'] = (1 + data['risky_returns']).cumprod()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['cumulative_returns'], label='Dynamic Allocation Strategy')
plt.plot(data['cumulative_risky'], label='Original Risky Asset')
plt.title('Cumulative Returns: Dynamic Allocation vs. Risky Asset')
plt.legend()
plt.show()

####################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Example data: returns of a risky asset and risk-free rate
np.random.seed(42)
n_periods = 252 * 10  # 10 years of daily data
risky_returns = np.random.normal(0.0005, 0.02, n_periods)  # Simulate daily returns
risk_free_rate = 0.0001  # Daily risk-free rate (e.g., 0.01% per day)

# Convert to pandas DataFrame
data = pd.DataFrame({
    'risky_returns': risky_returns,
    'risk_free_rate': risk_free_rate
})

# Fit a GARCH(1,1) model to predict volatility
# Initialize a list to store predicted variances
predicted_variances = []

# Loop through each day and fit the GARCH model using past returns up to that day
for i in range(len(data)):
    if i < 21:  # Minimum data points to fit the model
        predicted_variances.append(np.nan)
    else:
        model = arch_model(data['risky_returns'][:i], vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=1)
        predicted_variances.append(forecast.variance.values[-1][0])

# Add predicted variances to the DataFrame
data['predicted_variance'] = predicted_variances

# Set a target volatility (e.g., 10% annualized, which is ~0.0062 daily)
target_volatility = 0.10 / np.sqrt(252)

# Compute the scaling factor based on predicted volatility
data['scaling_factor'] = target_volatility / np.sqrt(data['predicted_variance'])
data['scaling_factor'] = data['scaling_factor'].clip(upper=1.0)  # Limit leverage to 1x

# Adjust returns based on the scaling factor
data['adjusted_returns'] = data['scaling_factor'] * data['risky_returns'] + \
                           (1 - data['scaling_factor']) * data['risk_free_rate']

# Compute the cumulative returns of the adjusted strategy
data['cumulative_returns'] = (1 + data['adjusted_returns']).cumprod()

# Compute the cumulative returns of the original risky asset for comparison
data['cumulative_risky'] = (1 + data['risky_returns']).cumprod()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['cumulative_returns'], label='Volatility-Managed Strategy with GARCH')
plt.plot(data['cumulative_risky'], label='Original Risky Asset')
plt.title('Cumulative Returns of Volatility-Managed Strategy vs. Risky Asset')
plt.legend()
plt.show()

####################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example data: returns of a risky asset and risk-free rate
np.random.seed(42)
n_periods = 252 * 10  # 10 years of daily data
risky_returns = np.random.normal(0.0005, 0.02, n_periods)  # Simulate daily returns
risk_free_rate = 0.0001  # Daily risk-free rate (e.g., 0.01% per day)

# Convert to pandas DataFrame
data = pd.DataFrame({
    'risky_returns': risky_returns,
    'risk_free_rate': risk_free_rate
})

# Calculate realized variance as the sum of squared daily returns within each month
# Assume 21 trading days per month for simplicity
data['month'] = (np.arange(len(data)) // 21)
monthly_realized_variance = data.groupby('month')['risky_returns'].apply(lambda x: np.sum((x - x.mean()) ** 2) / len(x))

# Propagate monthly variance back to daily data for proper scaling
data['realized_variance'] = data['month'].map(monthly_realized_variance)

# Set a target volatility (e.g., 10% annualized, which is ~0.0062 daily)
target_volatility = 0.10 / np.sqrt(252)

# Compute the scaling factor based on predicted (historical) volatility
data['scaling_factor'] = target_volatility / np.sqrt(data['realized_variance'])
data['scaling_factor'] = data['scaling_factor'].clip(upper=1.0)  # Limit leverage to 1x

# Adjust returns based on the scaling factor
data['adjusted_returns'] = data['scaling_factor'] * data['risky_returns'] + \
                           (1 - data['scaling_factor']) * data['risk_free_rate']

# Compute the cumulative returns of the adjusted strategy
data['cumulative_returns'] = (1 + data['adjusted_returns']).cumprod()

# Compute the cumulative returns of the original risky asset for comparison
data['cumulative_risky'] = (1 + data['risky_returns']).cumprod()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['cumulative_returns'], label='Volatility-Managed Strategy')
plt.plot(data['cumulative_risky'], label='Original Risky Asset')
plt.title('Cumulative Returns of Volatility-Managed Strategy vs. Risky Asset')
plt.legend()
plt.show()
