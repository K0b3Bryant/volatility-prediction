import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
import torch
import torch.nn as nn

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
