import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.regime_detection.utils import raise_arr_to_pd_obj

pd.set_option('display.max_columns', None)


class BuildFeaturesForSJM:
    """
    A class to build and standardize features for the SJM model based on SPX and VIX data.
    """

    def __init__(self, spx_data: pd.DataFrame, vix_data: pd.DataFrame, start_date: str = "2000-01-01"):
        """
        Initialize the BuildFeaturesForSJM with SPX and VIX data.

        Parameters:
            spx_data (pd.DataFrame): DataFrame containing daily SPX data with a 'Close' column.
            vix_data (pd.DataFrame): DataFrame containing VIX data.
            start_date (str): The starting date from which data is selected (default "2000-01-01").
        """
        # Validate inputs
        if not isinstance(spx_data, pd.DataFrame):
            raise TypeError("spx_data must be a pandas DataFrame.")
        if 'Close' not in spx_data.columns:
            raise ValueError("spx_data must contain a 'Close' column.")
        if not isinstance(vix_data, pd.DataFrame):
            raise TypeError("vix_data must be a pandas DataFrame.")

        self._start_date = pd.to_datetime(start_date)

        # Ensure indices are datetime
        if not np.issubdtype(spx_data.index.dtype, np.datetime64):
            spx_data.index = pd.to_datetime(spx_data.index)
        if not np.issubdtype(vix_data.index.dtype, np.datetime64):
            vix_data.index = pd.to_datetime(vix_data.index)

        # Filter data based on start_date
        self._spx_data = spx_data.loc[spx_data.index >= self._start_date].copy()
        self._vix_data = vix_data.loc[vix_data.index >= self._start_date].copy()

        self._features_scaled = None

    def get_standardized_features(self, rolling_windows: list = [6, 14]) -> pd.DataFrame:
        """
        Compute and standardize all features.

        Parameters:
            rolling_windows (list): List of window sizes for rolling features (default [6, 14]).

        Returns:
            pd.DataFrame: DataFrame of standardized features including the non-standardized SP500 log returns.
        """
        # Compute daily log returns for SPX
        self._spx_data['log_returns'] = np.log(self._spx_data['Close']).diff()
        self._spx_data.dropna(subset=['log_returns'], inplace=True)

        # Align VIX data with SPX data
        self._vix_data = self._vix_data.reindex(self._spx_data.index).ffill().dropna()

        # Compute rolling features
        rolling_features = self._calculate_rolling_features(self._spx_data['log_returns'], rolling_windows)

        # Compute absolute change features
        absolute_change_features = self._calculate_absolute_change_features(self._spx_data['log_returns'])

        # Rename VIX column
        self._vix_data = self._vix_data.rename(columns={self._vix_data.columns[0]: 'VIX'})

        # Merge all features (without 'VIX' and 'log_returns' at this stage)
        all_features = pd.concat([
            rolling_features,
            absolute_change_features
        ], axis=1).dropna()

        # Standardize features (exclude 'VIX' and 'log_returns')
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(all_features)

        # Convert scaled array back to DataFrame using raise_arr_to_pd_obj
        scaled_features = raise_arr_to_pd_obj(scaled_array, all_features, index_key='index', columns_key='columns',
                                              return_as_ser=False)

        # Add 'VIX' and 'log_returns' back to the scaled features
        scaled_features['VIX'] = self._vix_data['VIX']
        scaled_features['log_returns'] = self._spx_data['log_returns']

        self._features_scaled = scaled_features
        # Ensure all columns are numeric
        self._features_scaled = self._features_scaled.apply(pd.to_numeric, errors='coerce')

        return self._features_scaled

    @staticmethod
    def _calculate_rolling_features(log_returns: pd.Series, windows: list) -> pd.DataFrame:
        """
        Calculate rolling mean and standard deviation features.

        Parameters:
            log_returns (pd.Series): Series of logarithmic returns.
            windows (list): List of window sizes for calculation.

        Returns:
            pd.DataFrame: DataFrame with rolling features.
        """
        features = pd.DataFrame(index=log_returns.index)

        for w in windows:
            half_window = w // 2

            # Centered mean and std
            features[f'centered_mean_{w}'] = log_returns.rolling(window=w, min_periods=w).mean()
            features[f'centered_std_{w}'] = log_returns.rolling(window=w, min_periods=w).std()

            # Left (first half) mean and std
            features[f'left_mean_{w}'] = log_returns.rolling(window=w, min_periods=w).apply(
                lambda x: x[:half_window].mean(), raw=False)
            features[f'left_std_{w}'] = log_returns.rolling(window=w, min_periods=w).apply(
                lambda x: x[:half_window].std(), raw=False)

            # Right (second half) mean and std
            features[f'right_mean_{w}'] = log_returns.rolling(window=w, min_periods=w).apply(
                lambda x: x[half_window:].mean(), raw=False)
            features[f'right_std_{w}'] = log_returns.rolling(window=w, min_periods=w).apply(
                lambda x: x[half_window:].std(), raw=False)

        features.dropna(inplace=True)

        return features

    @staticmethod
    def _calculate_absolute_change_features(log_returns: pd.Series) -> pd.DataFrame:
        """
        Calculate absolute change features.

        Parameters:
            log_returns (pd.Series): Series of logarithmic returns.

        Returns:
            pd.DataFrame: DataFrame with absolute changes and previous absolute changes.
        """
        absolute_change = log_returns.diff().abs()
        previous_absolute_change = absolute_change.shift(1)

        changes_df = pd.DataFrame({
            'absolute_change': absolute_change,
            'previous_absolute_change': previous_absolute_change
        })

        changes_df.dropna(inplace=True)
        return changes_df

    @property
    def features_scaled(self) -> pd.DataFrame:
        """
        Get the scaled features.

        Returns:
            pd.DataFrame: DataFrame of scaled features.
        """
        if self._features_scaled is None:
            raise ValueError("Features have not been computed yet. Call get_standardized_features() first.")
        return self._features_scaled


# Example usage
if __name__ == '__main__':
    spx_data = pd.read_csv(filepath_or_buffer='../../data/raw_data/SP500_daily.csv', index_col='Date', parse_dates=True)
    vix_data = pd.read_csv(filepath_or_buffer='../../data/raw_data/VIX.csv', index_col='Date', parse_dates=True)

    # Create an instance of BuildFeaturesForSJM with a custom start date
    feature_engineer = BuildFeaturesForSJM(spx_data, vix_data, start_date="2000-01-01")

    # Compute standardized features with custom rolling windows
    features_scaled = feature_engineer.get_standardized_features(rolling_windows=[6, 14])

    # Print the first few rows of the standardized features
    print(features_scaled.head())
