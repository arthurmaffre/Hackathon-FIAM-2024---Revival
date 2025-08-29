import pandas as pd

pd.set_option('display.max_columns', None)

class OptionFeaturesCalculator:
    """
    A class to compute advanced features for options, such as implied volatility, skewness, and bid-ask spread.

    Methods:
    --------
    calculate_monthly_avg_impl_volatility():
        Calculates the monthly average implied volatility for options closest to 30 days to expiration.
    calculate_monthly_avg_skewness():
        Calculates the monthly average skewness between OTM puts and ATM options for options closest to 30 days to expiration.
    """

    def __init__(self, options_data: pd.DataFrame):
        self.data = options_data.copy()
        self.data['year_month'] = pd.to_datetime(self.data['date']).dt.to_period('M')

    def filter_options_by_ttm(self, dte_target=30, dte_tolerance=5):
        """
        Filters options based on days to expiration (TTM) to include only those closest to 30 days, with a tolerance.
        """
        ttm_min = dte_target - dte_tolerance
        ttm_max = dte_target + dte_tolerance
        return self.data[(self.data['TTM'] >= ttm_min) & (self.data['TTM'] <= ttm_max)]

    def calculate_monthly_avg_impl_volatility(self, dte_target=30, dte_tolerance=5):
        """Calculate the monthly average implied volatility for options closest to 30 days to expiration."""
        filtered_data = self.filter_options_by_ttm(dte_target, dte_tolerance)

        # Group by stock ticker and month to calculate monthly average implied volatility
        monthly_avg_iv = filtered_data.groupby(['stock_ticker', 'year_month'])['impl_volatility'].mean().reset_index(
            name='monthly_avg_impl_volatility')
        return monthly_avg_iv

    def calculate_monthly_avg_skewness(self, delta_otm_put=-0.2, delta_otm_tolerance=0.05, delta_atm=0.5,
                                       delta_atm_tolerance=0.05, dte_target=30, dte_tolerance=5):
        """
        Calculate the monthly average skewness between OTM puts and ATM options for options closest to 30 days to expiration.
        """
        # Filter for options closest to 30 days to expiration
        filtered_data = self.filter_options_by_ttm(dte_target, dte_tolerance)

        # Filter OTM puts with delta close to -0.2
        delta_otm_min = delta_otm_put - delta_otm_tolerance
        delta_otm_max = delta_otm_put + delta_otm_tolerance
        otm_puts = filtered_data[
            (filtered_data['cp_flag'] == 'P') &
            (filtered_data['delta'] >= delta_otm_min) &
            (filtered_data['delta'] <= delta_otm_max)
            ].copy()

        # Filter ATM options with abs(delta) close to 0.5
        delta_atm_min = delta_atm - delta_atm_tolerance
        delta_atm_max = delta_atm + delta_atm_tolerance
        atm_options = filtered_data[
            (filtered_data['abs_delta'] >= delta_atm_min) &
            (filtered_data['abs_delta'] <= delta_atm_max)
            ].copy()

        # Get avg implied volatility of ATM options
        avg_atm_iv = atm_options.groupby(['stock_ticker', 'date'])['impl_volatility'].mean().reset_index()
        avg_atm_iv = avg_atm_iv.rename(columns={'impl_volatility': 'avg_atm_iv'})

        # Get avg implied volatility of OTM puts
        otm_put_iv = otm_puts.groupby(['stock_ticker', 'date'])['impl_volatility'].mean().reset_index()
        otm_put_iv = otm_put_iv.rename(columns={'impl_volatility': 'otm_put_iv'})

        # Merge the data to compute skewness
        skewness_df = pd.merge(otm_put_iv, avg_atm_iv, on=['stock_ticker', 'date'], how='inner')

        # Compute skewness
        skewness_df['skewness'] = skewness_df['otm_put_iv'] - skewness_df['avg_atm_iv']

        # Create a 'year_month' column for aggregation
        skewness_df['year_month'] = skewness_df['date'].dt.to_period('M')

        # Calculate the monthly average skewness
        monthly_skewness = skewness_df.groupby(['stock_ticker', 'year_month'])['skewness'].mean().reset_index(
            name='monthly_avg_skewness')
        return monthly_skewness

    def calculate_monthly_avg_opt_baspread(self):
        """Calculate the monthly average bid-ask spread percentage."""
        # Avoid division by zero by filtering mid_price values
        self.data = self.data[self.data['mid_price'] > 0]

        self.data['opt_baspread'] = (self.data['best_offer'] - self.data['best_bid']) / self.data['mid_price']
        monthly_avg_baspread = self.data.groupby(['stock_ticker', 'year_month'])['opt_baspread'].mean().reset_index(
            name='monthly_avg_opt_baspread')
        return monthly_avg_baspread


if __name__ == "__main__":
    from src.option_features.option_preprocessor import OptionDataPreprocessor

    options_before_2010 = pd.read_parquet('../../data/raw_data/options_data_before_2010.parquet')
    options_after_2010 = pd.read_parquet('../../data/raw_data/options_data_after_2010.parquet')
    hackathon_df_preprocessed = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/preprocess_data/hackathon_df_preprocessed.csv',
        index_col=0, parse_dates=True)

    # Preprocess data
    preprocessor = OptionDataPreprocessor(options_before_2010, options_after_2010, hackathon_df_preprocessed)
    preprocessor.create_cusip_to_ticker_mapping()
    preprocessor.convert_date_column_to_datetime()
    preprocessor.map_cusip_to_ticker()
    preprocessor.calculate_mid_price()
    preprocessor.calculate_ttm()
    preprocessor.calculate_abs_delta()

    print(preprocessor.options_before_2010.head())
    print(preprocessor.options_after_2010.head())

    # Concatenate pre and post 2010 data after preprocessing
    options_data = pd.merge(preprocessor.options_before_2010, preprocessor.options_after_2010, how='outer')

    print(options_data.head())
    print(options_data.tail())

    # Calcul des indicateurs
    feature_calculator = OptionFeaturesCalculator(options_data)
    monthly_iv = feature_calculator.calculate_monthly_avg_impl_volatility()
    monthly_skewness = feature_calculator.calculate_monthly_avg_skewness()
    monthly_baspread = feature_calculator.calculate_monthly_avg_opt_baspread()

    # Fusionner les indicateurs en utilisant pd.merge
    all_indicators = monthly_iv.merge(monthly_skewness, on=['stock_ticker', 'year_month'], how='inner')
    all_indicators = all_indicators.merge(monthly_baspread, on=['stock_ticker', 'year_month'], how='inner')

    # Affichage des résultats pour vérification
    print(all_indicators.head())
    print(all_indicators.tail())

    # Enregistrer les résultats dans un fichier CSV
    all_indicators.to_csv('../../data/intermediate_data/options_features/options_indicators.csv', index=False)





