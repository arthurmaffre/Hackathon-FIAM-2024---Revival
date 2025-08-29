import pandas as pd
from typing import Dict


class OptionDataPreprocessor:
    """
    A class to preprocess CUSIP values, map them to stock tickers, and calculate basic option-related fields.

    Methods:
    --------
    create_cusip_to_ticker_mapping() -> Dict[str, str]:
        Create a dictionary mapping CUSIP to stock ticker using hackathon data.
    map_cusip_to_ticker():
        Apply the CUSIP to ticker mapping to the option datasets.
    calculate_mid_price():
        Calculate the mid-price for both option datasets.
    calculate_ttm():
        Calculate the time to maturity (TTM) for both option datasets.
    calculate_abs_delta():
        Calculate the absolute value of delta for both option datasets.
    """

    def __init__(self, options_before_2010: pd.DataFrame, options_after_2010: pd.DataFrame,
                 hackathon_df_preprocessed: pd.DataFrame):
        self.options_before_2010 = options_before_2010
        self.options_after_2010 = options_after_2010
        self.hackathon_df_preprocessed = hackathon_df_preprocessed
        self.cusip_to_ticker_map = None

    def create_cusip_to_ticker_mapping(self) -> Dict[str, str]:
        """Create a mapping of CUSIP to stock ticker."""
        if 'cusip' not in self.hackathon_df_preprocessed.columns or 'stock_ticker' not in self.hackathon_df_preprocessed.columns:
            raise ValueError("hackathon_df_preprocessed must contain 'cusip' and 'stock_ticker' columns.")
        valid_data = self.hackathon_df_preprocessed.dropna(subset=['cusip', 'stock_ticker'])
        self.cusip_to_ticker_map = valid_data.set_index('cusip')['stock_ticker'].to_dict()
        return self.cusip_to_ticker_map

    def map_cusip_to_ticker(self):
        """Map CUSIP values to stock tickers in both option datasets."""
        if self.cusip_to_ticker_map is None:
            raise ValueError(
                "CUSIP to ticker mapping has not been created. Call 'create_cusip_to_ticker_mapping' first.")

        for df in [self.options_before_2010, self.options_after_2010]:
            if 'cusip' in df.columns:
                df['stock_ticker'] = df['cusip'].map(self.cusip_to_ticker_map)
            else:
                raise ValueError(f"{df} does not contain 'cusip' column.")

    def convert_date_column_to_datetime(self):
        """Convert date columns to datetime format."""
        for df in [self.options_before_2010, self.options_after_2010]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError(f"{df} does not contain 'date' column.")

    def calculate_mid_price(self):
        """Calculate mid-price and add a 'mid_price' column."""
        for df in [self.options_before_2010, self.options_after_2010]:
            df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2

    def calculate_ttm(self):
        """Calculate time to maturity (TTM) in days and add a 'TTM' column."""
        for df in [self.options_before_2010, self.options_after_2010]:
            df['TTM'] = (pd.to_datetime(df['exdate']) - pd.to_datetime(df['date'])).dt.days

    def calculate_abs_delta(self):
        """Calculate the absolute value of delta and add a 'abs_delta' column."""
        for df in [self.options_before_2010, self.options_after_2010]:
            df['abs_delta'] = df['delta'].abs()


# Example usage:
if __name__ == '__main__':
    # Load the data (replace these with actual data loading steps)
    options_before_2010 = pd.read_parquet('../../data/raw_data/options_data_before_2010.parquet')
    options_after_2010 = pd.read_parquet('../../data/raw_data/options_data_after_2010.parquet')
    hackathon_df_preprocessed = pd.read_csv(filepath_or_buffer='../../data/intermediate_data/preprocess_data/hackathon_df_preprocessed.csv',
                                            index_col=0, parse_dates=True)

    # Initialize the mapper
    mapper = OptionDataPreprocessor(options_before_2010, options_after_2010, hackathon_df_preprocessed)

    # Create the CUSIP to ticker mapping
    cusip_to_ticker_mapping = mapper.create_cusip_to_ticker_mapping()

    # Map CUSIP to stock tickers in the options data
    mapper.map_cusip_to_ticker()

    # Calculate mid_price, TTM, and abs_delta
    mapper.convert_date_column_to_datetime()
    mapper.calculate_mid_price()
    mapper.calculate_ttm()
    mapper.calculate_abs_delta()

    print("First 5 rows of options_before_2010 with new columns:")
    print(options_before_2010.head())
    print("\nFirst 5 rows of options_after_2010 with new columns:")
    print(options_after_2010.head())
