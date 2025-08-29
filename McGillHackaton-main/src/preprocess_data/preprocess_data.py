import pandas as pd
from typing import Optional

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class PreprocessData(object):
    """
    Class containing data preprocessing methods for hackathons and finance-related projects.
    """

    @staticmethod
    def ensure_datetime(data: pd.DataFrame, column_name: str = None, format: Optional[str] = '%Y%m') -> pd.DataFrame:
        """
        Ensure that a specified column in the DataFrame is in a 'datetime' format.
        Converts the specified column to a 'datetime' format using the provided format string.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.
        column_name : str
            The name of the column to convert to 'datetime' format.
        format : str, optional
            The datetime format to use for conversion. Default is '%Y%m'.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the specified column converted to 'datetime' format.
        """

        if column_name is None:
            data.index = pd.to_datetime(data.index, errors='coerce', format=format)
            return data

        else:
            if column_name not in data.columns:
                raise KeyError(f"The specified column '{column_name}' does not exist in the DataFrame.")

            data[column_name] = pd.to_datetime(data[column_name], errors='coerce', format=format)
            return data

    @staticmethod
    def ensure_monthly_datetime(
            data: pd.DataFrame,
            column_name: Optional[str] = None,
            format: Optional[str] = '%Y%m'
    ) -> pd.DataFrame:
        """
        Ensures that a specified column or the index of the DataFrame is in a 'Month End' datetime format.
        If `column_name` is provided, converts that column to Month End format.
        Otherwise, converts the index to a Month End DateTimeIndex.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.
        column_name : str, optional
            The name of the column to convert to 'Month End' datetime format. If None, the index is converted.
        format : str, optional
            The datetime format to use for conversion. Default is '%Y%m%d'.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the specified column or index converted to 'Month End' datetime format.
        """
        if column_name:
            if column_name not in data.columns:
                raise KeyError(f"The specified column '{column_name}' does not exist in the DataFrame.")
            data[column_name] = pd.to_datetime(data[column_name], errors='coerce', format=format).dt.to_period('M')
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, errors='coerce', format=format).to_period('M')
            else:
                data.index = data.index.to_period('M')

        return data

    @staticmethod
    def forward_fill_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fills the data for each column between the first and last non-NaN values.

        Parameters:
        ----------
        data (pd.DataFrame): The input DataFrame with potential NaN values.

        Returns:
        -------
        pd.DataFrame: DataFrame with forward-filled values.
        """
        data = data.copy()

        for column in data.columns:
            col_data = data[column]
            non_nan_indices = col_data.dropna().index
            if not non_nan_indices.empty:
                start, end = non_nan_indices[0], non_nan_indices[-1]
                data.loc[start:end, column] = col_data.loc[start:end].ffill()

        return data

    @staticmethod
    def convert_int_to_datetime(df: pd.DataFrame, column_name: str = 'ret_eom') -> pd.DataFrame:
        """
        Converts a column containing integers in the YYYYMMDD format to a datetime format.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to convert.
        column_name : str, optional
            The name of the column to convert (default is 'ret_eom').

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the specified column converted to datetime format.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")

        try:
            df[column_name] = pd.to_datetime(df[column_name].astype(str), format='%Y%m%d')
        except ValueError as e:
            raise ValueError(f"Error converting column '{column_name}' to datetime: {e}")

        return df

    @staticmethod
    def extract_first_ticker_name(df: pd.DataFrame, column_name: str = 'Tickers') -> pd.DataFrame:
        """
        Extracts the first word from a column and creates a new 'Ticker_name' column.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing the column to process.
        column_name : str
            The name of the column from which to extract the first word.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with an added 'ticker_name' column.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")

        df['ticker_name'] = df[column_name].str.split().str[0]

        return df

    @staticmethod
    def keep_us_stocks_only(hackathon_df: pd.DataFrame, mapping_stocks_country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out non-US stocks in the stock mapping DataFrame based on the 'Country' column.

        Parameters:
        ----------
        hackathon_df : pd.DataFrame
            The DataFrame containing stock data (with 'stock_ticker').
        mapping_stocks_country_df : pd.DataFrame
            The DataFrame containing the ticker-country mappings (with 'ticker_name' and 'Country').

        Returns:
        -------
        pd.DataFrame
            The filtered DataFrame with only US stocks.
        """
        initial_rows = mapping_stocks_country_df.shape[0]

        # Filter to keep only US stocks
        mapping_stocks_country_df = mapping_stocks_country_df[mapping_stocks_country_df['Country'] == 'US']
        final_rows = mapping_stocks_country_df.shape[0]
        print(f"keep_us_stocks_only - Filtered US stocks only: {initial_rows - final_rows} rows dropped. Remaining: {final_rows} rows.")

        # Number of unique tickers before filtering by ticker_name
        unique_tickers_initial = mapping_stocks_country_df['ticker_name'].nunique()

        if 'stock_ticker' not in hackathon_df.columns:
            raise KeyError("Column 'stock_ticker' not found in the hackathon DataFrame.")
        if 'ticker_name' not in mapping_stocks_country_df.columns:
            raise KeyError("Column 'ticker_name' not found in the mapping DataFrame.")

        valid_tickers = hackathon_df['stock_ticker'].unique()
        mapping_stocks_country_df = mapping_stocks_country_df[
            mapping_stocks_country_df['ticker_name'].isin(valid_tickers)]

        unique_tickers_final = mapping_stocks_country_df['ticker_name'].nunique()
        print(f"keep_us_stocks_only - Tickers before filtering: {unique_tickers_initial}")
        print(f"keep_us_stocks_only - Tickers after filtering: {unique_tickers_final}")

        return mapping_stocks_country_df

    @staticmethod
    def map_stock_prices_to_tickers(hackathon_df: pd.DataFrame, stock_prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps tickers in the stock price DataFrame using the 'permno' column.

        Parameters:
        ----------
        hackathon_df : pd.DataFrame
            The DataFrame containing 'permno' and 'stock_ticker' columns.
        stock_prices_df : pd.DataFrame
            The DataFrame containing the 'permno' column to map.

        Returns:
        -------
        pd.DataFrame
            The stock price DataFrame with an added 'stock_ticker' column.
        """
        # Create a permno -> stock_ticker mapping dictionary
        mapping_df = hackathon_df.drop_duplicates(subset='permno', keep='last')
        permno_to_ticker_dict = dict(zip(mapping_df['permno'], mapping_df['stock_ticker']))

        # Add the 'stock_ticker' column to stock_prices_df
        stock_prices_df['stock_ticker'] = stock_prices_df['permno'].map(permno_to_ticker_dict)

        return stock_prices_df

    @staticmethod
    def pivot_stock_prices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a DataFrame with 'date', 'stock_ticker', and 'prc' columns into a pivoted format
        with stock_tickers as columns and prices as values. Dates will be unique in the index.

        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame containing 'date', 'stock_ticker', and 'prc' columns.

        Returns:
        -------
        pd.DataFrame
            Pivoted DataFrame with stock_tickers as columns, dates as index, and prices as values.
        """
        initial_shape = df.shape
        df_pivot = df.pivot_table(index='date', columns='stock_ticker', values='prc', aggfunc='first')

        final_shape = df_pivot.shape
        print(f"pivot_stock_prices - Initial shape: {initial_shape}, Final shape: {final_shape}.")

        return df_pivot

    @staticmethod
    def keep_common_stocks_name(pivoted_stocks_prices: pd.DataFrame, hackathon_df: pd.DataFrame) -> (
            pd.DataFrame, pd.DataFrame):
        """
        Keeps only the columns and rows for stocks that are common between the two DataFrames.

        Parameters:
        ----------
        pivoted_stocks_prices : pd.DataFrame
            The DataFrame containing the pivoted prices with 'stock_ticker' as columns.

        hackathon_df : pd.DataFrame
            The DataFrame containing tickers in a 'stock_ticker' column.

        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            The two modified DataFrames, containing only the common columns/rows between them.
        """
        tickers_in_pivot = set(pivoted_stocks_prices.columns)
        tickers_in_hackathon = set(hackathon_df['stock_ticker'].unique())

        common_tickers = tickers_in_pivot.intersection(tickers_in_hackathon)

        pivoted_stocks_prices_filtered = pivoted_stocks_prices[list(common_tickers)]
        hackathon_df_filtered = hackathon_df[hackathon_df['stock_ticker'].isin(common_tickers)]

        print(f"keep_common_stocks_name - Initial number of columns in pivoted_stocks_prices: {len(tickers_in_pivot)}")
        print(f"keep_common_stocks_name - Remaining columns after filtering: {len(common_tickers)}")
        print(f"keep_common_stocks_name - Dropped columns: {len(tickers_in_pivot) - len(common_tickers)}")

        print(f"keep_common_stocks_name - Initial number of rows in hackathon_df: {hackathon_df.shape[0]}")
        print(f"keep_common_stocks_name - Remaining rows after filtering: {hackathon_df_filtered.shape[0]}")
        print(f"keep_common_stocks_name - Dropped rows: {hackathon_df.shape[0] - hackathon_df_filtered.shape[0]}")

        return pivoted_stocks_prices_filtered, hackathon_df_filtered

    @staticmethod
    def keep_tickers_with_full_dates_range(df: pd.DataFrame,
                                           ticker_column: str = 'stock_ticker') -> pd.DataFrame:
        """
        Keeps only the tickers with data from the minimum to the maximum date in a panel DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing tickers and date columns.
        ticker_column : str, optional
            The name of the column containing tickers in df.

        Returns:
        -------
        pd.DataFrame
            The filtered DataFrame containing only the tickers with full date coverage.
        """

        # Ensure the index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # Find the minimum and maximum date in the DataFrame
        min_date = df.index.min()
        max_date = df.index.max()

        print(f"Minimum date in data: {min_date}")
        print(f"Maximum date in data: {max_date}")

        # Generate an index covering the full period from min_date to max_date, adding one month to include the last month
        full_period_range = pd.date_range(start=min_date, end=max_date + pd.offsets.MonthEnd(1), freq='ME', name='date')

        # Group by ticker and filter those with full date coverage
        tickers_with_full_range = df.groupby(ticker_column).filter(
            lambda x: x.index.nunique() == len(full_period_range)
        )[ticker_column].unique()

        # Filter the DataFrame to keep only tickers with full coverage
        df_filtered = df[df[ticker_column].isin(tickers_with_full_range)]

        print(f"Number of tickers before filtering: {df[ticker_column].nunique()}")
        print(f"Number of tickers after filtering: {df_filtered[ticker_column].nunique()}")

        return df_filtered

    @staticmethod
    def compute_monthly_change(data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the monthly percentage change for each column in the input matrix.

        Parameters:
        ----------
        data : pd.DataFrame
            Input matrix where the percentage change will be calculated.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the computed percentage change for each column.
        """
        # Calculate the monthly change for each column
        data_scaled = data.pct_change().dropna()

        return data_scaled

    def preprocess_data(self, hackathon_df: pd.DataFrame, stocks_prices_df: pd.DataFrame,
                        mapping_stocks_country_df: pd.DataFrame, macro_data: pd.DataFrame,
                        ticker_column: str = 'stock_ticker') -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

        hackathon_df = self.ensure_datetime(data=hackathon_df)
        hackathon_df = self.convert_int_to_datetime(hackathon_df)

        hackathon_df = self.keep_tickers_with_full_dates_range(hackathon_df, ticker_column=ticker_column)
        mapping_stocks_country_df = self.extract_first_ticker_name(mapping_stocks_country_df)
        stocks_prices_df = self.map_stock_prices_to_tickers(hackathon_df, stocks_prices_df)
        pivoted_stocks_prices_df = self.pivot_stock_prices(stocks_prices_df)
        pivoted_stocks_prices_df, hackathon_df = self.keep_common_stocks_name(pivoted_stocks_prices_df, hackathon_df)
        pivoted_stocks_prices_df = self.forward_fill_data(data=pivoted_stocks_prices_df)
        macro_data = self.ensure_datetime(data=macro_data)
        macro_data_scaled = self.compute_monthly_change(data=macro_data)

        return hackathon_df, pivoted_stocks_prices_df, mapping_stocks_country_df, macro_data_scaled


    # def preprocess_data_sktime(self, hackathon_df: pd.DataFrame, stocks_prices_df: pd.DataFrame,
    #                            mapping_stocks_country_df: pd.DataFrame, macro_data: pd.DataFrame,
    #                            ticker_column: str = 'stock_ticker') -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    #     """
    #     Data preprocessing pipeline.
    #     """
    #     hackathon_df = self.ensure_monthly_datetime(data=hackathon_df)
    #     hackathon_df = self.convert_int_to_datetime(hackathon_df)
    #     hackathon_df = self.keep_tickers_with_full_dates_range_sktime(hackathon_df, ticker_column=ticker_column)
    #     mapping_stocks_country_df = self.extract_first_ticker_name(mapping_stocks_country_df)
    #     stocks_prices_df = self.map_stock_prices_to_tickers(hackathon_df, stocks_prices_df)
    #     pivoted_stocks_prices_df = self.pivot_stock_prices(stocks_prices_df)
    #     pivoted_stocks_prices_df, hackathon_df = self.keep_common_stocks_name(pivoted_stocks_prices_df, hackathon_df)
    #     pivoted_stocks_prices_df = self.forward_fill_data(data=pivoted_stocks_prices_df)
    #     macro_data = self.ensure_monthly_datetime(data=macro_data)
    #     macro_data_scaled = self.compute_monthly_change(data=macro_data)
    #
    #     return hackathon_df, pivoted_stocks_prices_df, mapping_stocks_country_df, macro_data_scaled


if __name__ == '__main__':
    hackathon_df = pd.read_csv('../../data/raw_data/hackathon_sample_v2.csv', index_col=0, parse_dates=True)
    stocks_prices_df = pd.read_parquet('../../data/raw_data/stock_prices.parquet')
    mapping_stocks_country_df = pd.read_excel('../../data/raw_data/us_stocks_list.xlsx')[['Tickers', 'Name', 'Sector', 'Country']]
    macro_data = pd.read_csv('../../data/raw_data/macro_data_us.csv', index_col=0, parse_dates=True)

    print(hackathon_df.head())
    print(stocks_prices_df.head())
    print(mapping_stocks_country_df.head())

    preprocess_data = PreprocessData()

    hackathon_df, pivoted_stocks_prices_df, mapping_stocks_country_df, macro_data_scaled = preprocess_data.preprocess_data(
        hackathon_df,
        stocks_prices_df,
        mapping_stocks_country_df,
        macro_data=macro_data,
        ticker_column='stock_ticker',
    )

    print(hackathon_df.head())
    print(stocks_prices_df.head())
    print(mapping_stocks_country_df.head())
