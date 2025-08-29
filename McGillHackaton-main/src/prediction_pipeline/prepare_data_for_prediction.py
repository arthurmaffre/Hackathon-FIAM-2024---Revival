import pandas as pd
from typing import Optional


class PrepareDataForPrediction(object):
    def __init__(self):
        pass

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
        Ensures that a specified column or the index of the DataFrame is in a 'Period[M]' format.
        If `column_name` is provided, converts that column to Period[M] format.
        Otherwise, converts the index to a Period[M].

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.
        column_name : str, optional
            The name of the column to convert to Period[M] format. If None, the index is converted.
        format : str, optional
            The datetime format to use for conversion. Default is '%Y%m'.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the specified column or index converted to Period[M] format.
        """
        if column_name:
            if column_name not in data.columns:
                raise KeyError(f"The specified column '{column_name}' does not exist in the DataFrame.")

            # Check if the column is already a PeriodIndex with 'M' frequency
            if not (isinstance(data[column_name].dtype, pd.PeriodDtype) and data[column_name].dt.freq == 'M'):
                data[column_name] = pd.to_datetime(data[column_name], errors='coerce', format=format).dt.to_period('M')
        else:
            # Check if the index is already a PeriodIndex with 'M' frequency
            if not (isinstance(data.index, pd.PeriodIndex) and data.index.freq == 'M'):
                data.index = pd.to_datetime(data.index, errors='coerce', format=format).to_period('M')

        return data

    @staticmethod
    def convert_to_panel_with_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a DataFrame with 'stock_ticker' and 'date' (as PeriodIndex) to a hierarchical panel format with a MultiIndex,
        while ensuring the 'date' remains as a PeriodIndex with 'M' frequency.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with 'stock_ticker' column and a 'PeriodIndex' index.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with a MultiIndex (stock_ticker, date) where 'date' remains a PeriodIndex.
        """
        # Create the MultiIndex with 'stock_ticker' and 'date' (which remains a PeriodIndex)
        df_multiindex = df.set_index(['stock_ticker', df.index], append=False, drop=True)

        # Sort the index by both stock_ticker and date (chronologically)
        df_multiindex = df_multiindex.sort_index(level=[0, 1])

        return df_multiindex

    @staticmethod
    def shift_columns(df: pd.DataFrame, shift_dict: dict) -> pd.DataFrame:
        """
        Shift the values of specific columns in a MultiIndex DataFrame according to the given periods.

        Parameters:
        ----------
        df : pd.DataFrame
            A MultiIndex DataFrame where the index is (stock_ticker, date).
        shift_dict : dict
            A dictionary where the key is the column name to shift and the value is the period to shift.
            For example, {'column1': 1, 'column2': -2} will shift 'column1' down by 1 period and 'column2' up by 2 periods.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the specified columns shifted by the given periods.
        """

        # We will create a copy of the DataFrame to avoid modifying the original one
        df_shifted = df.copy()

        # Loop through the dictionary to shift each column
        for col, period in shift_dict.items():
            if col in df_shifted.columns:
                df_shifted[col] = df_shifted.groupby(level=0)[col].shift(period)
            else:
                raise KeyError(f"Column '{col}' not found in the DataFrame.")
        # drop rows where stock_exret is NaN
        df_shifted = df_shifted.dropna(subset=['stock_exret'])
        return df_shifted

    @staticmethod
    def forward_fill_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill for each 'stock_ticker' independently in a DataFrame with a MultiIndex.
        Count the missing values before and after the forward fill.

        Parameters :
        ----------
        df : pd.DataFrame
            The DataFrame containing a MultiIndex with 'stock_ticker' and 'date'.

        Returns :
        -------
        pd.DataFrame :
            The DataFrame with missing values forward filled for each 'stock_ticker'.
        """
        # Count missing values before forward fill
        missing_before = df.isna().sum().sum()
        print(f"Total missing values before forward fill: {missing_before}")

        # Apply forward fill by grouping by 'stock_ticker'
        df_filled = df.groupby(level='stock_ticker', group_keys=False).apply(lambda group: group.ffill())

        # Count missing values after forward fill
        missing_after = df_filled.isna().sum().sum()
        print(f"Total missing values after forward fill: {missing_after}")

        return df_filled

    @staticmethod
    def prepare_x_and_y_data(df: pd.DataFrame, target_col: str, drop_cols: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
            Extract the target column (y) and the feature columns (X) from a MultiIndex DataFrame.

            Parameters:
            ----------
            df : pd.DataFrame
                A MultiIndex DataFrame where the index is (stock_ticker, date).
            target_col : str
                The name of the column to be used as the target (y) for the model.
            drop_cols : list, optional
                A list of column names to drop from the feature set (X).

            Returns:
            -------
            X : pd.DataFrame
                The feature DataFrame (X) with specified columns dropped.
            y : pd.Series
                The target Series (y) corresponding to the target column.
            """

        # Check if the target column exists in the DataFrame
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in the DataFrame.")

        # Extract the target column (y)
        y = df[target_col]

        # Create the feature set (X) by dropping the target column and any specified columns
        X = df.drop(columns=[target_col])

        if drop_cols:
            X = X.drop(columns=drop_cols)

        return pd.DataFrame(X).reset_index(), pd.DataFrame(y).reset_index()

    def prepare_data_for_prediction(self, df: pd.DataFrame, shift_dict: dict,
                                    target_col: str, drop_cols: list = None, do_forward_filling: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the data for prediction by converting it to a MultiIndex DataFrame, shifting columns, and extracting the target and feature data.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame with 'stock_ticker' and 'date' (as index).
        shift_dict : dict
            A dictionary where the key is the column name to shift and the value is the period to shift.
        target_col : str
            The name of the column to be used as the target (y) for the model.
        drop_cols : list, optional
            A list of column names to drop from the feature set (X).
        do_forward_filling : bool, optional
            Whether to forward-fill the data for each group (stock_ticker) between the first and last non-NaN values.

        Returns:
        -------
        X : pd.DataFrame
            The feature DataFrame (X) with specified columns dropped.
        y : pd.Series
            The target Series (y) corresponding to the target column.
        """

        # Ensure the index (date) is in 'Month End' datetime format
        df = self.ensure_datetime(data=df)

        # Convert the DataFrame to a MultiIndex DataFrame
        df_multiindex = self.convert_to_panel_with_multiindex(df=df)

        # Shift the specified columns by the given periods
        df_shifted = self.shift_columns(df=df_multiindex, shift_dict=shift_dict)

        if do_forward_filling:
            # Forward-fill the data for each group (stock_ticker) between the first and last non-NaN values
            df_shifted = self.forward_fill_by_ticker(df=df_shifted)

        # Prepare the feature set (X) and target (y) data
        X, y = self.prepare_x_and_y_data(df=df_shifted, target_col=target_col, drop_cols=drop_cols)

        return X, y



if __name__ == '__main__':

    hackathon_df_preprocessed = pd.read_csv(filepath_or_buffer=
                                            '../../data/intermediate_data/preprocess_data/hackathon_df_preprocessed.csv',
                                            index_col=0, parse_dates=True)

    shift_dict = {'stock_exret': 1, 'eps_meanest': 1, 'eps_medest': 1, 'eps_stdevest': 1}

    prediction_pipeline = PrepareDataForPrediction()
    X, y = prediction_pipeline.prepare_data_for_prediction(
        df=hackathon_df_preprocessed, shift_dict=shift_dict, target_col='stock_exret',
        drop_cols=['ret_eom', 'permno', 'shrcd', 'exchcd', 'year', 'month', 'size_port', 'cusip', 'comp_name']
    )

    print(X)
    print(y)
