import pandas as pd
from typing import Optional


class StrategyReturnsCalculator:
    """
    A class to calculate realized returns of a strategy using drifted weights and daily prices.

    Attributes:
        drifted_weights (pd.DataFrame): DataFrame containing drifted weights with dates as index.
        daily_prices (pd.DataFrame): DataFrame containing daily prices with dates as index.
    """

    def __init__(self, drifted_weights: pd.DataFrame, daily_prices: pd.DataFrame):
        """
        Initialize the StrategyReturnsCalculator with drifted weights and daily prices.

        Args:
            drifted_weights (pd.DataFrame): DataFrame containing drifted weights with dates as index.
            daily_prices (pd.DataFrame): DataFrame containing daily prices with dates as index.
        """
        self.drifted_weights = self._ensure_datetime_index(drifted_weights).sort_index()
        self.daily_prices = self._ensure_datetime_index(daily_prices).sort_index()

    @staticmethod
    def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame index is in datetime format.

        If the index is a PeriodIndex, it is converted to Timestamp.

        Args:
            data (pd.DataFrame): DataFrame whose index needs to be checked.

        Returns:
            pd.DataFrame: DataFrame with the index in datetime or timestamp format.
        """
        # If the index is a PeriodIndex, convert it to Timestamp
        if isinstance(data.index, pd.PeriodIndex):
            data.index = data.index.to_timestamp()
        # If the index is not already datetime, convert it
        elif not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        return data

    def calculate_realized_returns(self) -> pd.DataFrame:
        """
        Calculate the daily realized returns of the strategy using drifted weights.

        Returns:
            pd.DataFrame: DataFrame with a single column of daily realized returns.
        """
        # Calculate daily returns
        daily_returns = self.daily_prices.pct_change()

        # Ensure all rebalancing dates are included in daily_returns
        # Reindex daily_returns to include all dates in drifted_weights, filling missing returns with zero
        daily_returns = daily_returns.reindex(self.drifted_weights.index).fillna(0)

        # Align drifted_weights and daily_returns on columns
        drifted_weights_aligned, daily_returns_aligned = self.drifted_weights.align(
            daily_returns, join='inner', axis=1
        )

        # Shift drifted_weights by 1 to represent weights at end of previous day
        drifted_weights_shifted = drifted_weights_aligned.shift(1)

        # Calculate portfolio returns
        portfolio_returns = (drifted_weights_shifted * daily_returns_aligned).sum(axis=1)

        # Create DataFrame with realized returns
        realized_returns = pd.DataFrame(portfolio_returns, columns=['Realized_Return'])

        return realized_returns
