import pandas as pd
from tqdm import tqdm
from typing import Optional

pd.set_option('display.max_columns', None)


class DriftedWeightsCalculator:
    """
    A class to calculate drifted weights for a portfolio based on daily prices and long/short weights.

    Attributes:
        daily_prices (pd.DataFrame): DataFrame containing daily prices with dates as the index.
        long_short_weights (pd.DataFrame): DataFrame containing long/short weights with dates as the index.
        drifted_weights (Optional[pd.DataFrame]): DataFrame to store the calculated drifted weights.
    """

    def __init__(self, daily_prices: pd.DataFrame, long_short_weights: pd.DataFrame):
        """
        Initialize the DriftedWeightsCalculator with the provided data.

        Args:
            daily_prices (pd.DataFrame): DataFrame containing daily prices with dates as the index.
            long_short_weights (pd.DataFrame): DataFrame containing long/short weights with dates as the index.
        """
        self.daily_prices = self._ensure_datetime_index(daily_prices)
        self.long_short_weights = self._ensure_datetime_index(long_short_weights)
        self.drifted_weights: Optional[pd.DataFrame] = None  # To store drifted weights after calculation

    @staticmethod
    def _ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame index is in datetime format.

        If the index is a PeriodIndex, it is converted to a Timestamp.

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

    def calculate_drifted_weights(self) -> pd.DataFrame:
        """
        Calculate the drifted weights for the portfolio.

        Returns:
            pd.DataFrame: DataFrame containing the drifted weights.
        """
        # Shift the dates to start from the first day of the following month
        self.long_short_weights.index += pd.offsets.MonthBegin(1)

        # Calculate total allocations for long and short portfolios at each rebalancing date
        allocations = self.long_short_weights.apply(self._calculate_allocations, axis=1)
        long_allocations = allocations['long_alloc']
        short_allocations = allocations['short_alloc']

        # Separate long and short weights
        long_weights = self.long_short_weights.clip(lower=0)
        short_weights = self.long_short_weights.clip(upper=0).abs()  # Use absolute values for shorts

        # Calculate drifted weights for long and short portfolios separately
        drifted_long_weights = self._calculate_portfolio_drifted_weights(
            long_weights, long_allocations, portfolio_type='long')
        drifted_short_weights = self._calculate_portfolio_drifted_weights(
            short_weights, short_allocations, portfolio_type='short')

        # Assign negative signs to drifted short weights
        drifted_short_weights = -drifted_short_weights

        # Combine drifted weights
        drifted_weights = drifted_long_weights.add(drifted_short_weights, fill_value=0)

        # Store drifted weights for later use
        self.drifted_weights = drifted_weights

        return drifted_weights

    @staticmethod
    def _calculate_allocations(weights_row: pd.Series) -> pd.Series:
        """
        Calculate total allocations for long and short portfolios from a row of weights.

        Args:
            weights_row (pd.Series): Series containing weights for a given date.

        Returns:
            pd.Series: Series with 'long_alloc' and 'short_alloc' as total allocations.
        """
        long_alloc = weights_row[weights_row > 0].sum()
        short_alloc = -weights_row[weights_row < 0].sum()  # Negative sum to get positive allocation
        return pd.Series({'long_alloc': long_alloc, 'short_alloc': short_alloc})

    def _calculate_portfolio_drifted_weights(
        self,
        portfolio_weights: pd.DataFrame,
        allocations: pd.Series,
        portfolio_type: str = 'long'
    ) -> pd.DataFrame:
        """
        Calculate drifted weights for a given portfolio (long or short).

        Args:
            portfolio_weights (pd.DataFrame): DataFrame containing portfolio weights with dates as the index.
            allocations (pd.Series): Series containing total allocation for the portfolio at each rebalancing date.
            portfolio_type (str): 'long' or 'short'.

        Returns:
            pd.DataFrame: DataFrame containing drifted weights for the portfolio.
        """
        # Ensure dates are sorted
        portfolio_weights.sort_index(inplace=True)
        allocations.sort_index(inplace=True)

        # Calculate daily returns
        daily_returns = self.daily_prices.pct_change()

        # Align portfolio weights and daily returns columns
        portfolio_weights_aligned, daily_returns_aligned = portfolio_weights.align(
            daily_returns, axis=1, join='inner'
        )
        portfolio_weights_aligned.fillna(0, inplace=True)

        # Ensure dates are sorted
        daily_returns_aligned.sort_index(inplace=True)
        portfolio_weights_aligned.sort_index(inplace=True)

        # Align start dates of DataFrames
        aligned_start_date = max(portfolio_weights_aligned.index.min(), daily_returns_aligned.index.min())
        portfolio_weights_aligned = portfolio_weights_aligned.loc[aligned_start_date:]
        daily_returns_aligned = daily_returns_aligned.loc[aligned_start_date:]
        allocations = allocations.loc[aligned_start_date:]

        # Create an index that is the union of portfolio weights and daily returns dates
        all_dates = daily_returns_aligned.index.union(portfolio_weights_aligned.index).sort_values()

        # Reindex DataFrames to include all dates
        daily_returns_aligned = daily_returns_aligned.reindex(all_dates).fillna(0)
        portfolio_weights_aligned = portfolio_weights_aligned.reindex(all_dates).fillna(0)
        allocations = allocations.reindex(all_dates).ffill()

        # Initialize DataFrames for drifted weights and position values
        drifted_weights = pd.DataFrame(index=all_dates, columns=portfolio_weights_aligned.columns)
        position_values = pd.DataFrame(index=all_dates, columns=portfolio_weights_aligned.columns)

        # Start date
        start_date = portfolio_weights_aligned.index[0]
        current_allocation = allocations.loc[start_date]

        # Initialize position values
        weights = portfolio_weights_aligned.loc[start_date]
        position_values.loc[start_date] = weights * current_allocation

        # Iterate over dates
        for date in tqdm(all_dates, desc=f"Calculating drifted weights for the {portfolio_type} portfolio"):
            if date in portfolio_weights.index and date != start_date:
                # Rebalancing date
                weights = portfolio_weights_aligned.loc[date]
                current_allocation = allocations.loc[date]
                position_values.loc[date] = weights * current_allocation
                # Update start_date
                start_date = date
            else:
                if date != start_date:
                    # Not a rebalancing date
                    prev_date = position_values.index[position_values.index.get_loc(date) - 1]
                    prev_position_values = position_values.loc[prev_date]
                    returns = daily_returns_aligned.loc[date]

                    # Update position values
                    adjusted_returns = returns.fillna(0)
                    if portfolio_type == 'long':
                        # For long positions
                        position_values.loc[date] = prev_position_values * (1 + adjusted_returns)
                    else:
                        # For short positions
                        position_values.loc[date] = prev_position_values * (1 - adjusted_returns)

            # Calculate total portfolio value
            total_portfolio_value = position_values.loc[date].sum()

            # Handle case where total portfolio value is zero
            if total_portfolio_value == 0:
                print(f"Total portfolio value for {portfolio_type} is zero on {date}. Assigning zero weights.")
                drifted_weights.loc[date] = 0
            else:
                # Calculate drifted weights
                drifted_weights.loc[date] = (position_values.loc[date] / total_portfolio_value) * current_allocation

        return drifted_weights

    def calculate_turnover_monthly(self) -> pd.DataFrame:
        """
        Calculate the turnover of weights between rebalancing dates and the previous day's weights.

        Returns:
            pd.DataFrame: DataFrame with turnover values at rebalancing dates.
        """
        rebalancing_dates = self.long_short_weights.index  # Monthly rebalancing dates

        # Calculate trade values at rebalancing dates
        trades = pd.Series(index=rebalancing_dates, dtype='float64')

        # Calculate total portfolio value at rebalancing dates
        total_portfolio = pd.Series(index=rebalancing_dates, dtype='float64')

        for date in rebalancing_dates:
            # Get weights on rebalancing date
            current_weights = self.drifted_weights.loc[date]

            # Get previous day's weights (last trading day before rebalancing)
            idx = self.drifted_weights.index.get_loc(date)
            if idx > 0:
                prev_date = self.drifted_weights.index[idx - 1]
                previous_weights = self.drifted_weights.loc[prev_date]
            else:
                # If there is no previous date, previous weights are zeros
                previous_weights = pd.Series(0, index=current_weights.index)

            # Replace NaN with 0 in previous_weights
            previous_weights = previous_weights.fillna(0)

            # Calculate total value of trades as half the sum of absolute differences
            trades_value = (current_weights - previous_weights).abs().sum() / 2

            # Calculate total portfolio value on rebalancing dates
            total_portfolio_value = (current_weights * self.daily_prices.loc[date]).sum()

            # Store trade values and total portfolio values
            trades.loc[date] = trades_value
            total_portfolio.loc[date] = total_portfolio_value

         # Calculate turnover as the total value of trades divided by the total portfolio value
        
        total_trades = trades.sum()
        average_portfolio = total_portfolio.mean()
        turnover = total_trades / average_portfolio
        
        return turnover

    def calculate_turnover(self) -> float:
        """
        Calculate the total portfolio turnover as a float.
        Turnover is calculated as the total dollar value of trades divided by the average portfolio value over the period.

        Returns:
            float: The turnover of the portfolio over the period.
        """
        # Align indices of drifted weights and daily prices
        all_dates = self.drifted_weights.index.union(self.daily_prices.index).sort_values()
        drifted_weights_aligned = self.drifted_weights.reindex(all_dates).fillna(0)
        daily_prices_aligned = self.daily_prices.reindex(all_dates).fillna(method='ffill')

        rebalancing_dates = self.long_short_weights.index  # Monthly rebalancing dates

        total_trades = 0.0  # Total dollar value of trades over the period
        total_portfolio_value = 0.0  # Sum of portfolio values at rebalancing dates
        num_rebalancing_dates = len(rebalancing_dates)

        for date in rebalancing_dates:
            # Get weights on rebalancing date
            current_weights = drifted_weights_aligned.loc[date]

            # Get previous day's weights (last trading day before rebalancing)
            idx = drifted_weights_aligned.index.get_loc(date)
            if idx > 0:
                prev_date = drifted_weights_aligned.index[idx - 1]
                previous_weights = drifted_weights_aligned.loc[prev_date]
            else:
                # If there is no previous date, previous weights are zeros
                previous_weights = pd.Series(0, index=current_weights.index)

            # Replace NaN with 0 in weights
            current_weights = current_weights.fillna(0)
            previous_weights = previous_weights.fillna(0)

            # Calculate change in weights
            delta_weights = current_weights - previous_weights

            # Get prices on the date (ensure prices are aligned)
            prices_on_date = daily_prices_aligned.loc[date]
            # Align prices with weights
            prices_on_date = prices_on_date.reindex(current_weights.index)
            # Drop assets with missing prices
            valid_assets = prices_on_date.notna()
            prices_on_date = prices_on_date[valid_assets]
            current_weights = current_weights[valid_assets]
            delta_weights = delta_weights[valid_assets]

            # Calculate per-asset dollar positions
            current_positions = current_weights * prices_on_date

            # Total portfolio value is the sum of absolute positions (long and short)
            portfolio_value = current_positions.abs().sum()

            # Calculate per-asset dollar trades
            trades_value = (delta_weights.abs() * prices_on_date).sum()

            total_trades += trades_value
            total_portfolio_value += portfolio_value

        # Calculate average portfolio value over the period
        average_portfolio_value = total_portfolio_value / num_rebalancing_dates

        # Handle case where average portfolio value is zero
        if average_portfolio_value == 0:
            print("Average portfolio value is zero. Cannot compute turnover.")
            return 0.0

        # Calculate turnover as total trades divided by average portfolio value
        turnover = total_trades / average_portfolio_value

        return turnover


if __name__ == '__main__':
    # Load your data
    long_short_weights = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/weighting/long_short_weights_hist_gradient_boosting_regressor.csv',
        index_col=0, parse_dates=True
    )

    daily_prices = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/preprocess_data/stocks_prices_df_preprocessed.csv',
        index_col=0, parse_dates=True
    )

    # Initialize the calculator
    calculator = DriftedWeightsCalculator(
        daily_prices=daily_prices,
        long_short_weights=long_short_weights
    )

    # Calculate drifted weights
    drifted_weights = calculator.calculate_drifted_weights()

    # Calculate turnover
    turnover = calculator.calculate_turnover()

    print(turnover)
