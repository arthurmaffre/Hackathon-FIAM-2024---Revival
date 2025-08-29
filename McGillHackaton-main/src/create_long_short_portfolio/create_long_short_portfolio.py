import pandas as pd
from src.rank_signals.signal_ranker import RankerFactory

class CreateLongShortPortfolio(object):
    """
    This class creates long and short portfolios from signals, based on a fixed threshold on ranked signals.
    It returns two DataFrames indicating the long and short positions with True/False.
    """

    def __init__(self, signals: pd.DataFrame) -> None:
        """
        Initializes the CreateLongShortPortfolio with signal data.

        Parameters:
        - signals: A DataFrame containing the signals for all assets.
        """
        self.signals = self._validate_and_clean_signals(signals)

    @staticmethod
    def _validate_and_clean_signals(signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and cleans the input signals DataFrame by removing rows where all values are NaN.

        Parameters:
        - signals: A DataFrame containing the signals for all assets.

        Returns:
        - A cleaned DataFrame with rows containing all NaN values removed.
        """
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("The signals must be provided as a pandas DataFrame.")
        return signals.dropna(axis=0, how='all')

    def create_long_short_portfolio(self, ranking_strategy: str = "simple",
                                    ascending: bool = True,
                                    method: str = 'first',
                                    fix_threshold: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates and returns long and short portfolios based on the fixed threshold using the specified ranking strategy.

        Parameters:
        - ranking_strategy: The strategy used to rank signals.
        - ascending: Whether to sort rankings in ascending order.
        - method: The method to use for breaking ties in rankings.
        - fix_threshold: The fixed threshold for selecting long/short positions.

        Returns:
        - A tuple containing two DataFrames: long_signals and short_signals with boolean values.
        """
        ranked_signals = self._rank_signals(ranking_strategy, ascending, method)
        long_signals, short_signals = self._apply_fix_method(ranked_signals, fix_threshold)
        return long_signals, short_signals

    def _rank_signals(self, ranking_strategy: str, ascending: bool, method: str) -> pd.DataFrame:
        """
        Ranks the signals using the specified ranking strategy.

        Parameters:
        - ranking_strategy: The strategy used to rank signals.
        - ascending: Whether to sort rankings in ascending order.
        - method: The method to use for breaking ties in rankings.

        Returns:
        - A DataFrame of ranked signals.
        """
        ranker_factory = RankerFactory()
        ranker = ranker_factory.select_ranker(ranking_strategy)
        return ranker.rank_signals(self.signals, ascending=ascending, method=method)

    @staticmethod
    def _apply_fix_method(ranked_signals: pd.DataFrame, fix_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the fixed threshold method to select long and short signals.

        Parameters:
        - ranked_signals: A DataFrame of ranked signals.
        - fix_threshold: The fixed threshold for selecting long/short positions.

        Returns:
        - A tuple containing two DataFrames: long_signals and short_signals with boolean values.
        """
        long_signals = ranked_signals <= fix_threshold
        short_signals = ranked_signals.apply(lambda row: row >= row.nlargest(fix_threshold).min(), axis=1)
        return pd.DataFrame(long_signals), pd.DataFrame(short_signals)

# Example usage
if __name__ == '__main__':
    y_pred = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/evaluate_model_performance/y_pred_tabular_hist_gradient_boosting_regressor.csv',
        index_col=0
    )

    long_short_portfolio_creator = CreateLongShortPortfolio(signals=y_pred)

    long_signals, short_signals = long_short_portfolio_creator.create_long_short_portfolio(
        ranking_strategy="simple",
        ascending=True,
        method='first',
        fix_threshold=25
    )

    print(long_signals.head())
    print(short_signals.head())


