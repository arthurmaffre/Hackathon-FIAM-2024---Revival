from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd


class SignalsRankerAbstract(ABC):
    """
    Abstract base class for different signal ranking strategies.
    """

    @abstractmethod
    def rank_signals(self, signals: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class SimpleSignalsRanker(SignalsRankerAbstract):
    """
    This class implements a simple ranking strategy for signals in a DataFrame.
    """
    def __init__(self):
        pass

    def rank_signals(self, signals: pd.DataFrame, ascending: bool = False,
                     method: Literal["average", "min", "max", "first", "dense"] = 'first') -> pd.DataFrame:
        return signals.rank(axis=1, ascending=ascending, method=method)


class RankerFactory(object):
    """
    Factory class to create instances of signal ranking strategies.
    """
    strategies = {
        "simple": SimpleSignalsRanker,
    }

    @classmethod
    def select_ranker(cls, ranking_strategy: str) -> SignalsRankerAbstract:
        """
        Selects and initializes the specified ranking strategy with provided arguments.

        Parameters:
        ----------
        strategy_type : str
            The type of the ranking strategy to create.
        **kwargs
            Additional keyword arguments to pass to the strategy initializer.

        Returns:
        -------
        SignalsRankerFactory
            An instance of the specified ranking strategy.

        Raises:
        ------
        ValueError
            If the strategy type is not recognized.
        """
        strategy_class = cls.strategies.get(ranking_strategy)
        if strategy_class:
            return strategy_class()
        else:
            valid_types = ", ".join(cls.strategies.keys())
            raise ValueError(f"Unknown ranker type: '{ranking_strategy}'. Valid types are: {valid_types}.")


if __name__ == '__main__':
    pass



