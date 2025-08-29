import pandas as pd
import matplotlib.pyplot as plt
from src.regime_detection.statistical_jump_model import JumpModel

import warnings
warnings.filterwarnings("ignore")


class RegimeBasedPortfolio:
    """
    A class to calculate regime probabilities using the Statistical Jump Model (SJM)
    and to compute defensive portfolio allocations based on those probabilities.
    """

    def __init__(self, data: pd.DataFrame,
                 n_regimes: int = 2,
                 n_iterations: int = 1000,
                 cont: bool = True,
                 random_state: int = 0,
                 defensive_bull: float = 0.2,
                 defensive_bear: float = 0.7,
                 jump_penalty: float = 1000):
        """
        Initializes the RegimeBasedPortfolio with the provided data and configuration parameters.

        Parameters:
            data (pd.DataFrame): A DataFrame containing the financial data, including log returns.
            n_regimes (int): Number of regimes to model (default 2).
            n_iterations (int): Number of iterations for the jump model (default 1000).
            cont (bool): Whether to use continuous data in the jump model (default True).
            random_state (int): Random state for reproducibility (default 0).
            defensive_bull (float): Defensive allocation percentage in a bull market (default 0.2).
            defensive_bear (float): Defensive allocation percentage in a bear market (default 0.7).
            jump_penalty (float): Penalty term for the jump model (default 1000).
        """
        self.data = data.sort_index()
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.cont = cont
        self.random_state = random_state
        self.defensive_bull = defensive_bull
        self.defensive_bear = defensive_bear
        self.jump_penalty = jump_penalty
        self.model = None  # Placeholder for the model
        self.regime_probabilities = pd.DataFrame()

    def get_defensive_allocation_for_rebalancing_date(self, rebalancing_date: pd.Period) -> float:
        """
        Calculates the defensive allocation percentage for the rebalancing date by training the SJM
        on data up to the end of the previous month and predicting the regime probabilities
        for the current month.

        Parameters:
            rebalancing_date (pd.Period): The rebalancing date as a Period (monthly frequency).

        Returns:
            float: The defensive allocation percentage for the rebalancing date.
        """
        # Convert rebalancing_date to Timestamp for consistency
        end_date = rebalancing_date.to_timestamp(how='end')
        # Calculate start_date as the first day of the current month
        start_date = end_date.replace(day=1)

        # Get training data up to the end of the previous month
        train_end_date = start_date - pd.Timedelta(days=1)
        train_data = self.data[self.data.index <= train_end_date]
        if train_data.empty:
            raise ValueError(f"No training data available before {start_date}")

        log_returns_train = train_data['log_returns']
        train_features = train_data.drop(columns=['log_returns'], errors='ignore')

        # Get test data for the current month
        test_data = self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
        if test_data.empty:
            raise ValueError(f"No data available between {start_date} and {end_date}")

        test_features = test_data.drop(columns=['log_returns'], errors='ignore')

        # Fit the model
        self.model = JumpModel(
            n_components=self.n_regimes,
            jump_penalty=self.jump_penalty,
            cont=self.cont,
            max_iter=self.n_iterations,
            random_state=self.random_state,
            verbose=False,
            mode_loss=True
        ).fit(train_features, log_returns_train)

        # Predict regime probabilities for the current month
        regime_probs = self.model.predict_proba_online(test_features)
        regime_probs.columns = ['Bull', 'Bear']

        # Get the probabilities for the end_date (rebalancing date)
        P_bull = regime_probs.at[end_date, 'Bull']
        P_bear = regime_probs.at[end_date, 'Bear']

        # Calculate defensive allocation
        defensive_allocation = (self.defensive_bull * P_bull) + (self.defensive_bear * P_bear)

        return defensive_allocation
