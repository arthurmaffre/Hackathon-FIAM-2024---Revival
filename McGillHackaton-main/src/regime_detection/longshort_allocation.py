import pandas as pd
import matplotlib.pyplot as plt
from src.regime_detection.statistical_jump_model import JumpModel
from tqdm import tqdm  # For progress tracking

import warnings
warnings.filterwarnings("ignore")

class RegimeBasedPortfolio:
    """
    A class to calculate regime probabilities using the Statistical Jump Model (SJM)
    and to compute short portfolio allocations based on those probabilities.
    """

    def __init__(self, data: pd.DataFrame, start_year: int = 2010, end_year: int = 2023,
                 n_regimes: int = 2, n_iterations: int = 1000, cont: bool = True,
                 random_state: int = 0, beta_bull: float = 0.0, beta_bear: float = 0.4, jump_penalty: float = 1000):
        """
        Initializes the RegimeBasedPortfolio with the provided data and configuration parameters.

        Parameters:
            data (pd.DataFrame): A DataFrame containing the financial data, including log returns.
            start_year (int): The starting year for the regime probability calculations (default 2010).
            end_year (int): The ending year for the regime probability calculations (default 2023).
            n_regimes (int): Number of regimes to model (default 2).
            n_iterations (int): Number of iterations for the jump model (default 1000).
            cont (bool): Whether to use continuous data in the jump model (default True).
            random_state (int): Random state for reproducibility (default 0).
            beta_bull (float): Short allocation percentage in a bull market (default 0.0).
            beta_bear (float): Short allocation percentage in a bear market (default 0.4).
        """
        self.data = data
        self.start_year = start_year
        self.end_year = end_year
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.cont = cont
        self.random_state = random_state
        self.beta_bull = beta_bull
        self.beta_bear = beta_bear
        self.jump_penalty = jump_penalty
        self.regime_probabilities = pd.DataFrame()

    def calculate_period_regime_probabilities(self) -> pd.DataFrame:
        """
        Calculates the regime probabilities for each month within the given period.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated 'Bull' and 'Bear' regime probabilities.
        """
        # Ensure we are working with a datetime index and generate months for the period
        self.data.index = pd.to_datetime(self.data.index)
        months_range = pd.date_range(start=f'{self.start_year}-01-01', end=f'{self.end_year}-12-31', freq='M')
        monthly_regime_probabilities = pd.DataFrame()

        print(f"Calculating regime probabilities from {self.start_year} to {self.end_year} on a monthly basis...")

        # Loop through each month and calculate regime probabilities
        for month in tqdm(months_range, desc="Processing months"):
            # Train data up to the current month
            train_data = self.data[self.data.index < month]
            log_returns_train = train_data['log_returns']
            train_features = train_data.drop(columns=['log_returns'])

            # Test data for the current month
            test_data = self.data[self.data.index.to_period('M') == month.to_period('M')]
            if test_data.empty:
                print(f"No data available for {month}. Skipping...")
                continue

            test_features = test_data.drop(columns=['log_returns'])

            # Fit the model and predict probabilities
            model = JumpModel(n_components=self.n_regimes, jump_penalty=self.jump_penalty, cont=self.cont, max_iter=self.n_iterations,
                              random_state=self.random_state, verbose=False, mode_loss=False).fit(
                train_features, log_returns_train)

            # Predict regime probabilities
            monthly_regimes = model.predict_proba_online(test_features).shift().dropna()

            # Append to overall results
            monthly_regime_probabilities = pd.concat([monthly_regime_probabilities, monthly_regimes], axis=0)

        # Set regime column names
        monthly_regime_probabilities.columns = ['Bull', 'Bear']
        # keep only last regime probabilities for each month
        self.regime_probabilities = monthly_regime_probabilities

        print(f"Regime probabilities calculated successfully for {len(self.regime_probabilities) // 12} months.")
        return monthly_regime_probabilities

    def calculate_short_allocation(self) -> pd.Series:
        """
        Calculate short portfolio allocation percentages based on bear and bull regime probabilities.

        Returns:
            pd.Series: A Series with 'Short_Allocation' based on bear and bull regime probabilities.
        """
        if self.regime_probabilities.empty:
            raise ValueError(
                "Regime probabilities have not been calculated yet. Call 'calculate_period_regime_probabilities' first.")

        P_bull = self.regime_probabilities['Bull']
        P_bear = self.regime_probabilities['Bear']

        # Calculate short allocation
        short_allocation = (self.beta_bull * P_bull) + (self.beta_bear * P_bear)
        short_allocation.index = pd.to_datetime(short_allocation.index)

        print(f"Short allocation calculated successfully for {len(short_allocation)} months.")
        return short_allocation

    def plot_regime_probabilities(self, spx_data: pd.DataFrame):
        """
        Plot the SPX data with regime probabilities (Bull and Bear) overlaid.

        Parameters:
            spx_data (pd.DataFrame): A DataFrame containing SPX price data with a 'Close' column.
        """
        if self.regime_probabilities.empty:
            raise ValueError(
                "Regime probabilities have not been calculated yet. Call 'calculate_period_regime_probabilities' first.")

        # Filter spx_data to match the index of regime_probabilities
        spx_data_filtered = spx_data.loc[self.regime_probabilities.index]

        # Plot SPX data
        plt.figure(figsize=(14, 7))
        plt.plot(spx_data_filtered.index, spx_data_filtered['Close'], label='SPX Close Price', color='blue')

        # Overlay regime probabilities
        bull_prob = self.regime_probabilities['Bull']
        bear_prob = self.regime_probabilities['Bear']

        plt.fill_between(bull_prob.index, spx_data_filtered['Close'].min(), spx_data_filtered['Close'].max(),
                         where=bull_prob > bear_prob, color='green', alpha=0.3, label='Bull Regime')
        plt.fill_between(bear_prob.index, spx_data_filtered['Close'].min(), spx_data_filtered['Close'].max(),
                         where=bear_prob > bull_prob, color='red', alpha=0.3, label='Bear Regime')

        plt.title('SPX Close Price with Regime Probabilities')
        plt.xlabel('Date')
        plt.ylabel('SPX Close Price')
        plt.legend()
        plt.show()




# Example usage of both classes
if __name__ == '__main__':
    from src.regime_detection.regime_model_features import BuildFeaturesForSJM
    spx_data = pd.read_csv(filepath_or_buffer='../../data/raw_data/SP500_daily.csv', index_col='Date', parse_dates=True)
    vix_data = pd.read_csv(filepath_or_buffer='../../data/raw_data/VIX.csv', index_col='Date', parse_dates=True)

    # Step 1: Build Features using BuildFeaturesForSJM
    feature_engineer = BuildFeaturesForSJM(spx_data, vix_data, start_date="2000-01-01")
    standardized_features = feature_engineer.get_standardized_features(rolling_windows=[6, 14])

    # Step 2: Create an instance of RegimeBasedPortfolio using the standardized features
    portfolio_model = RegimeBasedPortfolio(data=standardized_features, start_year=2010, end_year=2023,
                                           beta_bull=0.2, beta_bear=0.5)

    # Calculate regime probabilities
    regime_probs = portfolio_model.calculate_period_regime_probabilities()

    # Calculate short allocation
    short_allocation = portfolio_model.calculate_short_allocation()

    # Print short allocation
    print("Short Allocation:\n", short_allocation.head())

    short_allocation.to_csv(path_or_buf='../../data/intermediate_data/regime_change/short_allocation.csv')

    # Plot SPX with regime probabilities
    portfolio_model.plot_regime_probabilities(spx_data=spx_data)
