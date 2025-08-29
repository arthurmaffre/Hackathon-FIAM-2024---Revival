import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import logging

class ReturnsInterpretability:
    """
    A class to interpret portfolio returns using asset pricing models like Fama-French.

    This class provides methods to fit Fama-French models, summarize regression results,
    and plot rolling betas to analyze dynamic exposures to various risk factors.

    Attributes
    ----------
    portfolio_returns : pd.Series
        Series containing portfolio returns.
    benchmark_returns : pd.Series or None
        Series containing benchmark returns. Can be None if not applicable.
    ff_factors : pd.DataFrame
        DataFrame containing Fama-French factors.
    rolling_betas : pd.DataFrame
        DataFrame to store the rolling betas.
    window : int
        Rolling window size.

    Methods
    -------
    fit_fama_french_five_factor_model(returns: pd.Series)
        Fits the Fama-French 5-Factor model to the provided returns using the stored risk factors.
    summarize_fama_french_model()
        Prints the summary of the fitted Fama-French model.
    calculate_rolling_betas(returns: pd.Series)
        Calculates rolling betas for each Fama-French factor over the specified window.
    plot_rolling_betas()
        Plots the rolling betas for each Fama-French factor over time.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        ff_factors: pd.DataFrame,
        window: int = 60,
    ):
        """
        Initializes the ReturnsInterpretability class.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Series containing portfolio returns.
        benchmark_returns : pd.Series or None
            Series containing benchmark returns. Can be None if not applicable.
        ff_factors : pd.DataFrame
            DataFrame containing Fama-French factors.
        window : int, optional
            Rolling window size. Default is 60.
        """
        self.portfolio_returns = self._prepare_return_series(portfolio_returns, 'Portfolio Returns')
        self.benchmark_returns = self._prepare_return_series(benchmark_returns, 'Benchmark Returns') if benchmark_returns is not None else None
        self.ff_factors = self._prepare_ff_factors(ff_factors)
        self.rolling_betas = None  # To store the rolling betas DataFrame
        self.window = window
        self.ff_model = None  # To store the fitted model


    @staticmethod
    def _prepare_return_series(return_series: pd.Series, name: str) -> pd.Series:
        """
        Prepares the return series for analysis.

        Parameters
        ----------
        return_series : pd.Series
            Series containing returns.
        name : str
            Name of the return series for logging purposes.

        Returns
        -------
        pd.Series
            Cleaned return series.
        """
        if return_series is None:
            return None
        if not isinstance(return_series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")

        # Ensure the index is datetime
        if not isinstance(return_series.index, pd.DatetimeIndex):
            try:
                return_series.index = pd.to_datetime(return_series.index)
            except Exception as e:
                raise ValueError(f"Could not convert index of {name} to datetime: {e}")

        # Drop NaN values
        return_series = return_series.dropna()

        # Ensure numeric dtype
        return_series = pd.to_numeric(return_series, errors='coerce').dropna()

        return return_series

    @staticmethod
    def _prepare_ff_factors(ff_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the Fama-French factors DataFrame for analysis.

        Parameters
        ----------
        ff_factors : pd.DataFrame
            DataFrame containing the Fama-French factors.

        Returns
        -------
        pd.DataFrame
            Cleaned Fama-French factors DataFrame.
        """
        if not isinstance(ff_factors, pd.DataFrame):
            raise TypeError("Fama-French factors must be provided as a pandas DataFrame.")
        
        # Ensure datetime index
        if not isinstance(ff_factors.index, pd.DatetimeIndex):
            try:
                ff_factors.index = pd.to_datetime(ff_factors.index)
            except Exception as e:
                raise ValueError(f"Could not convert index of Fama-French factors to datetime: {e}")
        
        # Drop NaN values
        ff_factors = ff_factors.dropna()

        return ff_factors

    @staticmethod
    def _verify_data(returns: pd.Series, risk_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Verifies and aligns the returns and risk factors data.

        Parameters
        ----------
        returns : pd.Series
            The returns series to be analyzed.
        risk_factors : pd.DataFrame
            DataFrame containing the risk factors.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with aligned dates, containing returns and risk factors.

        Raises
        ------
        ValueError
            If the returns and risk factors cannot be aligned.
        """
        if not isinstance(returns, pd.Series):
            raise TypeError("Returns must be a pandas Series.")
        if not isinstance(risk_factors, pd.DataFrame):
            raise TypeError("Risk factors must be a pandas DataFrame.")

        # Ensure datetime index
        if not isinstance(returns.index, pd.DatetimeIndex):
            try:
                returns.index = pd.to_datetime(returns.index)
            except Exception as e:
                raise ValueError(f"Could not convert returns index to datetime: {e}")
        if not isinstance(risk_factors.index, pd.DatetimeIndex):
            try:
                risk_factors.index = pd.to_datetime(risk_factors.index)
            except Exception as e:
                raise ValueError(f"Could not convert risk factors index to datetime: {e}")

        # Drop NaNs
        returns = returns.dropna()
        risk_factors = risk_factors.dropna()

        # Align the dates
        combined = pd.concat([returns, risk_factors], axis=1, join='inner')

        if combined.empty:
            raise ValueError("No overlapping dates between returns and risk factors after alignment.")

        return combined

    def _fit_model(self, combined: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Fits the Fama-French 5-Factor model using the combined DataFrame.

        Parameters
        ----------
        combined : pd.DataFrame
            Combined DataFrame containing aligned returns and risk factors.

        Returns
        -------
        RegressionResultsWrapper
            The fitted OLS regression results.
        """
        required_columns = ['smb', 'hml', 'rmw', 'cma', 'umd', 'rf']
        for col in required_columns:
            if col not in combined.columns:
                raise ValueError(f"Risk factors DataFrame is missing required column: {col}")

        # Calculate excess returns: R_p - R_f
        combined.loc[:, 'Excess_Returns'] = combined.iloc[:, 0] - combined['rf']

        # Define dependent and independent variables
        y = combined['Excess_Returns']
        X = combined[['smb', 'hml', 'rmw', 'cma', 'umd']]
        X = sm.add_constant(X)

        # Fit the OLS regression model
        try:
            model = sm.OLS(y, X).fit()
            return model
        except Exception as e:
            logging.error(f"Error fitting the regression model: {e}")
            raise ValueError(f"Error fitting the regression model: {e}")

    def calculate_rolling_betas(self, returns: pd.Series) -> pd.DataFrame:
        """
        Calculates rolling betas for each Fama-French factor over the specified window.

        Parameters
        ----------
        returns : pd.Series
            The returns series to model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing rolling beta coefficients with dates as index and factors as columns.
        """
        # Verify data
        combined = self._verify_data(returns, self.ff_factors)

        # Initialize DataFrame to store rolling betas
        beta_columns = ['const', 'smb', 'hml', 'rmw', 'cma', 'umd']
        rolling_betas = pd.DataFrame(index=combined.index[self.window - 1:], columns=beta_columns)

        # Iterate over rolling windows
        for end in range(self.window, len(combined) + 1):
            window_data = combined.iloc[end - self.window:end]

            # Fit the model using the window data
            try:
                model = self._fit_model(window_data)
                rolling_betas.iloc[end - self.window] = model.params
            except Exception as e:
                logging.warning(f"Failed to fit model for window ending at {combined.index[end - 1]}: {e}")
                rolling_betas.iloc[end - self.window] = np.nan

        # Drop NaN values resulting from failed fits
        rolling_betas = rolling_betas.dropna()

        self.rolling_betas = rolling_betas

        return rolling_betas

    def plot_rolling_betas(self, figsize: tuple = (14, 7), save_path: str = None) -> None:
        """
        Plots the rolling betas for each Fama-French factor over time.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the plot. Default is (14, 7).
        save_path : str, optional
            If provided, saves the plot to the specified path.

        Returns
        -------
        None
        """
        if self.rolling_betas is None or self.rolling_betas.empty:
            raise AttributeError("Rolling betas have not been calculated yet. Call 'calculate_rolling_betas' first.")

        rolling_betas_df = self.rolling_betas

        plt.figure(figsize=figsize)

        for col in rolling_betas_df.columns:
            plt.plot(rolling_betas_df.index, rolling_betas_df[col], label=col)

        plt.title(f'Rolling Betas ({self.window} periods) - Fama-French 5-Factor Model')
        plt.xlabel('Date')
        plt.ylabel('Beta Coefficient')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Format the x-axis dates
        from matplotlib.dates import DateFormatter
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Rolling betas plot saved to {save_path}.")

        plt.show()

def plot_rolling_betas_comparison(sp500_rolling_betas: pd.DataFrame, 
                                  personal_portfolio_rolling_betas: pd.DataFrame, 
                                  sp500_returns: pd.Series, 
                                  personal_returns: pd.Series, 
                                  figsize: tuple = (14, 10)) -> None:
    """
    Plots 5 graphs comparing rolling betas of SP500 and personal portfolio, and their cumulative returns.

    Parameters
    ----------
    sp500_rolling_betas : pd.DataFrame
        Rolling beta coefficients for the SP500 portfolio. Expected columns: ['const', 'smb', 'hml', 'rmw', 'cma', 'umd']
    personal_portfolio_rolling_betas : pd.DataFrame
        Rolling beta coefficients for the personal portfolio. Expected columns: ['const', 'smb', 'hml', 'rmw', 'cma', 'umd']
    sp500_returns : pd.Series
        Returns of the SP500 portfolio.
    personal_returns : pd.Series
        Returns of the personal portfolio.
    figsize : tuple, optional
        Figure size for the plots. Default is (14, 10).

    Returns
    -------
    None
    """
    # List of factors to plot (excluding 'const' if not needed)
    factors = ['smb', 'hml', 'rmw', 'cma', 'umd']

    # Align the dates for rolling betas
    common_dates = sp500_rolling_betas.index.intersection(personal_portfolio_rolling_betas.index)
    sp500_rolling_betas = sp500_rolling_betas.loc[common_dates]
    personal_portfolio_rolling_betas = personal_portfolio_rolling_betas.loc[common_dates]

    if sp500_rolling_betas.empty or personal_portfolio_rolling_betas.empty:
        raise ValueError("Rolling betas DataFrames do not have overlapping dates for comparison.")

    # Align the dates for returns
    common_returns_dates = sp500_returns.index.intersection(personal_returns.index)
    sp500_returns_aligned = sp500_returns.loc[common_returns_dates]
    personal_returns_aligned = personal_returns.loc[common_returns_dates]

    # Calculate cumulative returns
    sp500_cum_returns = (1 + sp500_returns_aligned).cumprod() - 1
    personal_cum_returns = (1 + personal_returns_aligned).cumprod() - 1

    # Ensure cumulative returns are aligned with the rolling betas dates
    common_cum_return_dates = sp500_cum_returns.index.intersection(common_dates)
    sp500_cum_returns = sp500_cum_returns.loc[common_cum_return_dates]
    personal_cum_returns = personal_cum_returns.loc[common_cum_return_dates]

    # Create subplots: one for each factor and one for the cumulative returns
    num_subplots = len(factors) + 1  # Additional plot for cumulative returns
    fig, axes = plt.subplots(num_subplots, 1, figsize=figsize, sharex=True)

    # Plot rolling betas for each factor
    for i, factor in enumerate(factors):
        ax = axes[i]
        ax.plot(sp500_rolling_betas.index, sp500_rolling_betas[factor], label='SP500', color='blue', linestyle='-')
        ax.plot(personal_portfolio_rolling_betas.index, personal_portfolio_rolling_betas[factor], label='Personal Portfolio', color='orange', linestyle='--')

        ax.set_ylabel(f"{factor.upper()} Beta")
        ax.set_title(f"Rolling {factor.upper()} Beta Comparison")
        ax.legend(loc='best')
        ax.grid(True)

    # Plot cumulative returns on the last subplot
    ax_returns = axes[-1]
    ax_returns.plot(sp500_cum_returns.index, sp500_cum_returns, label='SP500 Cumulative Returns', color='green', linestyle='-')
    ax_returns.plot(personal_cum_returns.index, personal_cum_returns, label='Personal Portfolio Cumulative Returns', color='red', linestyle='--')
    ax_returns.set_ylabel('Cumulative Returns')
    ax_returns.set_title('Cumulative Portfolio Returns Comparison')
    ax_returns.legend(loc='best')
    ax_returns.grid(True)

    # Improve date formatting on the x-axis
    date_locator = AutoDateLocator()
    date_formatter = DateFormatter('%Y-%m')
    axes[-1].xaxis.set_major_locator(date_locator)
    axes[-1].xaxis.set_major_formatter(date_formatter)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.xlabel('Date')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Increase space between plots
    plt.show()

if __name__ == "__main__":
    # Load the portfolio returns
    portfolio_returns = pd.read_csv('data/raw_data/strategy_returns_hist_gradient_boosting_regressor.csv',
                                    parse_dates=True, index_col=0)
    sp500_prices = pd.read_csv('data/raw_data/sp500_daily.csv', parse_dates=True, index_col=0)

    # Calculate SP500 returns
    sp500_returns = sp500_prices['Close'].pct_change().dropna()

    # Load the Fama-French factors
    ff_factors = pd.read_csv('data/raw_data/five_factors_portfolio.csv',
                             parse_dates=True, index_col=0)

    # Ensure the indices are datetime indices
    # This is already done via parse_dates and index_col when reading CSVs

    # Extract the returns series from the DataFrames
    portfolio_returns = portfolio_returns['Realized_Return'].dropna()
    sp500_returns = sp500_returns.dropna()

    # Initialize the ReturnsInterpretability objects
    port_returns_interpretability = ReturnsInterpretability(portfolio_returns, None, ff_factors)
    sp500_returns_interpretability = ReturnsInterpretability(sp500_returns, None, ff_factors)

    # Calculate rolling betas
    port_rolling_betas = port_returns_interpretability.calculate_rolling_betas(portfolio_returns)
    sp500_rolling_betas = sp500_returns_interpretability.calculate_rolling_betas(sp500_returns)

    # Plot the rolling betas
    plot_rolling_betas_comparison(sp500_rolling_betas, port_rolling_betas, sp500_returns, portfolio_returns)
