import quantstats_lumi as qs
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union
import webbrowser
import numpy as np
import os
import shutil
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class StrategyPerformanceAnalyzer:
    """
    Une classe pour analyser la performance d'un portefeuille par rapport à un benchmark.

    Cette classe fournit des méthodes pour préparer les rendements du portefeuille et du benchmark pour l'analyse,
    générer des rapports de backtesting, et calculer des métriques de performance clés.
    """

    def __init__(
        self,
        portfolio_returns: pd.DataFrame,
        benchmark_prices: Union[pd.DataFrame, None],
        strategy_name: str
    ):
        self.portfolio_returns: pd.DataFrame = self._prepare_portfolio_returns(portfolio_returns=portfolio_returns)
        self.benchmark_returns: pd.Series = self._prepare_benchmark_returns(benchmark_prices=benchmark_prices)
        self.strategy_name: str = strategy_name

    @staticmethod
    def _prepare_portfolio_returns(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
        if portfolio_returns.index.dtype != "datetime64[ns]":
            portfolio_returns.index = pd.to_datetime(portfolio_returns.index, format="%Y-%m-%d")

        portfolio_returns.columns = ["Portfolio_Returns"]
        nan_count = portfolio_returns["Portfolio_Returns"].isnull().sum()
        if nan_count > 0:
            portfolio_returns = portfolio_returns.dropna()

        portfolio_returns = portfolio_returns.copy()
        portfolio_returns["Portfolio_Returns"] = pd.to_numeric(portfolio_returns["Portfolio_Returns"])

        return portfolio_returns

    @staticmethod
    def _prepare_benchmark_returns(benchmark_prices: pd.DataFrame) -> Union[pd.Series, None]:
        if benchmark_prices is None:
            return None

        if benchmark_prices.index.dtype != "datetime64[ns]":
            benchmark_prices.index = pd.to_datetime(benchmark_prices.index)

        benchmark_returns = benchmark_prices.pct_change().dropna()

        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] == 1:
                return pd.Series(benchmark_returns.iloc[:, 0], name='SP500')
            elif benchmark_returns.shape[1] > 1:
                # If there are multiple columns, assume the first one is the relevant benchmark
                return pd.Series(benchmark_returns.iloc[:, 0], name='SP500')
        elif isinstance(benchmark_returns, pd.Series):
            # If the result is already a Series, just return it
            return benchmark_returns

    @staticmethod
    def _move_file_to_directory(src_folder: str, dest_folder: str, file_name: str) -> None:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)

        os.makedirs(dest_folder, exist_ok=True)

        try:
            shutil.move(src_path, dest_path)
        except Exception as e:
            print(f"Erreur lors du déplacement du fichier {file_name} de {src_folder} vers {dest_folder}: {e}")

    def generate_backtesting_report_html(
            self,
            rf: float,
            periods_per_year: int,
            grayscale: bool = False,
            dest_folder: str = '../reports',
            match_dates: bool = True,
            open_in_browser: bool = False,
    ) -> None:
        output_file = f"{self.strategy_name}_backtesting_report.html"

        qs.reports.html(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            periods_per_year=periods_per_year,
            title=f"{self.strategy_name} Strategy",
            output=output_file,
            grayscale=grayscale,
            download_filename=output_file,
            match_dates=match_dates,
        )

        self._move_file_to_directory(
            src_folder=os.getcwd(),
            dest_folder=dest_folder,
            file_name=output_file
        )

        if open_in_browser:
            file_path = os.path.join(dest_folder, output_file)
            self._open_html_in_browser(file_path=file_path)

    def generate_backtesting_report_full(
            self, rf: float, grayscale: bool = False, match_dates: bool = True
    ) -> None:
        qs.reports.full(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            grayscale=grayscale,
            match_dates=match_dates,
        )

    def compute_performance_metrics(
            self,
            rf: float,
            mode: str = "full",
            prepare_returns: bool = False,
            match_dates: bool = True,
    ) -> None:
        qs.reports.metrics(
            returns=self.portfolio_returns["Portfolio_Returns"],
            benchmark=self.benchmark_returns,
            rf=rf,
            mode=mode,
            prepare_returns=prepare_returns,
            match_dates=match_dates,
        )

    def compute_alpha_beta_tracking_error(self, portfolio_returns, benchmark_returns, rf: float) -> dict:
        # Calcul des rendements excédentaires par rapport au taux sans risque
        excess_portfolio = portfolio_returns - rf
        excess_sp500 = benchmark_returns - rf

        # Aligner les deux séries sur les mêmes indices (dates)
        excess_portfolio, excess_sp500 = excess_portfolio.align(excess_sp500, join='inner', axis=0)

        # Ajouter une constante pour l'interception (alpha)
        X = sm.add_constant(excess_sp500)
        y = excess_portfolio

        # Effectuer la régression linéaire
        model = sm.OLS(y, X).fit()

        # Extraire alpha (interception) et beta (pente)
        alpha = model.params['const']
        beta = model.params[1]

        # L'erreur de suivi est l'écart-type des résidus
        tracking_error = np.std(model.resid)

        return {
            "Alpha": alpha,
            "Beta": beta,
            "Tracking Error": tracking_error
        }
    
    def calculate_rolling_alpha_beta_tracking_error(self, window: int) -> pd.DataFrame:
        """
        Calcule les valeurs d'alpha, de beta et de l'erreur de suivi (tracking error) sur une fenêtre glissante.

        :param window: La taille de la fenêtre de calcul roulante (en nombre de périodes).
        :return: Un DataFrame contenant les valeurs d'alpha, de beta et de l'erreur de suivi pour chaque fenêtre.
        """
        rolling_results = {
            "Alpha": [],
            "Beta": [],
            "Tracking Error": []
        }

        # Merge portfolio returns and benchmark returns into a single DataFrame
        combined_df = pd.concat([self.portfolio_returns, self.benchmark_returns], axis=1, join='inner')
        combined_df.columns = ["Portfolio_Returns", "Benchmark_Returns"]

        # Iterate through the combined DataFrame in rolling windows
        for i in range(len(combined_df) - window + 1):
            # Slice the data for the current window
            rolling_window = combined_df.iloc[i:i + window]

            # Extract portfolio and benchmark windows from the combined window
            portfolio_window = rolling_window["Portfolio_Returns"]
            benchmark_window = rolling_window["Benchmark_Returns"]

            # Ensure the window data is not empty and contains no NaN values
            if portfolio_window.empty or benchmark_window.empty or portfolio_window.isnull().sum() > 0 or benchmark_window.isnull().sum() > 0:
                continue  # Skip this window if it's empty or contains NaN values

            # Calculate alpha, beta, and tracking error using the compute_alpha_beta_tracking_error method
            result = self.compute_alpha_beta_tracking_error(portfolio_window, benchmark_window, rf=0.0)  # Assuming rf = 0.0 for simplicity
            rolling_results["Alpha"].append(result["Alpha"])
            rolling_results["Beta"].append(result["Beta"])
            rolling_results["Tracking Error"].append(result["Tracking Error"])

        # Create a date index that corresponds to the end of each rolling window
        rolling_index = combined_df.index[window - 1:]

        # Convert results into a DataFrame
        rolling_df = pd.DataFrame(rolling_results, index=rolling_index)

        return rolling_df


    @staticmethod
    def _open_html_in_browser(file_path: str) -> None:
        webbrowser.open(f"file://{os.path.abspath(file_path)}")

    @staticmethod
    def _compute_omega_ratio(returns: pd.Series, required_return: float = 0.0, periods: int = 252) -> float:
        if len(returns) < 2 or required_return <= -1:
            return np.nan

        return_threshold = (1 + required_return) ** (1.0 / periods) - 1 if periods != 1 else required_return

        returns_less_thresh = returns - return_threshold
        numer = returns_less_thresh[returns_less_thresh > 0.0].sum()
        denom = -returns_less_thresh[returns_less_thresh < 0.0].sum()

        if denom > 0.0:
            return numer / denom

        return np.nan

    def calculate_information_ratio(self) -> float:
        combined_returns = self.portfolio_returns.join(self.benchmark_returns.to_frame(), how='inner')
        excess_returns = combined_returns["Portfolio_Returns"] - combined_returns["SP500"]

        avg_excess_return = excess_returns.mean()
        std_excess_return = excess_returns.std()

        if std_excess_return != 0:
            return avg_excess_return / std_excess_return
        return np.nan

    @staticmethod
    def _compute_cumulative_returns(returns: pd.Series) -> float:
        return (1 + returns).prod() - 1

    def get_key_performance_metrics(self, turnover: float, rf: float, periods_per_year: int, annualize: bool = True) -> pd.DataFrame:
        portfolio_returns = self.portfolio_returns["Portfolio_Returns"]

        alpha_beta_te_dict = self.compute_alpha_beta_tracking_error(rf=rf)

        metrics = {
            "Cumulative_Returns": self._compute_cumulative_returns(returns=portfolio_returns),
            "CAGR": qs.stats.cagr(returns=portfolio_returns, rf=rf, periods=periods_per_year),
            "Alpha": alpha_beta_te_dict["Alpha"],
            "Beta": alpha_beta_te_dict["Beta"],
            "Tracking_Error": alpha_beta_te_dict["Tracking Error"],
            "Turnover": turnover,
            "Information Ratio": self.calculate_information_ratio(),
            "Sharpe": qs.stats.sharpe(returns=portfolio_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "Volatility": qs.stats.volatility(returns=portfolio_returns, periods=periods_per_year, annualize=annualize),
            "Max_Drawdown": qs.stats.max_drawdown(prices=portfolio_returns),
            "Sortino": qs.stats.sortino(returns=portfolio_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "R_Squared": qs.stats.r_squared(returns=portfolio_returns, benchmark=self.benchmark_returns),
            "Omega_Ratio": self._compute_omega_ratio(returns=portfolio_returns, periods=periods_per_year),
        }

        return pd.DataFrame(metrics, index=[self.strategy_name]).T

    def get_key_performance_metrics_benchmark(self, rf: float, periods_per_year: int, annualize: bool = True) -> pd.DataFrame:
        benchmark_returns = self.benchmark_returns

        metrics = {
            "Cumulative_Returns": self._compute_cumulative_returns(returns=benchmark_returns),
            "CAGR": qs.stats.cagr(returns=benchmark_returns, rf=rf, periods=periods_per_year),
            "Sharpe": qs.stats.sharpe(returns=benchmark_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "Volatility": qs.stats.volatility(returns=benchmark_returns, periods=periods_per_year, annualize=annualize),
            "Max_Drawdown": qs.stats.max_drawdown(prices=benchmark_returns),
            "Sortino": qs.stats.sortino(returns=benchmark_returns, rf=rf, periods=periods_per_year, annualize=annualize),
            "R_Squared": 1.0,
            "Omega_Ratio": self._compute_omega_ratio(returns=benchmark_returns, periods=periods_per_year),
        }

        return pd.DataFrame(metrics, index=["Benchmark"]).T
    
if __name__ == "__main__":
    # Load the portfolio returns
    portfolio_returns = pd.read_csv('data/raw_data/strategy_returns_hist_gradient_boosting_regressor.csv',
                                    parse_dates=True, index_col=0)
    sp500_prices = pd.read_csv('data/raw_data/sp500_daily.csv', parse_dates=True, index_col=0)["Close"]

    # Initialize the StrategyPerformanceAnalyzer object
    analyzer = StrategyPerformanceAnalyzer(
        portfolio_returns=portfolio_returns,
        benchmark_prices=sp500_prices,
        strategy_name="Gradient Boosting Regressor"
    )

    # compute rolling alpha, beta, and tracking error
    rolling_results = analyzer.calculate_rolling_alpha_beta_tracking_error(window=252)

    plt.plot(rolling_results["Beta"], label="Beta")
    plt.show()