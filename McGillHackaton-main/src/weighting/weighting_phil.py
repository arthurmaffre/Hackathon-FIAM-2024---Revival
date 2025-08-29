import pandas as pd
import riskfolio as rp
from src.weighting.defensive_allocation import RegimeBasedPortfolio

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)


class WeightingStrategyWithDynamicRegime:
    def __init__(self, returns, long_signals, regime_data, sp500_data, defensive_tickers, aggresive_tickers):
        """
        Initialize the WeightingStrategy with the provided data.

        Parameters:
            returns (pd.DataFrame): Returns data.
            long_signals (pd.DataFrame): Long signals for the assets.
            short_signals (pd.DataFrame, optional): Short signals for the assets.
            short_allocation (pd.Series, optional): Short allocation values (dynamically changing uppersht).
        """
        self.returns = returns.sort_index()

        # make sure index is a DatetimeIndex
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns.index = pd.to_datetime(self.returns.index)

        # Set attributes
        self.long_signals = long_signals.astype(bool)
        self.rebalancing_dates = self.long_signals.index
        self.regime_data = regime_data
        self.sp500_data = sp500_data
        self.sp500_returns = self.sp500_data['Close'].pct_change().dropna()
        self.defensive_tickers = defensive_tickers
        self.aggressive_tickers = aggresive_tickers

        # Ensure period[M] format for all relevant DataFrames
        self._ensure_period_index()
        self.parameters = {}
        self.weights = None  # To store combined long and short weights

    def _ensure_period_index(self):
        """
        Convert index of returns, long_signals, short_signals, and short_allocation to period[M] if necessary.
        """
        def convert_to_period(df):
            if not isinstance(df.index, pd.PeriodIndex):
                df.index = df.index.to_period('M')
            return df

        self.long_signals = convert_to_period(self.long_signals)

    def set_parameters(self,
                       method_mu: str = 'hist',
                       method_cov: str = 'hist',
                       model: str = 'Classic',
                       rm: str = 'MV',
                       obj: str = 'Sharpe',
                       rf: float = 0.0,
                       l: float = 0.0,
                       hist: bool = True,
                       window: int = 252,
                       budget: float = 1.0,
                       max_weight_long: float = 0.1,
                       min_weight_long: float = 0.001,
                       weight_threshold: float = 1e-8,
                       min_assets = 50,
                       max_assets = 100,
                       turnover_limit = 0.25,
                       **kwargs: dict) -> None:
        """
        Set optimization parameters.

        Parameters:
            method_mu (str): Method to estimate expected returns.
            method_cov (str): Method to estimate covariance matrix.
            model (str): Optimization model type.
            rm (str): Risk measure.
            obj (str): Objective function.
            rf (float): Risk-free rate.
            l (float): Risk aversion factor.
            hist (bool): Use historical data or not.
            window (int): Look-back window for returns.
            budget (float): Total budget.
            max_weight_long (float): Maximum weight for a long position.
            min_weight_long (float): Minimum weight for a long position.
            max_weight_short (float): Maximum weight for a short position.
            min_weight_short (float): Minimum weight for a short position.
            weight_threshold (float): Threshold to filter small weights.
            minimum_bear_alloc (float): Minimum bear allocation for short positions.
        """
        self.parameters = {
            'method_mu': method_mu,
            'method_cov': method_cov,
            'model': model,
            'rm': rm,
            'obj': obj,
            'rf': rf,
            'l': l,
            'hist': hist,
            'window': window,
            'budget': budget,
            'max_weight_long': max_weight_long,
            'min_weight_long': min_weight_long,
            'weight_threshold': weight_threshold,  # New parameter for weight threshold
            'min_assets': min_assets,
            'max_assets': max_assets,
            'turnover_limit': turnover_limit
        }
        self.parameters.update(kwargs)

    def optimize_portfolios(self):

        weights_list = []
        last_weights = None
        regime_model = RegimeBasedPortfolio(data=self.regime_data)

        for date in self.rebalancing_dates:
            # Calculate standard deviation of SP500 returns for past 252 days
            sp500_returns_window = self.sp500_returns.loc[:date.to_timestamp()].tail(252)
            sp500_std = sp500_returns_window.std()

            # Get defensive and aggressive tickers for the date
            defensive_allocation = regime_model.get_defensive_allocation_for_rebalancing_date(date)
            aggressive_allocation = 1 - defensive_allocation

            # Get long tickers for the date
            long_tickers = self.long_signals.loc[date]
            long_tickers = long_tickers[long_tickers].index.tolist()

            all_tickers = list(set(long_tickers))

            # Get returns data window
            window = self.parameters.get('window', 252)

            # Check if the returns index is already a PeriodIndex or a DatetimeIndex
            returns_window = self.returns.loc[:date.to_timestamp(), all_tickers].tail(window)  # Use period index directly


            if not returns_window.empty and returns_window.shape[0] >= 2:
                port = rp.Portfolio(returns=returns_window)
                port.assets_stats(method_mu=self.parameters['method_mu'], method_cov=self.parameters['method_cov'])

                # Dynamically adjust uppersht and upperlng for each date
                upperlng = 1.0  # Adjust upperlng based on uppersht
                port.sht = False
                port.upperdev = sp500_std + 0.01 # check if this is the right value
                port.budget = self.parameters['budget']
                port.upperlng = upperlng
                port.nea = self.parameters['min_assets']
                port.card = self.parameters['max_assets']
                port.allowTO = True
                port.turnover = self.parameters['turnover_limit']
                port.benchweights = last_weights
                sector_types = []

                for ticker in all_tickers:
                    if ticker in self.defensive_tickers:
                        sector_types.append('Defensive')
                    elif ticker in self.aggressive_tickers:
                        sector_types.append('Aggressive')

                asset_classes = pd.DataFrame({'Assets': all_tickers,
                                              'Sector': sector_types})
                
        
                constraints = pd.DataFrame({
                    'Disabled': [],
                    'Type': ['All Assets', 'All Assets', 'Classes', 'Classes'],
                    'Set': ['', '', 'Sector', 'Sector'],
                    'Position': ['', '', 'Defensive', 'Aggressive'],
                    'Sign': ['<=', '>=', '<=', '>='],
                    'Weight': [self.parameters['max_weight_long'], self.parameters['min_weight_long'], defensive_allocation, (aggressive_allocation - 0.10)],
                    'Type Relative': ['', '', '', ''],
                    'Relative Set': ['', '', '', ''],
                    'Relative': ['', '', '', ''],
                    'Factor': ['', '', '', ''],
                })

                A, B = rp.assets_constraints(constraints, asset_classes)
                port.ainequality = A
                port.binequality = B
                port.solvers = ['CLARABEL']  # Use CLARABEL solver to avoid warnings

                try:
                    w = port.optimization(
                        model=self.parameters['model'],
                        rm=self.parameters['rm'],
                        obj=self.parameters['obj'],
                        rf=self.parameters['rf'],
                        l=self.parameters['l'],
                        hist=self.parameters['hist']
                    )
                    if w is None:
                        raise ValueError("Optimization returned None")
                    else:
                        print(f"Optimization successful on {date}")

                    w = w.T
                    w.index = [date]

                    weights_list.append(w)
                    last_weights = w

                except Exception as e:
                    print(f"Optimization failed on {date}: {e}")
                    if last_weights is not None:
                        w = last_weights.copy()
                        w.index = [date]
                        weights_list.append(w)
            else:
                # Use last weights if not enough data
                if last_weights is not None:
                    w = last_weights.copy()
                    w.index = [date]
                    weights_list.append(w)

        if weights_list:
            self.weights = pd.concat(weights_list).fillna(0)
        else:
            self.weights = pd.DataFrame()

        return self.weights

# NEEDS TO BE CHANGED STILL
if __name__ == '__main__':
    # Assuming you already have the DataFrames 'returns', 'long_signals', 'short_signals', and 'short_allocation'

    returns = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/compute_returns/daily_returns.csv',
        index_col=0, parse_dates=True
    )
    long_signals = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/create_long_short_portfolio/long_signals.csv',
        index_col=0, parse_dates=True
    )
    short_signals = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/create_long_short_portfolio/short_signals.csv',
        index_col=0, parse_dates=True
    )
    short_allocation = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/regime_change/short_allocation.csv',
        index_col=0, parse_dates=True
    )

    # Initialize the portfolio optimizer
    optimizer = WeightingStrategyWithDynamicRegime(
        returns=returns, long_signals=long_signals, short_signals=short_signals, short_allocation=short_allocation
    )

    # Set optimization parameters
    optimizer.set_parameters(
        method_mu='hist',
        method_cov='ledoit',
        model='Classic',
        rm='MV',
        obj='Sharpe',
        rf=0,
        l=0,
        hist=True,
        window=1000,  # Use the last 1000 days of data
        budget=1.0,
        max_weight_long=0.1,  # Max weight per long asset
        min_weight_long=0.001,  # Min weight per long asset
        max_weight_short=-0.1,  # Max weight per short asset
        min_weight_short=-0.001,  # Min weight per short asset
        weight_threshold=1e-8  # Threshold for small weights
    )

    # Run the optimization
    weights_df = optimizer.optimize_portfolios()

    # Display the weights
    print("\nPortfolio Weights:")
    print(weights_df.head())

    # weighting_strategy_base = WeightingStrategyBase(returns=returns, long_signals=long_signals, short_signals=short_signals)
    # weighting_strategy_base.set_parameters(
    #     method_mu='hist',
    #     method_cov='ledoit',
    #     model='Classic',
    #     rm='MV',
    #     obj='Sharpe',
    #     rf=0,
    #     l=0,
    #     hist=True,
    #     window=252
    # )
    # long_weights, short_weights = weighting_strategy_base.optimize_portfolios()
    #
    # print("\nLong Portfolio Weights:")
    # print(long_weights.head())
    # print(long_weights.tail())
    #
    # print("\nShort Portfolio Weights:")
    # print(short_weights.head())
    # print(short_weights.tail())



    # Définition des paramètres d'optimisation
    # ------- method_mu -------#
    # - 'hist': use historical estimates.
    # - 'ewma1'': use ewma with adjust=True, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
    # - 'ewma2': use ewma with adjust=False, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
    # - 'JS': James-Stein estimator. For more information see :cite:`a-Meucci2005` and :cite:`a-Feng2016`.
    # - 'BS': Bayes-Stein estimator. For more information see :cite:`a-Jorion1986`.
    # - 'BOP': BOP estimator. For more information see :cite:`a-Bodnar2019`.

    # ------- method_cov -------#
    # - 'hist': use historical estimates.
    # - 'ewma1'': use ewma with adjust=True, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
    # - 'ewma2': use ewma with adjust=False, see `EWM <https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows>`_ for more details.
    # - 'ledoit': use the Ledoit and Wolf Shrinkage method.
    # - 'oas': use the Oracle Approximation Shrinkage method.
    # - 'shrunk': use the basic Shrunk Covariance method.
    # - 'gl': use the basic Graphical Lasso Covariance method.
    # - 'jlogo': use the j-LoGo Covariance method. For more information see: :cite:`a-jLogo`.
    # - 'fixed': denoise using fixed method. For more information see chapter 2 of :cite:`a-MLforAM`.
    # - 'spectral': denoise using spectral method. For more information see chapter 2 of :cite:`a-MLforAM`.
    # - 'shrink': denoise using shrink method. For more information see chapter 2 of :cite:`a-MLforAM`.
    # - 'gerber1': use the Gerber statistic 1. For more information see: :cite:`a-Gerber2021`.
    # - 'gerber2': use the Gerber statistic 2. For more information see: :cite:`a-Gerber2021`.

    # ------- model -------#
    # model : str can be {'Classic', 'BL', 'FM' or 'BLFM'}
    #         The model used for optimize the portfolio.
    #         The default is 'Classic'. Possible values are:
    #
    #         - 'Classic': use estimates of expected return vector and covariance matrix that depends on historical data.
    #         - 'BL': use estimates of expected return vector and covariance matrix based on the Black Litterman model.
    #         - 'FM': use estimates of expected return vector and covariance matrix based on a Risk Factor model specified by the user.
    #         - 'BLFM': use estimates of expected return vector and covariance matrix based on Black Litterman applied to a Risk Factor model specified by the user.
    #
    #     rm : str, optional
    #         The risk measure used to optimize the portfolio.
    #         The default is 'MV'. Possible values are:
    #
    #         - 'MV': Standard Deviation.
    #         - 'KT': Square Root of Kurtosis.
    #         - 'MAD': Mean Absolute Deviation.
    #         - 'GMD': Gini Mean Difference.
    #         - 'MSV': Semi Standard Deviation.
    #         - 'SKT': Square Root of Semi Kurtosis.
    #         - 'FLPM': First Lower Partial Moment (Omega Ratio).
    #         - 'SLPM': Second Lower Partial Moment (Sortino Ratio).
    #         - 'CVaR': Conditional Value at Risk.
    #         - 'TG': Tail Gini.
    #         - 'EVaR': Entropic Value at Risk.
    #         - 'RLVaR': Relativistic Value at Risk.
    #         - 'WR': Worst Realization (Minimax).
    #         - 'RG': Range of returns.
    #         - 'CVRG': CVaR range of returns.
    #         - 'TGRG': Tail Gini range of returns.
    #         - 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
    #         - 'ADD': Average Drawdown of uncompounded cumulative returns.
    #         - 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
    #         - 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
    #         - 'RLDaR': Relativistic Drawdown at Risk of uncompounded cumulative returns.
    #         - 'UCI': Ulcer Index of uncompounded cumulative returns.
    #
    #     obj : str can be {'MinRisk', 'Utility', 'Sharpe' or 'MaxRet'}.
    #         Objective function of the optimization model.
    #         The default is 'Sharpe'. Possible values are:
    #
    #         - 'MinRisk': Minimize the selected risk measure.
    #         - 'Utility': Maximize the Utility function :math:`\mu w - l \phi_{i}(w)`.
    #         - 'Sharpe': Maximize the risk adjusted return ratio based on the selected risk measure.
    #         - 'MaxRet': Maximize the expected return of the portfolio.