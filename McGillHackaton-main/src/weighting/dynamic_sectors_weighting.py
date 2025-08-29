import pandas as pd
import numpy as np
import riskfolio as rp

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)


class WeightingStrategyWithDynamicRegime:
    def __init__(
        self,
        returns: pd.DataFrame,
        long_signals: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        defensive_allocation: pd.DataFrame,
        us_stocks_sectors: pd.DataFrame,
    ):
        """
        Initialize the WeightingStrategy with the provided data.

        Parameters:
            returns (pd.DataFrame): Daily returns data indexed by date.
            long_signals (pd.DataFrame): Long signals for the assets indexed by date.
            benchmark_prices (pd.DataFrame): Benchmark prices (e.g., S&P 500) indexed by date.
            defensive_allocation (pd.DataFrame): Defensive allocation percentages indexed by date.
            us_stocks_sectors (pd.DataFrame): DataFrame mapping tickers to their sectors and categories.
        """
        self._prepare_data(
            returns,
            long_signals,
            benchmark_prices,
            defensive_allocation,
            us_stocks_sectors,
        )
        self.parameters = {}
        self.weights = None  # To store the optimized weights

    def _prepare_data(
        self,
        returns: pd.DataFrame,
        long_signals: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        defensive_allocation: pd.DataFrame,
        us_stocks_sectors: pd.DataFrame,
    ):
        self.returns = returns.copy()
        self.returns.index = pd.to_datetime(self.returns.index)
        self.returns.sort_index(inplace=True)

        self.long_signals = long_signals.astype(bool)

        self.benchmark_returns = benchmark_prices['Close'].pct_change().dropna()

        self.us_stocks_sectors = us_stocks_sectors.copy()
        self._clean_tickers_column()

        self.aggressive_tickers = self._get_tickers_by_sector_category('Aggressive')
        self.defensive_tickers = self._get_tickers_by_sector_category('Defensive')

        self.defensive_allocation = defensive_allocation.copy()
        self._ensure_period_index()

    def _ensure_period_index(self):
        def convert_to_period(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df.index, pd.PeriodIndex):
                df.index = df.index.to_period('M')
            return df

        self.long_signals = convert_to_period(self.long_signals)
        self.defensive_allocation = convert_to_period(self.defensive_allocation)
        self.defensive_allocation = self.defensive_allocation.groupby(level=0).tail(1)
        self.rebalancing_dates = self.long_signals.index

    def _clean_tickers_column(self):
        self.us_stocks_sectors['Tickers'] = self.us_stocks_sectors['Tickers'].str.replace(
            ' US EQUITY', '', regex=False
        )

    def _get_tickers_by_sector_category(self, category: str) -> list:
        return self.us_stocks_sectors[
            self.us_stocks_sectors['Sector Category'] == category
        ]['Tickers'].tolist()

    def set_parameters(
        self,
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
        max_turnover: float = 0.25,
        **kwargs,
    ) -> None:
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
            'weight_threshold': weight_threshold,
            'max_turnover': max_turnover,
        }
        self.parameters.update(kwargs)

    def _calculate_benchmark_monthly_volatility(
        self, date: pd.Period, monthly_trading_days: int = 21
    ) -> float:
        start_of_month = (date.to_timestamp() - pd.DateOffset(months=1)).date()
        end_of_month = date.to_timestamp().date()
        returns_last_month = self.benchmark_returns.loc[start_of_month:end_of_month]

        if returns_last_month.empty:
            raise ValueError(
                f"No benchmark data available from {start_of_month} to {end_of_month}"
            )

        monthly_volatility = returns_last_month.std() * np.sqrt(monthly_trading_days)
        return monthly_volatility

    def _print_category_allocation(
            self, weights: pd.DataFrame, aggressive_alloc: float, defensive_alloc: float, date: pd.Period
    ):
        aggressive_tickers_in_weights = list(set(self.aggressive_tickers) & set(weights.columns))
        defensive_tickers_in_weights = list(set(self.defensive_tickers) & set(weights.columns))

        aggressive_weights = weights[aggressive_tickers_in_weights].sum(axis=1).iloc[
            0] if aggressive_tickers_in_weights else 0.0
        defensive_weights = weights[defensive_tickers_in_weights].sum(axis=1).iloc[
            0] if defensive_tickers_in_weights else 0.0

        print(f"\nAllocation on {date}:")
        print(f"Expected Aggressive Allocation: {aggressive_alloc * 100:.2f}%")
        print(f"Actual Aggressive Allocation: {aggressive_weights * 100:.2f}%")
        print(f"Expected Defensive Allocation: {defensive_alloc * 100:.2f}%")
        print(f"Actual Defensive Allocation: {defensive_weights * 100:.2f}%\n")

    def optimize_portfolios(self) -> pd.DataFrame:
        weights_list = []
        last_weights = None

        for date in self.rebalancing_dates:
            defensive_alloc = self.defensive_allocation.loc[date].values[0]
            aggressive_alloc = 1 - defensive_alloc

            long_tickers = self.long_signals.loc[date]
            long_tickers = long_tickers[long_tickers].index.tolist()

            if not long_tickers:
                print(f"No long signals on {date}. Skipping optimization.")
                continue

            returns_window = self._get_returns_window(date, long_tickers)
            if returns_window.empty or returns_window.shape[0] < 2:
                print(f"Not enough data for optimization on {date}. Skipping.")
                continue

            benchmark_vol = self._calculate_benchmark_monthly_volatility(date)
            asset_classes = self._get_asset_classes(long_tickers)
            constraints = self._get_constraints(aggressive_alloc, defensive_alloc)

            w = self._optimize_portfolio(
                date, returns_window, last_weights, benchmark_vol, asset_classes, constraints
            )

            if w is not None:
                self._print_category_allocation(w, aggressive_alloc, defensive_alloc, date)
                weights_list.append(w)
                last_weights = w
            else:
                print(f"Optimization failed on {date}. Using last weights.")
                if last_weights is not None:
                    w = last_weights.copy()
                    w.index = [date]
                    weights_list.append(w)

        self.weights = pd.concat(weights_list).fillna(0) if weights_list else pd.DataFrame()
        return self.weights

    def _get_returns_window(self, date: pd.Period, tickers: list) -> pd.DataFrame:
        window = self.parameters.get('window', 252)
        end_date = date.to_timestamp()
        start_date = end_date - pd.Timedelta(days=window)
        returns_window = self.returns.loc[start_date:end_date, tickers].dropna()
        return returns_window

    def _get_asset_classes(self, tickers: list) -> pd.DataFrame:
        sector_mapping = {
            ticker: 'Aggressive' if ticker in self.aggressive_tickers else
            'Defensive' if ticker in self.defensive_tickers else 'Other'
            for ticker in tickers
        }
        asset_classes = pd.DataFrame({
            'Assets': tickers,
            'Sector': [sector_mapping[ticker] for ticker in tickers],
        })
        return asset_classes

    def _get_constraints(self, aggressive_alloc: float, defensive_alloc: float) -> pd.DataFrame:
        constraints = pd.DataFrame({
            'Disabled': [False, False, False, False],
            'Type': ['All Assets', 'All Assets', 'Classes', 'Classes'],
            'Set': ['', '', 'Sector', 'Sector'],
            'Position': ['', '', 'Aggressive', 'Defensive'],
            'Sign': ['<=', '>=', '<=', '<='],
            'Weight': [
                self.parameters['max_weight_long'],
                self.parameters['min_weight_long'],
                aggressive_alloc,
                defensive_alloc,
            ],
            'Type Relative': ['', '', '', ''],
            'Relative Set': ['', '', '', ''],
            'Relative': ['', '', '', ''],
            'Factor': ['', '', '', ''],
        })
        return constraints

    def _optimize_portfolio(
            self,
            date: pd.Period,
            returns_window: pd.DataFrame,
            last_weights: pd.DataFrame,
            benchmark_vol: float,
            asset_classes: pd.DataFrame,
            constraints: pd.DataFrame,
    ) -> pd.DataFrame:
        port = rp.Portfolio(returns=returns_window)
        port.assets_stats(
            method_mu=self.parameters['method_mu'],
            method_cov=self.parameters['method_cov'],
        )

        port.sht = False  # No short positions
        port.allowTO = True
        port.turnover = self.parameters['max_turnover']
        port.budget = self.parameters['budget']
        port.upperdev = benchmark_vol
        port.upperlng = 1.0  # Sum of long positions

        if last_weights is not None:
            bench_weights = last_weights.copy()
            bench_weights = bench_weights.iloc[0]
            bench_weights = bench_weights.to_frame(name='weight')
            port.benchweights = bench_weights
        else:
            port.benchweights = None

        A, B = rp.assets_constraints(constraints, asset_classes)
        port.ainequality = A
        port.binequality = B
        port.solvers = ['CLARABEL']

        w = port.optimization(
            model=self.parameters['model'],
            rm=self.parameters['rm'],
            obj=self.parameters['obj'],
            rf=self.parameters['rf'],
            l=self.parameters['l'],
            hist=self.parameters['hist'],
        )

        if w is None:
            print(f"Optimization returned None on {date}")
            return None

        w = w.T
        w.index = [date]

        w = self._normalize_weights(w)
        return w

    def _normalize_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        total_weight = weights.sum(axis=1).iloc[0]
        if abs(total_weight - self.parameters['budget']) > 1e-5:
            weights *= self.parameters['budget'] / total_weight
        return weights

    def calculate_turnover(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the portfolio turnover between consecutive months.

        Parameters:
            weights (pd.DataFrame): Optimized weights where rows are dates and columns are assets.

        Returns:
            pd.DataFrame: DataFrame with turnover values per date.
        """
        turnover_list = [0.0]  # First month turnover is 0
        previous_weights = weights.iloc[0]

        for i in range(1, len(weights)):
            current_weights = weights.iloc[i]
            number_of_positive_weights = np.sum(current_weights > 0)
            turnover = np.sum(np.abs(current_weights - previous_weights))
            relative_turnover = turnover / (self.parameters['max_turnover'] * number_of_positive_weights)
            turnover_list.append(relative_turnover)
            previous_weights = current_weights

        turnover_df = pd.DataFrame(turnover_list, index=weights.index, columns=["Turnover (%)"])
        return turnover_df


if __name__ == '__main__':
    returns = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/compute_returns/daily_returns.csv',
        index_col=0, parse_dates=True
    )
    long_signals = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/create_long_short_portfolio/long_signals.csv',
        index_col=0, parse_dates=True
    )

    bear_allocation = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/regime_change/short_allocation.csv',
        index_col=0, parse_dates=True
    )

    us_stocks_sectors = pd.read_csv(
        filepath_or_buffer='../../data/raw_data/us_stocks_sectors.csv',
    )

    benchmark_prices = pd.read_csv(
        filepath_or_buffer='../../data/raw_data/SP500_daily.csv',
        index_col=0, parse_dates=True
    )

    # Initialize the portfolio optimizer
    optimizer = WeightingStrategyWithDynamicRegime(
        returns=returns, long_signals=long_signals, benchmark_prices=benchmark_prices, defensive_allocation=bear_allocation,
        us_stocks_sectors=us_stocks_sectors
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
        window=1000,
        budget=1.0,
        max_weight_long=0.1,
        min_weight_long=0.001,
        weight_threshold=1e-8
    )

    # Run the optimization
    weights_df = optimizer.optimize_portfolios()

    # Calculate the turnover
    turnover_df = optimizer.calculate_turnover(weights_df)

    # Display the weights and turnover
    print("\nPortfolio Weights:")
    print(weights_df.head())
    print("\nTurnover per date:")
    print(turnover_df)
