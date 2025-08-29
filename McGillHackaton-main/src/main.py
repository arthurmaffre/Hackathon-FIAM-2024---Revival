import pandas as pd
from abc import ABC, abstractmethod
import pickle
from typing import Any

from src.read_config.config_reader import get_merged_config
from src.preprocess_data.preprocess_data import PreprocessData
from src.option_features.compute_options_features import OptionFeaturesCalculator
from src.option_features.option_preprocessor import OptionDataPreprocessor
from src.add_more_variables.add_more_variables import AddMoreVariables
from src.compute_returns.from_prices_to_returns import FromPricesToReturns
from src.prediction_pipeline.prepare_data_for_prediction import PrepareDataForPrediction
from src.prediction_pipeline.prediction_pipeline_skforecast import PredictionPipelineSkforecast, PredictionPipelineSkforecastNoVal
from skforecast.utils import save_forecaster, load_forecaster
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from src.evaluate_model_performance.evaluate_model_performance import EvaluateModelPerformance
from src.model_explainability.model_explainability import ModelExplainability
from src.create_long_short_portfolio.create_long_short_portfolio import CreateLongShortPortfolio
from src.regime_detection.regime_model_features import BuildFeaturesForSJM
from src.regime_detection.longshort_allocation import RegimeBasedPortfolio
from src.weighting.dynamic_sectors_weighting import WeightingStrategyWithDynamicRegime
from src.drifted_weights.compute_drifted_weights import DriftedWeightsCalculator
from src.strategy_returns.compute_strategy_returns import StrategyReturnsCalculator
from src.analyze_portfolio_returns.analyze_portfolio_returns import StrategyPerformanceAnalyzer

import warnings

warnings.filterwarnings('once')
warnings.filterwarnings('ignore')


class ReturnsPredictionPipelineAbstract(ABC):

    prediction_pipeline_config_path = '../config/meta_config/prediction_pipeline.yaml'
    strategy_config_path = '../config/strategy_config/'

    def __init__(self, config: dict, strategy_name: str = None):
        self.config = config
        self.strategy_name = strategy_name

    @abstractmethod
    def load_data(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def preprocess_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_more_variables(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def compute_returns(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def prepare_data_for_prediction(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def prediction_pipeline(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries]:
        pass

    @abstractmethod
    def evaluate_model_performance(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def model_explainability(self, **kwargs) -> None:
        pass

    @abstractmethod
    def create_long_short_portfolio(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def compute_bear_regime_probabilities(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def weighting(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def compute_drifted_weights(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def compute_strategy_returns(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def analyze_strategy_returns(self, **kwargs) -> None:
        pass


    @abstractmethod
    def main(self):
        pass

    @staticmethod
    def print_separator(message: str) -> None:
        """
        Prints a stylized separator with the given message centered.

        Args:
            message (str): The message to be displayed within the separator.
        """
        separator_line = "═" * 80
        total_width = 80
        padding_each_side = (total_width - len(message) - 2) // 2  # calculate padding for each side
        extra_padding = (total_width - len(message) - 2) % 2  # for odd-length messages

        # Create the title lines with borders
        title_line_1 = "║" + " " * (total_width - 2) + "║"
        title_line_2 = (
                "║"
                + " " * padding_each_side
                + message
                + " " * (padding_each_side + extra_padding)
                + "║"
        )
        title_line_3 = title_line_1

        # Print the separator with title lines
        print(f"\n{separator_line}\n{title_line_1}\n{title_line_2}\n{title_line_3}\n{separator_line}\n")


class ReturnsPredictionPipeline(ReturnsPredictionPipelineAbstract):


    def load_data(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        load_data_dict = self.config['load_data']

        hackathon_df = pd.read_csv(filepath_or_buffer=load_data_dict['hackathon_csv_path'], index_col=0, parse_dates=True)
        stocks_prices_df = pd.read_parquet(path=load_data_dict['stocks_prices_parquet_path'])
        mapping_stocks_country_df = pd.read_excel(io=load_data_dict['mapping_stocks_country_excel_path'])[
            ['Tickers', 'Name', 'Sector', 'Country']]
        sp500_df = pd.read_csv(filepath_or_buffer=load_data_dict['sp500_csv_path'], index_col=0, parse_dates=True)
        vix_df = pd.read_csv(filepath_or_buffer=load_data_dict['vix_csv_path'], index_col=0, parse_dates=True)
        macro_data = pd.read_csv(
            filepath_or_buffer=load_data_dict['macro_data_csv_path'],
            index_col=0,
            parse_dates=True,
            sep=','
        )

        us_stocks_sectors = pd.read_csv(filepath_or_buffer=load_data_dict['us_stocks_sectors_csv_path'])

        self.print_separator("Data loaded successfully")

        return hackathon_df, stocks_prices_df, mapping_stocks_country_df, sp500_df, vix_df, macro_data, us_stocks_sectors

    def preprocess_data(self, hackathon_df: pd.DataFrame,
                        stocks_prices_df: pd.DataFrame, mapping_stocks_country_df: pd.DataFrame, macro_data: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        preprocess_data_dict = self.config['preprocess_data']

        preprocess_data = PreprocessData()
        hackathon_df, stocks_prices_df, mapping_stocks_country_df, macro_data_scaled = preprocess_data.preprocess_data(
            hackathon_df=hackathon_df, stocks_prices_df=stocks_prices_df,
            mapping_stocks_country_df=mapping_stocks_country_df, macro_data=macro_data,
            ticker_column=preprocess_data_dict['col_to_count'],
        )

        if preprocess_data_dict['save_data']:
            hackathon_df.to_csv(
                path_or_buf=f"{preprocess_data_dict['preprocess_data_path']}hackathon_df_preprocessed.csv"
            )
            stocks_prices_df.to_csv(
                path_or_buf=f"{preprocess_data_dict['preprocess_data_path']}stocks_prices_df_preprocessed.csv"
            )
            mapping_stocks_country_df.to_csv(
                path_or_buf=f"{preprocess_data_dict['preprocess_data_path']}mapping_stocks_country_df_preprocessed.csv"
            )
            macro_data_scaled.to_csv(
                path_or_buf=f"{preprocess_data_dict['preprocess_data_path']}macro_data_scaled.csv"
            )

        self.print_separator("Data preprocessed successfully")

        return hackathon_df, stocks_prices_df, mapping_stocks_country_df, macro_data_scaled


    def add_more_variables(self, hackathon_df_preprocessed: pd.DataFrame, macro_data_scaled: pd.DataFrame) -> pd.DataFrame:
        add_more_variables_dict = self.config['add_more_variables']

        add_more_variables = AddMoreVariables(hackathon_df=hackathon_df_preprocessed)

        if add_more_variables_dict['use_existing_options_indicators']:
            options_indicators = pd.read_csv(add_more_variables_dict['options_indicators_path'])
        else:
            options_before_2010 = pd.read_parquet(add_more_variables_dict['options_before_2010_path'])
            options_after_2010 = pd.read_parquet(add_more_variables_dict['options_after_2010_path'])
            hackathon_df_preprocessed = pd.read_csv(
                filepath_or_buffer=add_more_variables_dict['hackathon_df_preprocessed_path'],
                index_col=0, parse_dates=True)

            # Preprocess data
            preprocessor = OptionDataPreprocessor(
                options_before_2010=options_before_2010,
                options_after_2010=options_after_2010,
                hackathon_df_preprocessed=hackathon_df_preprocessed
            )

            preprocessor.create_cusip_to_ticker_mapping()
            preprocessor.convert_date_column_to_datetime()
            preprocessor.map_cusip_to_ticker()
            preprocessor.calculate_mid_price()
            preprocessor.calculate_ttm()
            preprocessor.calculate_abs_delta()

            print(preprocessor.options_before_2010.head())
            print(preprocessor.options_after_2010.head())

            # Concatenate pre and post 2010 data after preprocessing
            options_data = pd.merge(preprocessor.options_before_2010, preprocessor.options_after_2010, how='outer')

            print(options_data.head())
            print(options_data.tail())

            # Calcul des indicateurs
            feature_calculator = OptionFeaturesCalculator(options_data)
            monthly_iv = feature_calculator.calculate_monthly_avg_impl_volatility()
            monthly_skewness = feature_calculator.calculate_monthly_avg_skewness()
            monthly_baspread = feature_calculator.calculate_monthly_avg_opt_baspread()

            # Fusionner les indicateurs en utilisant pd.merge
            options_indicators = monthly_iv.merge(monthly_skewness, on=['stock_ticker', 'year_month'], how='inner')
            options_indicators = options_indicators.merge(monthly_baspread, on=['stock_ticker', 'year_month'], how='inner')


        hackathon_df_with_more_variables = add_more_variables.add_macro_data(macro_data=macro_data_scaled)
        hackathon_df_with_more_variables = add_more_variables.add_option_indicators(
            option_indicators=options_indicators, hackathon_df=hackathon_df_with_more_variables
        )

        if add_more_variables_dict['save_data']:
            hackathon_df_with_more_variables.to_csv(
                path_or_buf=f"{add_more_variables_dict['add_more_variables_path']}hackathon_df_with_more_variables.csv"
            )

        self.print_separator("Additional variables added successfully")

        return hackathon_df_with_more_variables


    def compute_returns(self, stocks_prices_df: pd.DataFrame) -> pd.DataFrame:
        compute_returns_dict = self.config['compute_returns']

        from_prices_to_returns = FromPricesToReturns()
        returns_df = from_prices_to_returns.compute_returns(
            data=stocks_prices_df,
            return_type=compute_returns_dict['return_type'],
            binarize=compute_returns_dict['binarize']
        )

        if compute_returns_dict['save_data']:
            returns_df.to_csv(
                path_or_buf=f"{compute_returns_dict['compute_returns_path']}daily_returns.csv"
            )

        self.print_separator("Returns computed successfully")

        return returns_df

    def prepare_data_for_prediction(self, hackathon_df_with_more_variables: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        prepare_data_for_prediction_dict = self.config['prepare_data_for_prediction']

        prepare_data_for_prediction = PrepareDataForPrediction()

        X, y = prepare_data_for_prediction.prepare_data_for_prediction(
            df=hackathon_df_with_more_variables,
            shift_dict=prepare_data_for_prediction_dict['shift_dict'],
            target_col=prepare_data_for_prediction_dict['target_col'],
            drop_cols=prepare_data_for_prediction_dict['drop_cols'],
            do_forward_filling=prepare_data_for_prediction_dict['do_forward_filling'],
        )

        if prepare_data_for_prediction_dict['save_data']:
            X.to_csv(
                path_or_buf=f"{prepare_data_for_prediction_dict['prepare_data_for_prediction_path']}exogenous_data.csv"
            )
            y.to_csv(
                path_or_buf=f"{prepare_data_for_prediction_dict['prepare_data_for_prediction_path']}endogenous_data.csv"
            )

        self.print_separator("Data prepared for prediction successfully")

        return X, y

    def prediction_pipeline(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, ForecasterAutoregMultiSeries | None]:
        prediction_pipeline_dict = self.config['prediction_pipeline']

        prediction_pipeline = PredictionPipelineSkforecast(
            endogenous_data=endogenous_data,
            exogenous_data=exogenous_data,
        )

        if prediction_pipeline_dict['use_existing_predictions']:
            # Load y_pred from CSV
            y_pred_df = pd.read_csv(
                filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_pred_{self.strategy_name}.csv",
                parse_dates=True, index_col=0
            )

            # Load y_test, y_val, and y_train from CSV files and convert them back to DataFrames
            y_test_df = pd.read_csv(
                filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_test_{self.strategy_name}.csv",
                parse_dates=True, index_col=0
            )
            y_val_df = pd.read_csv(
                filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_val_{self.strategy_name}.csv",
                parse_dates=True, index_col=0
            )
            y_train_df = pd.read_csv(
                filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.csv",
                parse_dates=True, index_col=0
            )
            # Load y_train and X_train from pickle files as dictionaries
            with open(f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.pkl",
                      'rb') as f:
                y_train_dict = pickle.load(f)

            with open(f"{prediction_pipeline_dict['prediction_pipeline_path']}X_train_{self.strategy_name}.pkl",
                      'rb') as f:
                X_train_dict = pickle.load(f)

            # Load the forecaster model
            forecaster = load_forecaster(
                file_name=f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
                verbose=True
            )

            y_pred_df.index = pd.to_datetime(y_pred_df.index)
            y_test_df.index = pd.to_datetime(y_test_df.index)
            y_val_df.index = pd.to_datetime(y_val_df.index)
            y_train_df.index = pd.to_datetime(y_train_df.index)

        else:
            # Run the prediction pipeline and obtain predictions
            metrics_level, y_pred_df = prediction_pipeline.backtest_model_with_tuning(
                regressor=prediction_pipeline_dict['estimator_name'],
                regressor_dict=prediction_pipeline_dict['regressor_dict'],
                lags=prediction_pipeline_dict['lags'],
                lags_grid=prediction_pipeline_dict['lags_grid'],
                end_train=prediction_pipeline_dict['end_train'],
                end_val=prediction_pipeline_dict['end_val'],
                grid_search_params=prediction_pipeline_dict['grid_search_params'],
                grid_search_params_grid=prediction_pipeline_dict['grid_search_params_grid'],
                bayes_search_params=prediction_pipeline_dict['bayes_search_params'],
                bayes_search_params_grid=prediction_pipeline_dict['bayes_search_params_grid'],
                backtest_params=prediction_pipeline_dict['backtest_params'],
                forecaster_params=prediction_pipeline_dict['forecaster_params'],
                tuning_method=prediction_pipeline_dict['tuning_method'],
                save_path=prediction_pipeline_dict['prediction_pipeline_path']
            )

            forecaster = prediction_pipeline.forecaster
            y_train_dict = prediction_pipeline.endogenous_train
            y_val_dict = prediction_pipeline.endogenous_val
            y_test_dict = prediction_pipeline.endogenous_test
            X_train_dict = prediction_pipeline.exogenous_train

            # Convert dictionaries to DataFrames
            y_train_df = prediction_pipeline.convert_exogenous_dict_to_panel(exogenous_dict=y_train_dict)
            y_val_df = prediction_pipeline.convert_exogenous_dict_to_panel(exogenous_dict=y_val_dict)
            y_test_df = prediction_pipeline.convert_exogenous_dict_to_panel(exogenous_dict=y_test_dict)

        if prediction_pipeline_dict['save_data']:
            # Save the forecaster model
            save_forecaster(
                forecaster=forecaster,
                file_name=f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
            )
            # Save y_pred as a CSV
            y_pred_df.to_csv(
                path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_pred_{self.strategy_name}.csv"
            )
            # Save y_train and X_train as dictionaries using pickle
            with open(f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.pkl",
                      'wb') as f:
                pickle.dump(y_train_dict, f)

            with open(f"{prediction_pipeline_dict['prediction_pipeline_path']}X_train_{self.strategy_name}.pkl",
                      'wb') as f:
                pickle.dump(X_train_dict, f)

            # Save y_test, y_val, and y_train as CSVs
            y_test_df.to_csv(
                path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_test_{self.strategy_name}.csv"
            )
            y_val_df.to_csv(
                path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_val_{self.strategy_name}.csv"
            )
            y_train_df.to_csv(
                path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.csv"
            )

        self.print_separator("Predictions made successfully")

        print(f"y_pred: {y_pred_df}")

        return y_pred_df, y_test_df, y_train_df, y_train_dict, X_train_dict, forecaster

    # def prediction_pipeline(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame) \
    #         -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries | None]:
    #     """
    #     A modified version of the prediction pipeline function that does not use a validation set,
    #     following the 'PredictionPipelineSkforecastNoVal' class for data handling.
    #
    #     Parameters:
    #     ----------
    #     endogenous_data : pd.DataFrame
    #         The endogenous (target) data for the pipeline.
    #     exogenous_data : pd.DataFrame
    #         The exogenous data (features) for the pipeline.
    #
    #     Returns:
    #     -------
    #     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ForecasterAutoregMultiSeries | None]
    #         A tuple containing the predicted values (y_pred_df), the test set (y_test_df),
    #         the training set (y_train_df), and the trained forecaster.
    #     """
    #     prediction_pipeline_dict = self.config['prediction_pipeline']
    #
    #     prediction_pipeline = PredictionPipelineSkforecastNoVal(
    #         endogenous_data=endogenous_data,
    #         exogenous_data=exogenous_data,
    #     )
    #
    #     if prediction_pipeline_dict['use_existing_predictions']:
    #         # Load y_pred from CSV
    #         y_pred_df = pd.read_csv(
    #             filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_pred_{self.strategy_name}.csv",
    #             index_col=0,  # Ensure the index is loaded correctly
    #             parse_dates=True
    #         )
    #
    #         # Load y_test and y_train from CSV files and convert them back to DataFrames
    #         y_test_df = pd.read_csv(
    #             filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_test_{self.strategy_name}.csv",
    #             parse_dates=True
    #         )
    #         y_train_df = pd.read_csv(
    #             filepath_or_buffer=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.csv",
    #             parse_dates=True
    #         )
    #
    #         # Load the forecaster model
    #         forecaster = load_forecaster(
    #             file_name=f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
    #             verbose=True
    #         )
    #
    #         y_pred_df.index = pd.to_datetime(y_pred_df.index)
    #         y_test_df.index = pd.to_datetime(y_test_df.index)
    #         y_train_df.index = pd.to_datetime(y_train_df.index)
    #
    #     else:
    #         # Run the prediction pipeline and obtain predictions
    #         metrics_level, y_pred_df = prediction_pipeline.backtest_model_with_tuning(
    #             regressor=prediction_pipeline_dict['estimator_name'],
    #             regressor_dict=prediction_pipeline_dict['regressor_dict'],
    #             lags=prediction_pipeline_dict['lags'],
    #             lags_grid=prediction_pipeline_dict['lags_grid'],
    #             end_train=prediction_pipeline_dict['end_train'],
    #             grid_search_params=prediction_pipeline_dict['grid_search_params'],
    #             grid_search_params_grid=prediction_pipeline_dict['grid_search_params_grid'],
    #             bayes_search_params=prediction_pipeline_dict['bayes_search_params'],
    #             bayes_search_params_grid=prediction_pipeline_dict['bayes_search_params_grid'],
    #             backtest_params=prediction_pipeline_dict['backtest_params'],
    #             forecaster_params=prediction_pipeline_dict['forecaster_params'],
    #             tuning_method=prediction_pipeline_dict['tuning_method'],
    #             save_path=prediction_pipeline_dict['prediction_pipeline_path']
    #         )
    #
    #         forecaster = prediction_pipeline.forecaster
    #         y_train = prediction_pipeline.endogenous_train
    #         y_test = prediction_pipeline.endogenous_test
    #
    #         # Convert dictionaries to DataFrames
    #         y_train_df = prediction_pipeline.convert_dict_to_panel(data_dict=y_train)
    #         y_test_df = prediction_pipeline.convert_dict_to_panel(data_dict=y_test)
    #
    #     if prediction_pipeline_dict['save_data']:
    #         # Save the forecaster model
    #         save_forecaster(
    #             forecaster=forecaster,
    #             file_name=f"{prediction_pipeline_dict['prediction_pipeline_path']}{self.strategy_name}_forecaster.pkl",
    #         )
    #         # Save y_pred as a CSV
    #         y_pred_df.to_csv(
    #             path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_pred_{self.strategy_name}.csv"
    #         )
    #         # Save y_test and y_train as CSVs
    #         y_test_df.to_csv(
    #             path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_test_{self.strategy_name}.csv"
    #         )
    #         y_train_df.to_csv(
    #             path_or_buf=f"{prediction_pipeline_dict['prediction_pipeline_path']}y_train_{self.strategy_name}.csv"
    #         )
    #
    #     self.print_separator("Predictions made successfully")
    #
    #     print(f"y_pred: {y_pred_df}")
    #
    #     return y_pred_df, y_test_df, y_train_df, forecaster

    def evaluate_model_performance(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, y_train: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        evaluate_model_performance_dict = self.config['evaluate_model_performance']

        evaluate_model_performance = EvaluateModelPerformance(
            y_pred=y_pred,
            y_test=y_test,
            y_train=y_train,
            metrics=evaluate_model_performance_dict['metrics']
        )

        evaluation_results = evaluate_model_performance.evaluate()
        aggregated_metrics = evaluate_model_performance.aggregate_metrics()

        print(f"evaluation_results: {evaluation_results}")
        print(f"aggregated_metrics: {aggregated_metrics}")

        if evaluate_model_performance_dict['save_data']:
            evaluation_results.to_csv(
                path_or_buf=
                f"{evaluate_model_performance_dict['evaluate_model_performance_path']}evaluation_results_{self.strategy_name}.csv"
            )
            aggregated_metrics.to_csv(
                path_or_buf=
                f"{evaluate_model_performance_dict['evaluate_model_performance_path']}aggregated_metrics_{self.strategy_name}.csv"
            )
            evaluate_model_performance.y_test.to_csv(
                path_or_buf=
                f"{evaluate_model_performance_dict['evaluate_model_performance_path']}y_test_tabular_{self.strategy_name}.csv"
            )
            evaluate_model_performance.y_pred.to_csv(
                path_or_buf=
                f"{evaluate_model_performance_dict['evaluate_model_performance_path']}y_pred_tabular_{self.strategy_name}.csv"
            )

        self.print_separator("Model performance evaluated successfully")

        return evaluate_model_performance.y_pred, evaluation_results

    def model_explainability(self, y_train_dict: dict, X_train_dict: dict, forecaster: ForecasterAutoregMultiSeries | Any) -> None:
        model_explainability_dict = self.config['model_explainability']

        model_explainability = ModelExplainability(
            y_train_dict=y_train_dict,
            X_train_dict=X_train_dict,
            forecaster=forecaster
        )

        model_explainability.fit_model(forecaster=forecaster)
        model_explainability.compute_shap_values()

        model_explainability.plot_summary_shap(
            max_display=model_explainability_dict['max_display'],
            save_path=f"{model_explainability_dict['save_path']}summary_plot_{self.strategy_name}.png"
        )

        model_explainability.plot_bar_summary_shap(
            max_display=model_explainability_dict['max_display'],
            save_path=f"{model_explainability_dict['save_path']}summary_bar_plot_{self.strategy_name}.png"
        )

        print("Model explainability computed successfully")

        return None



    def create_long_short_portfolio(self, y_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        create_long_short_portfolio_dict = self.config['create_long_short_portfolio']

        create_long_short_portfolio = CreateLongShortPortfolio(signals=y_pred)

        long_signals, short_signals = create_long_short_portfolio.create_long_short_portfolio(
            ranking_strategy=create_long_short_portfolio_dict['ranking_strategy'],
            ascending=create_long_short_portfolio_dict['ascending'],
            method=create_long_short_portfolio_dict['method'],
            fix_threshold=create_long_short_portfolio_dict['fix_threshold']
        )

        if create_long_short_portfolio_dict['save_data']:
            long_signals.to_csv(
                path_or_buf=f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}long_signals_{self.strategy_name}.csv"
            )
            short_signals.to_csv(
                path_or_buf=f"{create_long_short_portfolio_dict['create_long_short_portfolio_path']}short_signals_{self.strategy_name}.csv"
            )

        return long_signals, short_signals

    def compute_bear_regime_probabilities(self, sp500_df: pd.DataFrame, vix_df: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        compute_bear_regime_probabilities_dict = self.config['compute_bear_regime_probabilities']

        regime_model_features = BuildFeaturesForSJM(
            spx_data=sp500_df,
            vix_data=vix_df,
            start_date=compute_bear_regime_probabilities_dict['start_date']
        )

        standardized_features = regime_model_features.get_standardized_features(
            rolling_windows=compute_bear_regime_probabilities_dict['rolling_windows']
        )

        regime_based_portfolio = RegimeBasedPortfolio(
            data=standardized_features,
            start_year=y_pred.index.year.min(),
            end_year=y_pred.index.year.max(),
            beta_bull=compute_bear_regime_probabilities_dict['beta_bull'],
            beta_bear=compute_bear_regime_probabilities_dict['beta_bear'],
            jump_penalty=compute_bear_regime_probabilities_dict['jump_penalty'],
        )
        regime_probabilities = regime_based_portfolio.calculate_period_regime_probabilities()
        defensive_allocation = regime_based_portfolio.calculate_short_allocation()

        regime_based_portfolio.plot_regime_probabilities(spx_data=sp500_df)

        if compute_bear_regime_probabilities_dict['save_data']:
            defensive_allocation.to_csv(
                path_or_buf=f"{compute_bear_regime_probabilities_dict['compute_bear_regime_probabilities_path']}short_allocation_{self.strategy_name}.csv"
            )

        return pd.DataFrame(defensive_allocation)


    def weighting(self, daily_returns_df: pd.DataFrame, long_signals: pd.DataFrame, benchmark_prices: pd.DataFrame,
        defensive_allocation: pd.DataFrame, us_stocks_sectors: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        weighting_dict = self.config['weighting']

        weighting_strategy = WeightingStrategyWithDynamicRegime(
            returns=daily_returns_df,
            long_signals=long_signals,
            benchmark_prices=benchmark_prices,
            defensive_allocation=defensive_allocation,
            us_stocks_sectors=us_stocks_sectors
        )

        weighting_strategy.set_parameters(
            method_mu=weighting_dict['method_mu'],
            method_cov=weighting_dict['method_cov'],
            model=weighting_dict['model'],
            rm=weighting_dict['rm'],
            obj=weighting_dict['obj'],
            rf=weighting_dict['rf'],
            l=weighting_dict['l'],
            hist=weighting_dict['hist'],
            window=weighting_dict['window'],
            budget=weighting_dict['budget'],
            max_weight_long=weighting_dict['max_weight_long'],
            min_weight_long=weighting_dict['min_weight_long'],
            max_turnover=weighting_dict['max_turnover'],
        )

        long_weights = weighting_strategy.optimize_portfolios()
        turnover_df = weighting_strategy.calculate_turnover(weights=long_weights)
        mean_turnover = turnover_df.mean().iloc[0]

        if weighting_dict['save_data']:
            long_weights.to_csv(
                path_or_buf=f"{weighting_dict['weighting_path']}long_short_weights_{self.strategy_name}.csv"
            )
            turnover_df.to_csv(
                path_or_buf=f"{weighting_dict['weighting_path']}turnover_{self.strategy_name}.csv"
            )

        print(long_weights)
        print(turnover_df)

        self.print_separator("Weights computed successfully")

        return long_weights, mean_turnover

    def compute_drifted_weights(self, long_short_weights: pd.DataFrame, stocks_prices: pd.DataFrame) \
            -> pd.DataFrame:
        compute_drifted_weights_dict = self.config['compute_drifted_weights']

        strategy_returns_calculator = DriftedWeightsCalculator(
            daily_prices=stocks_prices,
            long_short_weights=long_short_weights
        )

        drifted_weights = strategy_returns_calculator.calculate_drifted_weights()

        # turnover = strategy_returns_calculator.calculate_turnover_monthly()
        # turnover_float = strategy_returns_calculator.calculate_turnover()

        if compute_drifted_weights_dict['save_data']:
            drifted_weights.to_csv(
                path_or_buf=
                f"{compute_drifted_weights_dict['compute_drifted_weights_path']}drifted_weights_{self.strategy_name}.csv"
            )
            # turnover.to_csv(
            #     path_or_buf=
            #     f"{compute_drifted_weights_dict['compute_drifted_weights_path']}turnover_{self.strategy_name}.csv"
            # )

        print(drifted_weights)

        self.print_separator("Drifted weights computed successfully")

        return drifted_weights

    def compute_strategy_returns(self, drifted_weights: pd.DataFrame, stocks_prices: pd.DataFrame) -> pd.DataFrame:
        compute_strategy_returns_dict = self.config['compute_strategy_returns']

        strategy_returns_calculator = StrategyReturnsCalculator(
            drifted_weights=drifted_weights,
            daily_prices=stocks_prices
        )

        strategy_returns = strategy_returns_calculator.calculate_realized_returns()

        if compute_strategy_returns_dict['save_data']:
            strategy_returns.to_csv(
                path_or_buf=f"{compute_strategy_returns_dict['compute_strategy_returns_path']}strategy_returns_{self.strategy_name}.csv"
            )

        return strategy_returns

    def analyze_strategy_returns(self, strategy_returns: pd.DataFrame, benchmark_prices: pd.DataFrame, turnover_float: float) -> pd.DataFrame:
        analyze_strategy_returns_dict = self.config['analyze_strategy_returns']

        strategy_performance_analyzer = StrategyPerformanceAnalyzer(
            portfolio_returns=strategy_returns,
            benchmark_prices=benchmark_prices,
            strategy_name=analyze_strategy_returns_dict['strategy_name']
        )

        strategy_performance_analyzer.generate_backtesting_report_html(
            rf=analyze_strategy_returns_dict['rf'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
            grayscale=analyze_strategy_returns_dict['grayscale'],
            dest_folder=analyze_strategy_returns_dict['dest_folder'],
            match_dates=analyze_strategy_returns_dict['match_dates'],
        )

        portfolio_key_metrics = strategy_performance_analyzer.get_key_performance_metrics(
            rf=analyze_strategy_returns_dict['rf'],
            annualize=analyze_strategy_returns_dict['annualize'],
            periods_per_year=analyze_strategy_returns_dict['periods_per_year'],
            turnover=turnover_float
        )

        if analyze_strategy_returns_dict['save_data']:
            portfolio_key_metrics.to_csv(
                path_or_buf=f"{analyze_strategy_returns_dict['analyze_strategy_returns_path']}portfolio_key_metrics_{self.strategy_name}.csv"
            )

        self.print_separator("Strategy returns analyzed successfully")

        print(portfolio_key_metrics)

        return portfolio_key_metrics


    def main(self):
        hackathon_df, stocks_prices_df, mapping_stocks_country_df, sp500_df, vix_df, macro_data, us_stocks_sectors = self.load_data()
        hackathon_df_preprocessed, stocks_prices_df_preprocessed, mapping_stocks_country_df_preprocessed, macro_data_scaled = self.preprocess_data(
            hackathon_df=hackathon_df,
            stocks_prices_df=stocks_prices_df,
            mapping_stocks_country_df=mapping_stocks_country_df,
            macro_data=macro_data
        )
        hackathon_df_with_more_variables = self.add_more_variables(
            hackathon_df_preprocessed=hackathon_df_preprocessed,
            macro_data_scaled=macro_data_scaled
        )
        daily_returns_df = self.compute_returns(stocks_prices_df=stocks_prices_df_preprocessed)
        exogenous_data, endogenous_data = self.prepare_data_for_prediction(
            hackathon_df_with_more_variables=hackathon_df_with_more_variables)
        y_pred_df, y_test_df, y_train_df, y_train_dict, X_train_dict, forecaster = self.prediction_pipeline(
            endogenous_data=endogenous_data,
            exogenous_data=exogenous_data
        )
        y_pred_tabular, evaluation_results = self.evaluate_model_performance(y_pred=y_pred_df, y_test=y_test_df, y_train=y_train_df)
        self.model_explainability(y_train_dict=y_train_dict, X_train_dict=X_train_dict, forecaster=forecaster)
        long_signals, short_signals = self.create_long_short_portfolio(y_pred=y_pred_tabular)
        defensive_allocation = self.compute_bear_regime_probabilities(sp500_df=sp500_df, vix_df=vix_df, y_pred=y_pred_df)
        long_weights, mean_turnover = self.weighting(
            daily_returns_df=daily_returns_df,
            long_signals=long_signals,
            benchmark_prices=sp500_df,
            defensive_allocation=defensive_allocation,
            us_stocks_sectors=us_stocks_sectors
        )
        drifted_weights = self.compute_drifted_weights(
            long_short_weights=long_weights,
            stocks_prices=stocks_prices_df_preprocessed
        )
        strategy_returns = self.compute_strategy_returns(
            drifted_weights=drifted_weights,
            stocks_prices=stocks_prices_df_preprocessed
        )
        portfolio_metrics = self.analyze_strategy_returns(
            strategy_returns=strategy_returns,
            benchmark_prices=sp500_df[['Close']],
            turnover_float=mean_turnover
        )


class ARIMAPipeline(ReturnsPredictionPipeline):

    def __init__(self):
        self.strategy_name = 'arima'
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class HistGradientBoostingRegressorPipeline(ReturnsPredictionPipeline):

    def __init__(self):
        self.strategy_name = 'hist_gradient_boosting_regressor'
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)

class ExtraTreesRegressorPipeline(ReturnsPredictionPipeline):

    def __init__(self):
        self.strategy_name = 'extra_trees_regressor'
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)


class RandomForestRegressorPipeline(ReturnsPredictionPipeline):

    def __init__(self):
        self.strategy_name = 'random_forest_regressor'
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)

class LightGBMRegressorPipeline(ReturnsPredictionPipeline):

    def __init__(self):
        self.strategy_name = 'light_gbm_regressor'
        config_paths = [f'{self.prediction_pipeline_config_path}',
                        f'{self.strategy_config_path}{self.strategy_name}.yaml']
        config = get_merged_config(config_paths=config_paths, strategy_name=self.strategy_name)
        super().__init__(config=config, strategy_name=self.strategy_name)



if __name__ == '__main__':
    hist_gradient_boosting_regressor_pipeline = HistGradientBoostingRegressorPipeline()
    hist_gradient_boosting_regressor_pipeline.main()

    # arima_pipeline = ARIMAPipeline()
    # arima_pipeline.main()

    # random_forest_regressor_pipeline = RandomForestRegressorPipeline()
    # random_forest_regressor_pipeline.main()

    # extra_trees_regressor_pipeline = ExtraTreesRegressorPipeline()
    # extra_trees_regressor_pipeline.main()

    # light_gbm_regressor_pipeline = LightGBMRegressorPipeline()
    # light_gbm_regressor_pipeline.main()

