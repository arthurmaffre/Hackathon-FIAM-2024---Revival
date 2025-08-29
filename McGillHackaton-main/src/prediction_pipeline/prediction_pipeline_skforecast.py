import logging
logging.getLogger('gluonts').setLevel(logging.ERROR)

# Data manipulation
# ==============================================================================
import optuna.pruners
import pandas as pd
import os

# Modelling and Forecasting
# ==============================================================================
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries
from skforecast.model_selection_multiseries import bayesian_search_forecaster_multiseries
from skforecast.preprocessing import series_long_to_dict, exog_long_to_dict

from src.prediction_pipeline.ESTIMATORS import ESTIMATORS
from src.prediction_pipeline.METRICS import METRICS
from src.prediction_pipeline.TRANSFORMERS import TRANSFORMERS

from typing import Any, Optional, Tuple, List

# Warnings configuration
# ==============================================================================
import warnings

warnings.filterwarnings('once')
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)



class PredictionPipelineSkforecast:
    def __init__(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame):
        """
        Initialise la classe avec les données endogènes et exogènes sous forme de DataFrames.
        Les DataFrames sont ensuite convertis en dictionnaires.

        :param endogenous_data: DataFrame des séries endogènes.
        :param exogenous_data: DataFrame des séries exogènes.
        """
        self.endogenous_dict, self.exogenous_dict = self.prepare_data(endogenous_data, exogenous_data)
        self.endogenous_train, self.endogenous_val, self.endogenous_test = None, None, None
        self.exogenous_train, self.exogenous_val, self.exogenous_test = None, None, None
        self.forecaster = None
        self.initial_train_size = None  # Taille initiale de l'ensemble d'entraînement pour le backtesting

    @staticmethod
    def ensure_datetime(data: pd.DataFrame, column_name: str = None, format: Optional[str] = '%Y-%m-%d') -> pd.DataFrame:
        """
        Ensure that a specified column in the DataFrame is in a 'datetime' format.
        Converts the specified column to a 'datetime' format using the provided format string.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.
        column_name : str
            The name of the column to convert to 'datetime' format.
        format : str, optional
            The datetime format to use for conversion. Default is '%Y%m'.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the specified column converted to 'datetime' format.
        """

        if column_name is None:
            data.index = pd.to_datetime(data.index, errors='coerce', format=format)
            return data

        else:
            if column_name not in data.columns:
                raise KeyError(f"La colonne spécifiée '{column_name}' n'existe pas dans le DataFrame.")

            data[column_name] = pd.to_datetime(data[column_name], errors='coerce', format=format)
            return data

    def prepare_data(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Convertit les DataFrames endogènes et exogènes en dictionnaires pour chaque ticker.

        :param endogenous_data: DataFrame des séries endogènes.
        :param exogenous_data: DataFrame des séries exogènes.
        :return: Tuple de dictionnaires (endogenous_dict, exogenous_dict) avec un ticker comme clé et un DataFrame associé.
        """

        endogenous_data = self.ensure_datetime(data=endogenous_data, column_name='date')
        exogenous_data = self.ensure_datetime(data=exogenous_data, column_name='date')

        # Convertir les DataFrames en dictionnaires
        endogenous_dict = series_long_to_dict(
            data=endogenous_data,
            series_id='stock_ticker',
            index='date',
            values='stock_exret',
            freq='ME'
        )
        exogenous_dict = exog_long_to_dict(
            data=exogenous_data,
            series_id='stock_ticker',
            index='date',
            freq='ME'
        )

        return endogenous_dict, exogenous_dict

    def split_data(self, end_train: str, end_val: str) -> None:
        """
        Divise les données endogènes et exogènes en ensembles d'entraînement, de validation et de test.

        :param end_train: Date de fin de l'ensemble d'entraînement.
        :param end_val: Date de fin de l'ensemble de validation.
        """
        self.end_train = pd.to_datetime(end_train)
        self.end_val = pd.to_datetime(end_val)

        # Split des données endogènes
        self.endogenous_train = {k: v.loc[v.index <= self.end_train] for k, v in self.endogenous_dict.items()}
        self.endogenous_val = {k: v.loc[(v.index > self.end_train) & (v.index <= self.end_val)] for k, v in self.endogenous_dict.items()}
        self.endogenous_test = {k: v.loc[v.index > self.end_val] for k, v in self.endogenous_dict.items()}

        # Split des données exogènes
        self.exogenous_train = {k: v.loc[v.index <= self.end_train] for k, v in self.exogenous_dict.items()}
        self.exogenous_val = {k: v.loc[(v.index > self.end_train) & (v.index <= self.end_val)] for k, v in self.exogenous_dict.items()}
        self.exogenous_test = {k: v.loc[v.index > self.end_val] for k, v in self.exogenous_dict.items()}

        # Calcul de la taille initiale d'entraînement pour le backtesting
        first_ticker = list(self.endogenous_train.keys())[0]
        self.initial_train_size = len(self.endogenous_train[first_ticker]) + len(self.endogenous_val[first_ticker])

        return None

    @staticmethod
    def convert_endogenous_dict_to_panel(endogenous_dict: dict) -> pd.DataFrame:
        """
        Converts a dictionary of DataFrames (each representing a stock ticker's time series) back into a
        DataFrame with 'date' as the index and 'stock_ticker' as columns.

        Parameters:
        ----------
        data_dict : dict
            A dictionary where the keys are 'stock_ticker' and the values are DataFrames with time series data.

        Returns:
        -------
        pd.DataFrame
            A DataFrame where 'date' is the index and each 'stock_ticker' is a column, with values representing 'stock_exret'.
        """
        # Concatenate all the DataFrames along rows and assign the 'stock_ticker' as a column
        panel_df = pd.concat(endogenous_dict, names=['stock_ticker', 'date']).reset_index(level=0).reset_index()

        # Rename column '0' to 'stock_exret' if '0' exists in the column names
        if 0 in panel_df.columns:
            panel_df = panel_df.rename(columns={0: 'stock_exret'})

        # Pivot the DataFrame to have 'stock_ticker' as columns and 'date' as the index
        panel_df_pivot = panel_df.pivot(index='date', columns='stock_ticker', values='stock_exret')

        # Sort by date to ensure chronological order
        panel_df_pivot = panel_df_pivot.sort_index()

        return panel_df_pivot

    @staticmethod
    def convert_exogenous_dict_to_panel(exogenous_dict: dict):
        """
        Converts a dictionary of DataFrames (each representing a stock ticker's exogenous time series) back into a
        DataFrame with 'date' as the index and 'stock_ticker' as columns.

        Parameters:
        ----------
        exogenous_dict : dict
            A dictionary where the keys are 'stock_ticker' and the values are DataFrames with exogenous time series data.

        Returns:
        -------
        pd.DataFrame
            A DataFrame where 'date' is the index and each 'stock_ticker' is a column, with values representing the exogenous time series.
        """
        # Concatenate all the DataFrames along rows and assign the 'stock_ticker' as a column
        panel_df = pd.concat(exogenous_dict, names=['stock_ticker', 'date']).reset_index(level=0).reset_index()

        # Rename column '0' to 'exogenous' if '0' exists in the column names
        if 0 in panel_df.columns:
            panel_df = panel_df.rename(columns={0: 'exogenous'})

        # Pivot the DataFrame to have 'stock_ticker' as columns and 'date' as the index
        panel_df_pivot = panel_df.pivot(index='date', columns='stock_ticker', values='exogenous')

        # Sort by date to ensure chronological order
        panel_df_pivot = panel_df_pivot.sort_index()

        return panel_df_pivot

    @staticmethod
    def configure_transformers(forecaster_params: dict, transformers_dict: dict):
        """
        Modifie les dictionnaires contenant une clé 'transformer_exog' en remplaçant la chaîne de caractères
        par la fonction correspondante du dictionnaire TRANSFORMERS.

        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.
        :param transformers_dict: Dictionnaire des transformateurs (ex: TRANSFORMERS).
        """
        if 'transformer_exog' in forecaster_params:
            transformer_str = forecaster_params['transformer_exog']
            if transformer_str in transformers_dict:
                forecaster_params['transformer_exog'] = transformers_dict[transformer_str]()
            elif transformer_str is None:
                forecaster_params['transformer_exog'] = None
            else:
                raise ValueError(f"Transformateur '{transformer_str}' non reconnu. "
                                 f"Veuillez choisir parmi: {list(transformers_dict.keys())}")

    def _build_forecaster(self, regressor: str, regressor_dict: dict | None, lags: int,
                          forecaster_params: dict = None) -> None:
        """
        Construit et assigne un modèle de régression pour la prédiction de séries temporelles à l'attribut `self.forecaster`.

        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param forecaster_params: Dictionnaire contenant les paramètres supplémentaires pour le forecaster.

        :return: None. Assigne le forecaster à l'attribut `self.forecaster`.
        """
        if forecaster_params is None:
            forecaster_params = {}

        # Configuration des transformateurs exogènes
        self.configure_transformers(forecaster_params=forecaster_params, transformers_dict=TRANSFORMERS)

        self.forecaster = ForecasterAutoregMultiSeries(
            regressor=ESTIMATORS[regressor](**regressor_dict),
            lags=lags,
            **forecaster_params
        )

    def _hyperparameters_tuning_by_grid(self, lags_grid: list, grid_search_params_grid: dict, grid_search_params: dict) \
            -> pd.DataFrame:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche par grille.

        :param lags_grid: Grille des lags à tester.
        :param grid_search_params_grid: Grille des hyperparamètres à tester.
        :param grid_search_params: Dictionnaire contenant les paramètres pour la recherche par grille.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres.
        """
        grid_search_params['initial_train_size'] = self.initial_train_size

        # Concaténation des ensembles d'entraînement et de validation pour la recherche d'hyperparamètres
        endogenous_train_val = {k: pd.concat([self.endogenous_train[k], self.endogenous_val[k]]) for k in
                                self.endogenous_train.keys()}
        exogenous_train_val = {k: pd.concat([self.exogenous_train[k], self.exogenous_val[k]]) for k in
                               self.exogenous_train.keys()}

        # Données de test
        endogenous_test = self.endogenous_test
        exogenous_test = self.exogenous_test

        # Concaténation de toutes les données pour le backtesting
        series_full = {k: pd.concat([endogenous_train_val[k], endogenous_test[k]]) for k in
                       endogenous_train_val.keys()}
        exog_full = {k: pd.concat([exogenous_train_val[k], exogenous_test[k]]) for k in exogenous_train_val.keys()}

        tuning_results = grid_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            param_grid=grid_search_params_grid,
            lags_grid=lags_grid,
            **grid_search_params
        )

        return tuning_results

    def _hyperparameters_tuning_by_bayes(self, param_distributions: dict, bayes_search_params: dict,
                                         lags_grid: List[int]) \
            -> tuple[pd.DataFrame, object]:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche bayésienne.

        :param param_distributions: Distributions des hyperparamètres à tester.
        :param bayes_search_params: Dictionnaire contenant les paramètres pour la recherche bayésienne.
        :param lags_grid: Liste des lags à tester et à ajouter aux paramètres bayésiens.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres et l'objet `best_trial`.
        """
        bayes_search_params['initial_train_size'] = self.initial_train_size
        bayes_search_params['kwargs_create_study']['pruner'] = optuna.pruners.SuccessiveHalvingPruner()

        # Ajout de 'lags' à l'espace de recherche
        param_distributions['lags'] = lags_grid

        def search_space(trial):
            search_space_dict = {}
            for param_name, param_values in param_distributions.items():
                if isinstance(param_values, list):
                    search_space_dict[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    low, high = param_values
                    if isinstance(low, int) and isinstance(high, int):
                        search_space_dict[param_name] = trial.suggest_int(param_name, low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                        search_space_dict[param_name] = trial.suggest_float(param_name, low, high)
                    else:
                        raise ValueError(f"Paramètre non supporté : {param_name} avec les valeurs {param_values}.")
                else:
                    raise ValueError(f"Format de paramètre incorrect pour : {param_name}.")
            return search_space_dict

        # Concaténation des ensembles d'entraînement et de validation pour la recherche d'hyperparamètres
        endogenous_train_val = {k: pd.concat([self.endogenous_train[k], self.endogenous_val[k]]) for k in
                                self.endogenous_train.keys()}
        exogenous_train_val = {k: pd.concat([self.exogenous_train[k], self.exogenous_val[k]]) for k in
                               self.exogenous_train.keys()}

        # Données de test
        endogenous_test = self.endogenous_test
        exogenous_test = self.exogenous_test

        # Concaténation de toutes les données pour le backtesting
        series_full = {k: pd.concat([endogenous_train_val[k], endogenous_test[k]]) for k in
                       endogenous_train_val.keys()}
        exog_full = {k: pd.concat([exogenous_train_val[k], exogenous_test[k]]) for k in exogenous_train_val.keys()}

        tuning_results, best_trial = bayesian_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            search_space=search_space,
            **bayes_search_params
        )

        return tuning_results, best_trial

    def _backtest_model(self, backtest_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Effectue un backtesting du modèle de régression multi-séries.

        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        backtest_params['initial_train_size'] = self.initial_train_size

        # Concaténation des ensembles d'entraînement et de validation pour le backtesting
        endogenous_train_val = {k: pd.concat([self.endogenous_train[k], self.endogenous_val[k]]) for k in self.endogenous_train.keys()}
        exogenous_train_val = {k: pd.concat([self.exogenous_train[k], self.exogenous_val[k]]) for k in self.exogenous_train.keys()}

        # Données de test
        endogenous_test = self.endogenous_test
        exogenous_test = self.exogenous_test

        # Concaténation de toutes les données pour le backtesting
        series_full = {k: pd.concat([endogenous_train_val[k], endogenous_test[k]]) for k in endogenous_train_val.keys()}
        exog_full = {k: pd.concat([exogenous_train_val[k], exogenous_test[k]]) for k in exogenous_train_val.keys()}

        metrics_level, backtest_predictions = backtesting_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            **backtest_params
        )

        return metrics_level, backtest_predictions

    @staticmethod
    def configure_metrics(param_dicts: list, metrics_dict: dict):
        """
        Modifie les dictionnaires contenant une clé 'metric' en remplaçant les chaînes de caractères
        par les fonctions correspondantes du dictionnaire METRICS.

        :param param_dicts: Liste de dictionnaires à modifier (ex: [backtest_params, grid_search_params]).
        :param metrics_dict: Dictionnaire des métriques (ex: METRICS).
        """
        for param_dict in param_dicts:
            if 'metric' in param_dict:
                param_dict['metric'] = [metrics_dict[metric_str] for metric_str in param_dict['metric']]

    def save_tuning_results(self, tuning_results: pd.DataFrame, relative_path: str) -> None:
        """
        Enregistre les résultats du tuning dans un fichier CSV à l'emplacement spécifié.

        :param tuning_results: DataFrame contenant les résultats du tuning.
        :param relative_path: Chemin relatif où le fichier CSV sera enregistré.
        """
        try:
            # Convertir le chemin relatif en chemin absolu
            absolute_path = os.path.abspath(relative_path)

            # Enregistrer le DataFrame à l'emplacement spécifié
            tuning_results.to_csv(absolute_path, index=False)
            print(f"Tuning results saved successfully to {absolute_path}")
        except Exception as e:
            print(f"An error occurred while saving the tuning results: {e}")

    def backtest_model_with_tuning(self, regressor: Any, regressor_dict: Optional[dict], lags: int,
                                   lags_grid: List[int],
                                   end_train: str, end_val: str,
                                   grid_search_params: dict, grid_search_params_grid: dict,
                                   bayes_search_params: dict, bayes_search_params_grid: dict,
                                   backtest_params: dict, forecaster_params: Optional[dict] = None,
                                   tuning_method: str = 'grid', save_path: Optional[str] = None) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Optimise les hyperparamètres du modèle, entraîne le modèle avec les meilleurs paramètres,
        et effectue un backtesting.

        :param tuning_method: Méthode d'optimisation des hyperparamètres ('grid' ou 'bayes').
        :param save_path: Si fourni, les résultats du tuning seront enregistrés à ce chemin.
        :param grid_search_params_grid:
        :param lags_grid: Grille de lags à tester lors de l'optimisation d'hyperparamètres.
        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param end_train: Date de fin de l'ensemble d'entraînement.
        :param end_val: Date de fin de l'ensemble de validation.
        :param grid_search_params: Dictionnaire contenant les paramètres pour la recherche par grille.
        :param bayes_search_params: Dictionnaire contenant les paramètres pour la recherche bayésienne.
        :param bayes_search_params_grid: Grille des hyperparamètres pour la recherche bayésienne.
        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.
        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        self.configure_metrics(param_dicts=[grid_search_params, bayes_search_params], metrics_dict=METRICS)

        # Split des données
        self.split_data(end_train=end_train, end_val=end_val)

        # Construction du forecaster
        self._build_forecaster(regressor=regressor, regressor_dict=regressor_dict, lags=lags,
                               forecaster_params=forecaster_params)

        # Optimisation des hyperparamètres
        if tuning_method == 'grid':
            tuning_results = self._hyperparameters_tuning_by_grid(lags_grid=lags_grid,
                                                                  grid_search_params_grid=grid_search_params_grid,
                                                                  grid_search_params=grid_search_params)
        elif tuning_method == 'bayes':
            tuning_results, _ = self._hyperparameters_tuning_by_bayes(param_distributions=bayes_search_params_grid,
                                                                      bayes_search_params=bayes_search_params,
                                                                      lags_grid=lags_grid)
        else:
            raise ValueError("Invalid tuning method. Choose 'grid' or 'bayes'.")

        # Enregistrement des résultats du tuning si un chemin est fourni
        if save_path:
            self.save_tuning_results(tuning_results=tuning_results, relative_path=save_path)

        # Backtesting avec les meilleurs hyperparamètres
        metrics_level, backtest_predictions = self._backtest_model(backtest_params=backtest_params)

        return metrics_level, backtest_predictions

    def backtest_model_without_tuning(self, regressor: Any, regressor_dict: dict | None, lags: int,
                                      end_train: str, end_val: str, backtest_params: dict, forecaster_params: dict = None) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Entraîne le modèle sans optimisation des hyperparamètres et effectue un backtesting.

        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param end_train: Date de fin de l'ensemble d'entraînement.
        :param end_val: Date de fin de l'ensemble de validation.
        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.
        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        self.configure_metrics(param_dicts=[backtest_params], metrics_dict=METRICS)

        # Split des données
        self.split_data(end_train=end_train, end_val=end_val)

        # Construction du forecaster
        self._build_forecaster(regressor=regressor, regressor_dict=regressor_dict, lags=lags,
                               forecaster_params=forecaster_params)

        # Backtesting
        metrics_level, backtest_predictions = self._backtest_model(backtest_params=backtest_params)

        return metrics_level, backtest_predictions


class PredictionPipelineSkforecastNoVal:
    def __init__(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame):
        """
        Initialise la classe avec les données endogènes et exogènes sous forme de DataFrames.
        Les DataFrames sont ensuite convertis en dictionnaires.

        :param endogenous_data: DataFrame des séries endogènes.
        :param exogenous_data: DataFrame des séries exogènes.
        """
        self.endogenous_dict, self.exogenous_dict = self.prepare_data(endogenous_data, exogenous_data)
        self.endogenous_train, self.endogenous_test = None, None
        self.exogenous_train, self.exogenous_test = None, None
        self.forecaster = None
        self.initial_train_size = None  # Taille initiale de l'ensemble d'entraînement pour le backtesting

    @staticmethod
    def ensure_datetime(data: pd.DataFrame, column_name: str = None, format: Optional[str] = '%Y-%m-%d') -> pd.DataFrame:
        """
        Ensure that a specified column in the DataFrame is in a 'datetime' format.
        Converts the specified column to a 'datetime' format using the provided format string.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame.
        column_name : str
            The name of the column to convert to 'datetime' format.
        format : str, optional
            The datetime format to use for conversion. Default is '%Y%m'.

        Returns:
        -------
        pd.DataFrame
            DataFrame with the specified column converted to 'datetime' format.
        """
        if column_name is None:
            data.index = pd.to_datetime(data.index, errors='coerce', format=format)
            return data
        else:
            if column_name not in data.columns:
                raise KeyError(f"La colonne spécifiée '{column_name}' n'existe pas dans le DataFrame.")
            data[column_name] = pd.to_datetime(data[column_name], errors='coerce', format=format)
            return data

    def prepare_data(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame) -> Tuple[dict, dict]:
        """
        Convertit les DataFrames endogènes et exogènes en dictionnaires pour chaque ticker.

        :param endogenous_data: DataFrame des séries endogènes.
        :param exogenous_data: DataFrame des séries exogènes.
        :return: Tuple de dictionnaires (endogenous_dict, exogenous_dict) avec un ticker comme clé et un DataFrame associé.
        """
        endogenous_data = self.ensure_datetime(data=endogenous_data, column_name='date')
        exogenous_data = self.ensure_datetime(data=exogenous_data, column_name='date')

        # Convertir les DataFrames en dictionnaires
        endogenous_dict = series_long_to_dict(
            data=endogenous_data,
            series_id='stock_ticker',
            index='date',
            values='stock_exret',
            freq='ME'
        )
        exogenous_dict = exog_long_to_dict(
            data=exogenous_data,
            series_id='stock_ticker',
            index='date',
            freq='ME'
        )

        return endogenous_dict, exogenous_dict

    def split_data(self, end_train: str) -> None:
        """
        Divise les données endogènes et exogènes en ensembles d'entraînement et de test uniquement.

        :param end_train: Date de fin de l'ensemble d'entraînement.
        """
        self.end_train = pd.to_datetime(end_train)

        # Split des données endogènes
        self.endogenous_train = {k: v.loc[v.index <= self.end_train] for k, v in self.endogenous_dict.items()}
        self.endogenous_test = {k: v.loc[v.index > self.end_train] for k, v in self.endogenous_dict.items()}

        # Split des données exogènes
        self.exogenous_train = {k: v.loc[v.index <= self.end_train] for k, v in self.exogenous_dict.items()}
        self.exogenous_test = {k: v.loc[v.index > self.end_train] for k, v in self.exogenous_dict.items()}

        # Calcul de la taille initiale d'entraînement pour le backtesting
        first_ticker = list(self.endogenous_train.keys())[0]
        self.initial_train_size = len(self.endogenous_train[first_ticker])

        return None

    @staticmethod
    def convert_dict_to_panel(data_dict: dict) -> pd.DataFrame:
        """
        Converts a dictionary of DataFrames (each representing a stock ticker's time series) back into a
        DataFrame with 'date' as the index and 'stock_ticker' as columns.

        Parameters:
        ----------
        data_dict : dict
            A dictionary where the keys are 'stock_ticker' and the values are DataFrames with time series data.

        Returns:
        -------
        pd.DataFrame
            A DataFrame where 'date' is the index and each 'stock_ticker' is a column, with values representing 'stock_exret'.
        """
        # Concatenate all the DataFrames along rows and assign the 'stock_ticker' as a column
        panel_df = pd.concat(data_dict, names=['stock_ticker', 'date']).reset_index(level=0).reset_index()

        # Rename column '0' to 'stock_exret' if '0' exists in the column names
        if 0 in panel_df.columns:
            panel_df = panel_df.rename(columns={0: 'stock_exret'})

        # Pivot the DataFrame to have 'stock_ticker' as columns and 'date' as the index
        panel_df_pivot = panel_df.pivot(index='date', columns='stock_ticker', values='stock_exret')

        # Sort by date to ensure chronological order
        panel_df_pivot = panel_df_pivot.sort_index()

        return panel_df_pivot

    @staticmethod
    def configure_transformers(forecaster_params: dict, transformers_dict: dict):
        """
        Modifie les dictionnaires contenant une clé 'transformer_exog' en remplaçant la chaîne de caractères
        par la fonction correspondante du dictionnaire TRANSFORMERS.

        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.
        :param transformers_dict: Dictionnaire des transformateurs (ex: TRANSFORMERS).
        """
        if 'transformer_exog' in forecaster_params:
            transformer_str = forecaster_params['transformer_exog']
            if transformer_str in transformers_dict:
                forecaster_params['transformer_exog'] = transformers_dict[transformer_str]()
            elif transformer_str is None:
                forecaster_params['transformer_exog'] = None
            else:
                raise ValueError(f"Transformateur '{transformer_str}' non reconnu. "
                                 f"Veuillez choisir parmi: {list(transformers_dict.keys())}")

    def _build_forecaster(self, regressor: str, regressor_dict: dict | None, lags: int,
                          forecaster_params: dict = None) -> None:
        """
        Construit et assigne un modèle de régression pour la prédiction de séries temporelles à l'attribut `self.forecaster`.

        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param forecaster_params: Dictionnaire contenant les paramètres supplémentaires pour le forecaster.

        :return: None. Assigne le forecaster à l'attribut `self.forecaster`.
        """
        if forecaster_params is None:
            forecaster_params = {}

        self.forecaster = ForecasterAutoregMultiSeries(
            regressor=ESTIMATORS[regressor](**regressor_dict),
            lags=lags,
            **forecaster_params
        )

    def _backtest_model(self, backtest_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Effectue un backtesting du modèle de régression multi-séries.

        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        backtest_params['initial_train_size'] = self.initial_train_size

        # Concaténation des ensembles d'entraînement et de test pour le backtesting
        series_full = {k: pd.concat([self.endogenous_train[k], self.endogenous_test[k]]) for k in self.endogenous_train.keys()}
        exog_full = {k: pd.concat([self.exogenous_train[k], self.exogenous_test[k]]) for k in self.exogenous_train.keys()}

        metrics_level, backtest_predictions = backtesting_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            **backtest_params
        )

        return metrics_level, backtest_predictions

    def _hyperparameters_tuning_by_grid(self, lags_grid: list, grid_search_params_grid: dict, grid_search_params: dict) \
            -> pd.DataFrame:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche par grille.

        :param lags_grid: Grille des lags à tester.
        :param grid_search_params_grid: Grille des hyperparamètres à tester.
        :param grid_search_params: Dictionnaire contenant les paramètres pour la recherche par grille.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres.
        """
        grid_search_params['initial_train_size'] = self.initial_train_size

        # Concaténation des ensembles d'entraînement et de test pour la recherche d'hyperparamètres
        series_full = {k: pd.concat([self.endogenous_train[k], self.endogenous_test[k]]) for k in self.endogenous_train.keys()}
        exog_full = {k: pd.concat([self.exogenous_train[k], self.exogenous_test[k]]) for k in self.exogenous_train.keys()}

        tuning_results = grid_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            param_grid=grid_search_params_grid,
            lags_grid=lags_grid,
            **grid_search_params
        )

        return tuning_results

    def _hyperparameters_tuning_by_bayes(self, param_distributions: dict, bayes_search_params: dict,
                                         lags_grid: List[int]) \
            -> tuple[pd.DataFrame, object]:
        """
        Optimise les hyperparamètres du modèle de régression en utilisant une recherche bayésienne.

        :param param_distributions: Distributions des hyperparamètres à tester.
        :param bayes_search_params: Dictionnaire contenant les paramètres pour la recherche bayésienne.
        :param lags_grid: Liste des lags à tester et à ajouter aux paramètres bayésiens.

        :return: DataFrame avec les résultats de l'optimisation des hyperparamètres et l'objet `best_trial`.
        """
        bayes_search_params['initial_train_size'] = self.initial_train_size

        # Ajout de 'lags' à l'espace de recherche
        param_distributions['lags'] = lags_grid

        def search_space(trial):
            search_space_dict = {}
            for param_name, param_values in param_distributions.items():
                if isinstance(param_values, list):
                    search_space_dict[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    low, high = param_values
                    if isinstance(low, int) and isinstance(high, int):
                        search_space_dict[param_name] = trial.suggest_int(param_name, low, high)
                    elif isinstance(low, float) and isinstance(high, float):
                        search_space_dict[param_name] = trial.suggest_float(param_name, low, high)
                    else:
                        raise ValueError(f"Paramètre non supporté : {param_name} avec les valeurs {param_values}.")
                else:
                    raise ValueError(f"Format de paramètre incorrect pour : {param_name}.")
            return search_space_dict

        # Concaténation des ensembles d'entraînement et de test pour la recherche d'hyperparamètres
        series_full = {k: pd.concat([self.endogenous_train[k], self.endogenous_test[k]]) for k in self.endogenous_train.keys()}
        exog_full = {k: pd.concat([self.exogenous_train[k], self.exogenous_test[k]]) for k in self.exogenous_train.keys()}

        tuning_results, best_trial = bayesian_search_forecaster_multiseries(
            forecaster=self.forecaster,
            series=series_full,
            exog=exog_full,
            search_space=search_space,
            **bayes_search_params
        )

        return tuning_results, best_trial

    def backtest_model_with_tuning(self, regressor: Any, regressor_dict: Optional[dict], lags: int,
                                   lags_grid: List[int], end_train: str,
                                   grid_search_params: dict, grid_search_params_grid: dict,
                                   bayes_search_params: dict, bayes_search_params_grid: dict,
                                   backtest_params: dict, forecaster_params: Optional[dict] = None,
                                   tuning_method: str = 'grid', save_path: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimise les hyperparamètres du modèle, entraîne le modèle avec les meilleurs paramètres,
        et effectue un backtesting.

        :param tuning_method: Méthode d'optimisation des hyperparamètres ('grid' ou 'bayes').
        :param save_path: Si fourni, les résultats du tuning seront enregistrés à ce chemin.
        :param lags_grid: Grille de lags à tester lors de l'optimisation d'hyperparamètres.
        :param regressor: Modèle de régression.
        :param regressor_dict: Dictionnaire des hyperparamètres du modèle de régression.
        :param lags: Nombre de retards à inclure dans le modèle.
        :param end_train: Date de fin de l'ensemble d'entraînement.
        :param grid_search_params: Dictionnaire contenant les paramètres pour la recherche par grille.
        :param bayes_search_params: Dictionnaire contenant les paramètres pour la recherche bayésienne.
        :param bayes_search_params_grid: Grille des hyperparamètres pour la recherche bayésienne.
        :param backtest_params: Dictionnaire contenant les paramètres pour le backtesting.
        :param forecaster_params: Dictionnaire contenant les paramètres pour la configuration du forecaster.

        :return: Tuple contenant les métriques par niveau et les prédictions du backtesting.
        """
        # Split des données
        self.split_data(end_train=end_train)

        # Construction du forecaster
        self._build_forecaster(regressor=regressor, regressor_dict=regressor_dict, lags=lags, forecaster_params=forecaster_params)

        # Optimisation des hyperparamètres
        if tuning_method == 'grid':
            tuning_results = self._hyperparameters_tuning_by_grid(lags_grid=lags_grid,
                                                                  grid_search_params_grid=grid_search_params_grid,
                                                                  grid_search_params=grid_search_params)
        elif tuning_method == 'bayes':
            tuning_results, _ = self._hyperparameters_tuning_by_bayes(param_distributions=bayes_search_params_grid,
                                                                      bayes_search_params=bayes_search_params,
                                                                      lags_grid=lags_grid)
        else:
            raise ValueError("Invalid tuning method. Choose 'grid' or 'bayes'.")

        # Backtesting avec les meilleurs hyperparamètres
        metrics_level, backtest_predictions = self._backtest_model(backtest_params=backtest_params)

        return metrics_level, backtest_predictions






if __name__ == '__main__':
    endogenous_data = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/prepare_data_for_prediction/endogenous_data.csv', index_col=0
    )
    exogenous_data = pd.read_csv(
        filepath_or_buffer='../../data/intermediate_data/prepare_data_for_prediction/exogenous_data.csv', index_col=0
    )

    print(endogenous_data.head())
    print(exogenous_data.head())

    prediction_pipeline = PredictionPipelineSkforecast(endogenous_data=endogenous_data, exogenous_data=exogenous_data)

    print(prediction_pipeline.endogenous_dict)
    print(prediction_pipeline.exogenous_dict)

    endogenous_back_to_df = prediction_pipeline.convert_dict_to_panel(data_dict=prediction_pipeline.endogenous_dict)

    print(endogenous_back_to_df.head())

    lags_grid = [12, 24]
    grid_search_params_grid = {'n_estimators': [100, 300], 'learning_rate': [0.01, 0.001],
                               'max_depth': [3, 7], 'subsample': [0.8, 1.0]}

    grid_search_params = {
        'steps': 1,
        'metric': ['mean_squared_error'],
        'aggregate_metric': 'average',
        'fixed_train_size': True,
        'gap': 0,
        'skip_folds': None,
        'allow_incomplete_fold': True,
        'levels': None,
        'refit': False,
        'return_best': True,
        'n_jobs': 'auto',
        'verbose': False,
        'show_progress': True,
        'suppress_warnings': False,
        'output_file': None
    }

    backtest_params = {
        'steps': 1,
        'metric': ['mean_squared_error'],
        'fixed_train_size': False,
        'gap': 0,
        'skip_folds': None,
        'allow_incomplete_fold': True,
        'levels': None,
        'add_aggregated_metric': True,
        'refit': False,
        'interval': None,
        'n_boot': 500,
        'random_state': 123,
        'in_sample_residuals': True,
        'n_jobs': 'auto',
        'verbose': False,
        'show_progress': True,
        'suppress_warnings': True
    }

    forecaster_params = {
        'encoding': 'ordinal_category',
        'transformer_series': None,
        'transformer_exog': None,
        'weight_func': None,
        'series_weights': None,
        'differentiation': None,
        'dropna_from_series': False,
        'fit_kwargs': None,
        'forecaster_id': None
    }

    regressor_dict = {
        'boosting_type': "gbdt",
        'num_leaves': 31,
        'max_depth': 3,
        'learning_rate': 0.01,
        'n_estimators': 100,
        'subsample_for_bin': 200000,
        'objective': None,  # Peut être une chaîne de caractères ou une fonction objective personnalisée
        'class_weight': None,  # Peut être un dictionnaire ou une chaîne de caractères
        'min_split_gain': 0.0,
        'min_child_weight': 1e-3,
        'min_child_samples': 20,
        'subsample': 1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'random_state': None,  # Peut être un entier, un RandomState de NumPy, ou un Generator
        'n_jobs': None,  # Peut être un entier
        'importance_type': "split",  # Options: 'split' ou 'gain'
    }

    bayes_search_params_grid = {
        'n_estimators': (50, 500),  # Range of n_estimators to explore
        'learning_rate': (0.001, 0.1),  # Range of learning rates
        'max_depth': (3, 10),  # Range of max_depth
        'subsample': (0.6, 1.0),  # Range for subsample ratio
        'colsample_bytree': (0.6, 1.0),  # Range for colsample_bytree ratio
        'min_child_samples': (10, 100),  # Minimum number of child samples
        'num_leaves': (31, 128),  # Number of leaves in each tree
    }

    # Example bayesian_search_params dictionary
    bayes_search_params = {
        'steps': 12,  # Number of steps to predict
        'metric': ['mean_squared_error'],  # Metric to optimize
        'aggregate_metric': 'average',  # How to aggregate metrics across multiple series
        'fixed_train_size': True,  # Fixed training size during cross-validation
        'gap': 0,  # No gap between training and testing sets
        'skip_folds': None,  # Not skipping any folds
        'allow_incomplete_fold': True,  # Allow incomplete fold in cross-validation
        'levels': None,  # Optimize for all levels
        'refit': False,  # Refit the model with the best parameters
        'return_best': True,  # Return the best trial's parameters
        'n_trials': 5,  # Number of trials for the Bayesian search
        'random_state': 123,  # Random state for reproducibility
        'n_jobs': 'auto',  # Use all available cores
        'verbose': True,  # Minimal verbosity
        'show_progress': True,  # Show progress bar
        'suppress_warnings': False,  # Display warnings
        'engine': 'optuna',  # Use Optuna for Bayesian optimization
        'kwargs_create_study': {},  # Additional arguments for creating the Optuna study
        'kwargs_study_optimize': {}  # Additional arguments for optimizing the study
    }

    metrics_level, backtest_predictions = prediction_pipeline.backtest_model_with_tuning(
        regressor='light_gbm_regressor', regressor_dict=regressor_dict, lags=12, lags_grid=lags_grid,
        end_train='2007-12-31', end_val='2010-12-31',
        grid_search_params=grid_search_params, grid_search_params_grid=grid_search_params_grid,
        bayes_search_params=bayes_search_params, bayes_search_params_grid=bayes_search_params_grid,
        backtest_params=backtest_params, forecaster_params=forecaster_params,
        tuning_method='bayes', save_path=None
    )

    print(metrics_level)
    print(backtest_predictions)


