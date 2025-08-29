from sktime.forecasting.compose import make_reduction, ForecastingPipeline, TransformedTargetForecaster
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from src.prediction_pipeline.ESTIMATORS import ESTIMATORS
from src.prediction_pipeline.TRANSFORMERS import TRANSFORMERS
from typing import Any, Dict
import pandas as pd
import numpy as np
import warnings
import os

# Activer la solution de repli sur CPU pour les opérations non prises en charge par MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

warnings.simplefilter(action='ignore', category=FutureWarning)


class TransformerFactory(object):
    """Factory class to create transformers."""

    @staticmethod
    def create_transformer(transformer_name: str, params: dict):
        transformer_class = TRANSFORMERS.get(transformer_name)
        if transformer_class is None:
            raise ValueError(
                f"Transformer {transformer_name} not found in TRANSFORMERS dictionary: {TRANSFORMERS.keys()}."
            )

        if 'sklearn' in str(transformer_class):
            return TabularToSeriesAdaptor(transformer=transformer_class(**params))
        return transformer_class(**params)


class EstimatorFactory(object):
    """Factory class to create forecasters and regressors."""

    @staticmethod
    def create_estimator(estimator_name: str, estimator_params: dict, make_reduction_params: dict = None):
        estimator_class = ESTIMATORS.get(estimator_name)
        if estimator_class is None:
            raise ValueError(f"Estimator {estimator_name} not found in ESTIMATORS dictionary: {ESTIMATORS.keys()}.")

        if 'sklearn' in str(estimator_class):
            return make_reduction(estimator=estimator_class(**estimator_params), **make_reduction_params)
        return estimator_class(**estimator_params)


class PredictionPipelineSktime(object):
    def __init__(self, endogenous_data: pd.DataFrame, exogenous_data: pd.DataFrame, test_size: float = 0.2):
        """
        Initialise la pipeline de prédiction.

        :param endogenous_data: Données endogènes.
        :param exogenous_data: Données exogènes.
        :param test_size: Taille de l'ensemble de test.
        """
        self.endogenous_data = endogenous_data
        self.exogenous_data = exogenous_data
        self.y_train, self.y_test, self.X_train, self.X_test = self._split_train_test_data(
            test_size=test_size
        )
        self.fh = self._compute_forecasting_horizon()
        self.pipeline = None

    def _compute_forecasting_horizon(self, series_key=None) -> ForecastingHorizon:
        """
        Calcule l'horizon de prévision pour une seule série à partir d'un DataFrame avec un MultiIndex.

        :param series_key: Un tuple représentant les clés du MultiIndex pour sélectionner une série spécifique.
                           Par exemple, ('ticker', 'date').
                           Si None, la première série disponible dans y_test sera utilisée.

        :return: Un objet ForecastingHorizon pour la série sélectionnée.
        """
        if series_key is not None:
            # Sélectionne la série spécifique dans y_test en fonction du series_key fourni
            y_series = self.y_test.loc[series_key]
        else:
            # Utilise la première série disponible si aucune clé n'est fournie
            y_series = self.y_test.groupby(level=1).first()

        # Calcule la taille de la série sélectionnée
        series_length = len(y_series)

        # Crée l'objet ForecastingHorizon en fonction de la taille de la série sélectionnée
        return ForecastingHorizon(values=np.arange(1, series_length + 1))

    def _split_train_test_data(self, test_size: float = 0.2):
        """
        Sépare les données en ensembles d'entraînement et de test.

        :param test_size: Taille de l'ensemble de test.

        :return: y_train, y_test, X_train, X_test
        """
        y_train, y_test, X_train, X_test = temporal_train_test_split(
            y=self.endogenous_data,
            X=self.exogenous_data,
            test_size=test_size
        )
        return y_train, y_test, X_train, X_test

    def fit_pipeline(self, endogenous_pipeline_dict: Dict[str, dict], exogenous_pipeline_dict: Dict[str, dict],
                     estimator_name: str, estimator_params: dict, make_reduction_params: dict = None):
        """
        Ajuste la pipeline de données en fonction de l'estimateur (régression).

        :param endogenous_pipeline_dict: Dictionnaire des étapes de la pipeline endogène.
        :param exogenous_pipeline_dict: Dictionnaire des étapes de la pipeline exogène.
        :param estimator_name: Nom de l'estimateur.
        :param estimator_params: Paramètres de l'estimateur.
        :param make_reduction_params: Paramètres pour la fonction make_reduction (si applicable).
        """
        endogenous_pipeline_dict = endogenous_pipeline_dict or {}
        exogenous_pipeline_dict = exogenous_pipeline_dict or {}

        self._fit_forecasting_pipeline(
            endogenous_pipeline_dict=endogenous_pipeline_dict,
            exogenous_pipeline_dict=exogenous_pipeline_dict,
            forecaster_name=estimator_name,
            forecaster_params=estimator_params,
            make_reduction_params=make_reduction_params
        )

    def _fit_forecasting_pipeline(self, endogenous_pipeline_dict: Dict[str, dict], exogenous_pipeline_dict: Dict[str, dict],
                                  forecaster_name: str, forecaster_params: dict, make_reduction_params: dict):
        """
        Ajuste la pipeline de prévision.

        :param endogenous_pipeline_dict: Dictionnaire des étapes de la pipeline endogène.
        :param exogenous_pipeline_dict: Dictionnaire des étapes de la pipeline exogène.
        :param forecaster_name: Nom du forecaster.
        :param forecaster_params: Paramètres du forecaster.
        :param make_reduction_params: Paramètres pour la fonction make_reduction.
        """
        # Create the endogenous pipeline
        endogenous_steps = [(step_name, TransformerFactory.create_transformer(transformer_name=step_name, params=params))
                            for step_name, params in endogenous_pipeline_dict.items()]

        forecaster = EstimatorFactory.create_estimator(
            estimator_name=forecaster_name,
            estimator_params=forecaster_params,
            make_reduction_params=make_reduction_params
        )
        endogenous_steps.append(("forecaster", forecaster))
        endogenous_pipeline = TransformedTargetForecaster(steps=endogenous_steps)

        # Create the exogenous pipeline
        exogenous_steps: Any = [(step_name, TransformerFactory.create_transformer(transformer_name=step_name, params=params))
                                for step_name, params in exogenous_pipeline_dict.items()]
        exogenous_steps.append(("forecaster", endogenous_pipeline))
        self.pipeline = ForecastingPipeline(steps=exogenous_steps)

        # Fit the exogenous pipeline with endogenous data as target
        self.pipeline.fit(y=self.y_train, X=self.X_train, fh=self.fh)

    def predict(self):
        """
        Prédit les valeurs pour les données de test.

        :return: Les prédictions.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not fitted.")

        return self.pipeline.predict(X=self.X_test, fh=self.fh)


if __name__ == '__main__':

    endogenous_data = pd.read_csv(filepath_or_buffer='../../data/intermediate_data/prepare_data_for_prediction/endogenous_data.csv', index_col=[0, 1])
    exogenous_data = pd.read_csv(filepath_or_buffer='../../data/intermediate_data/prepare_data_for_prediction/exogenous_data.csv', index_col=[0, 1])

    print(endogenous_data)
    print(exogenous_data)

    # --------- VARIABLES DE TEST --------- #


    # Dictionnaire de pipeline endogène et exogène
    endogenous_pipeline_dict = {}

    exogenous_pipeline_dict = {
        "min_max_scaler": {},
    }

    estimator_name = "decision_tree_classifier"
    estimator_params = {}
    make_reduction_params = {"strategy": "recursive", "window_length": 12, "scitype": "tabular-regressor"}

    # ------------------------------------- #

    # Initialisation et utilisation de la classe PredictionPipelineSktime
    pp = PredictionPipelineSktime(endogenous_data=endogenous_data, exogenous_data=exogenous_data)
    pp.fit_pipeline(
        endogenous_pipeline_dict=endogenous_pipeline_dict,
        exogenous_pipeline_dict=exogenous_pipeline_dict,
        estimator_name=estimator_name,
        estimator_params=estimator_params,
        make_reduction_params=make_reduction_params
    )
    predictions = pp.predict()

    print(predictions)
