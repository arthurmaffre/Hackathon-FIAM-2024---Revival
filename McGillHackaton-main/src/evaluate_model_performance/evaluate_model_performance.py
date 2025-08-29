from typing import Any, Callable, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np


class EvaluateModelPerformance:
    def __init__(self, y_test: pd.DataFrame, y_pred: pd.DataFrame,
                 y_train: pd.DataFrame, metrics: Dict[str, Dict[str, Any]]) -> None:
        """
        Initialise la classe avec les vraies valeurs, les valeurs prédites, les valeurs en échantillon et les métriques à utiliser.

        :param y_test: Véritables étiquettes (DataFrame avec un MultiIndex sur 'stock_ticker' et la date).
        :param y_pred: Étiquettes prédites (DataFrame avec un MultiIndex sur 'stock_ticker' et la date).
        :param y_train: Valeurs en échantillon (DataFrame avec un MultiIndex sur 'stock_ticker' et la date).
        :param metrics: Dictionnaire des métriques à utiliser et leurs paramètres supplémentaires.
        """
        # Convertir les données de panel en format tabulaire
        self.y_test: pd.DataFrame = y_test
        self.y_pred: pd.DataFrame = y_pred
        self.y_train: pd.DataFrame = y_train
        self._metrics: Dict[str, Dict[str, Any]] = metrics
        self._metric_functions: Dict[str, Callable[..., float]] = self._initialize_metric_functions()

    # @staticmethod
    # def _convert_panel_to_tabular(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Convertit un DataFrame de format panel (avec un MultiIndex sur 'stock_ticker' et la date)
    #     en un format tabulaire avec la date mensuelle en index, les tickers en colonnes, et les rendements en valeurs.
    #
    #     :param df: DataFrame en format panel.
    #     :return: DataFrame en format tabulaire.
    #     """
    #     # Vérifier que le DataFrame a un MultiIndex avec 'stock_ticker' et la date
    #     if not isinstance(df.index, pd.MultiIndex):
    #         raise ValueError("Le DataFrame doit avoir un MultiIndex avec 'stock_ticker' et 'date'.")
    #
    #     # rename pd.multi index level 0 par stock_ticker and level 1 by date
    #     df.index.rename(['stock_ticker', 'date'], inplace=True)
    #
    #     # Unstack pour obtenir les tickers en colonnes et la date en index
    #     df_tabular = df.unstack(level='stock_ticker')
    #
    #     # Si les colonnes sont multi-niveaux, les aplatir
    #     if isinstance(df_tabular.columns, pd.MultiIndex):
    #         # On suppose qu'il n'y a qu'une seule colonne de valeurs, on récupère donc le niveau -1 des colonnes
    #         df_tabular.columns = df_tabular.columns.get_level_values(-1)
    #
    #     # Optionnel : trier les colonnes par ordre alphabétique des tickers
    #     df_tabular = df_tabular.sort_index(axis=1)
    #
    #     return df_tabular

    @staticmethod
    def _r_squared_oos(y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Calcule le R² hors échantillon.

        :param y_true: Véritables étiquettes.
        :param y_pred: Étiquettes prédites.
        :return: Valeur de R² hors échantillon.
        """
        return 1 - np.sum(np.square((y_true - y_pred))) / np.sum(np.square(y_true))

    def _initialize_metric_functions(self) -> Dict[str, Callable[..., float]]:
        """
        Initialise le dictionnaire des fonctions de métriques.

        :return: Dictionnaire des fonctions de métriques.
        """
        return {
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "r2_score": r2_score,
            "r2_score_oos": self._r_squared_oos
        }

    def _evaluate_metric(self, metric_name: str, y_true_col: pd.Series, y_pred_col: pd.Series, metric_params: Dict[str, Any]) -> float:
        """
        Évalue une métrique spécifique.

        :param metric_name: Nom de la métrique.
        :param y_true_col: Véritables étiquettes pour une colonne spécifique.
        :param y_pred_col: Étiquettes prédites pour une colonne spécifique.
        :param metric_params: Paramètres supplémentaires pour la métrique.
        :return: Valeur de la métrique évaluée.
        """
        metric_function = self._metric_functions.get(metric_name)
        if not metric_function:
            raise ValueError(f"Métrique '{metric_name}' non reconnue.")

        # Aligner les index pour s'assurer que les séries sont comparables
        y_true_col, y_pred_col = y_true_col.align(y_pred_col, join='inner')

        return metric_function(y_true_col, y_pred_col, **metric_params)

    def evaluate(self) -> pd.DataFrame:
        """
        Évalue les performances du modèle selon les métriques spécifiées pour chaque ticker.

        :return: DataFrame des résultats des métriques par ticker.
        """
        results: Dict[str, Dict[str, float]] = {metric_name: {} for metric_name in self._metrics.keys()}
        tickers = self.y_test.columns

        for ticker in tickers:
            y_true_col = self.y_test[ticker].dropna()
            y_pred_col = self.y_pred[ticker].reindex(y_true_col.index).dropna()

            for metric_name, metric_params in self._metrics.items():
                try:
                    results[metric_name][ticker] = self._evaluate_metric(
                        metric_name, y_true_col, y_pred_col, metric_params
                    )
                except Exception as e:
                    print(f"Erreur lors du calcul de '{metric_name}' pour le ticker '{ticker}': {e}")
                    results[metric_name][ticker] = np.nan

        return pd.DataFrame(results).T

    def aggregate_metrics(self) -> pd.DataFrame:
        """
        Calcule la moyenne des métriques évaluées sur tous les tickers.

        :return: DataFrame contenant les moyennes des métriques.
        """
        evaluation_results = self.evaluate()

        # Calculer la moyenne des métriques par ligne (chaque ligne est une métrique)
        aggregated_metrics = evaluation_results.mean(axis=1)

        # Retourner sous forme de DataFrame pour conserver une présentation cohérente
        return pd.DataFrame(aggregated_metrics, columns=['mean'])

# Exemple d'utilisation de la classe avec des données factices
if __name__ == '__main__':
    # Chargement des données
    y_test = pd.read_csv('../../data/intermediate_data/prediction_pipeline/y_test.csv', index_col=['stock_ticker', 'date'])
    y_pred = pd.read_csv('../../data/intermediate_data/prediction_pipeline/y_pred.csv', index_col=['stock_ticker', 'date'])
    y_train = pd.read_csv('../../data/intermediate_data/prediction_pipeline/y_train.csv', index_col=['stock_ticker', 'date'])

    # Définition des métriques à évaluer
    metrics = {
        "mean_squared_error": {},
        "mean_absolute_error": {},
        "r2_score_oos": {}
    }

    # Initialisation de la classe
    evaluator = EvaluateModelPerformance(y_test=y_test, y_pred=y_pred, y_train=y_train, metrics=metrics)

    # Évaluation des performances
    performance_results = evaluator.evaluate()

    # Calcul des métriques agrégées par la moyenne
    aggregated_metrics = evaluator.aggregate_metrics()

    # Affichage des résultats
    print("Performance par ticker:")
    print(performance_results)

    print("\nMoyenne des métriques agrégées:")
    print(aggregated_metrics)
