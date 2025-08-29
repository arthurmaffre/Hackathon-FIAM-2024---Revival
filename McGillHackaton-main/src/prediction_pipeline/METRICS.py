import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, mean_absolute_error, median_absolute_error)


def r_squared_modified(y_true: pd.Series, y_pred: pd.Series, y_train: pd.Series) -> float:
    """
    Calcule le R² ajusté pour les valeurs en échantillon et hors échantillon.

    :param y_true: Véritables étiquettes.
    :param y_pred: Étiquettes prédites.
    :param y_train: Valeurs en échantillon.
    :return: Valeur de R² ajusté.
    """
    SSR = np.sum((y_true - y_pred) ** 2)
    SST = np.sum((y_true - np.mean(y_train)) ** 2)
    return 1 - SSR / SST


def _r_squared_oos(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calcule le R² hors échantillon.

    :param y_true: Véritables étiquettes.
    :param y_pred: Étiquettes prédites.
    :return: Valeur de R² hors échantillon.
    """
    return 1 - np.sum(np.square((y_true - y_pred))) / np.sum(np.square(y_true))


def pesaran_timmermann_stat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Directional Accuracy Score, Pesaran-Timmermann statistic, and its p-value.

    Parameters:
    -----------
    y_true : np.ndarray
        A NumPy array containing the true values.
    y_pred : np.ndarray
        A NumPy array containing the predicted values.

    Returns:
    --------
    directional_accuracy : float
        The directional accuracy score.
    """
    # Number of observations
    n_observations = y_true.shape[0]

    # Calculate the directional accuracy score
    directional_accuracy = np.sum(np.sign(y_true) == np.sign(y_pred)) / n_observations

    # Calculate the proportion of positive true values
    proportion_true_positive = np.sum(y_true > 0) / n_observations
    variance_true = proportion_true_positive * (1 - proportion_true_positive) / n_observations

    # Calculate the proportion of positive predicted values
    proportion_pred_positive = np.sum(y_pred > 0) / n_observations
    variance_pred = proportion_pred_positive * (1 - proportion_pred_positive) / n_observations

    # Calculate the expected proportion of directional accuracy under independence
    expected_proportion = proportion_true_positive * proportion_pred_positive + (1 - proportion_true_positive) * (1 - proportion_pred_positive)

    # Calculate the variance of the expected proportion
    variance_expected = expected_proportion * (1 - expected_proportion) / n_observations

    # Calculate the variance of the test statistic
    variance_statistic = ((2 * proportion_true_positive - 1) ** 2) * variance_pred + ((2 * proportion_pred_positive - 1) ** 2) * variance_true + 4 * variance_true * variance_pred

    # Calculate the Pesaran-Timmermann statistic
    pt_statistic = (directional_accuracy - expected_proportion) / np.sqrt(variance_expected - variance_statistic)

    # Calculate the p-value of the Pesaran-Timmermann statistic
    p_value = 1 - stats.norm.cdf(pt_statistic, 0, 1)

    return directional_accuracy


METRICS = {
    'mean_squared_error': mean_squared_error,
    'r2_score': r2_score,
    'accuracy_score': accuracy_score,
    'f1_score': f1_score,
    'confusion_matrix': confusion_matrix,
    'r_squared_modified': r_squared_modified,
    'r_squared_oos': _r_squared_oos,
    'pesaran_timmermann_stat': pesaran_timmermann_stat,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'mean_absolute_error': mean_absolute_error,
    'median_absolute_error': median_absolute_error
}

