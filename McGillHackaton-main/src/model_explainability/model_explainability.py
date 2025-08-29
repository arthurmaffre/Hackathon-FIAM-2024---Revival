import shap
import pandas as pd
from typing import Optional, Union, Any
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.utils import load_forecaster
import matplotlib.pyplot as plt



class ModelExplainability:
    def __init__(self, y_train_dict: dict, X_train_dict: dict, forecaster: ForecasterAutoregMultiSeries | Any):
        self.y_train_dict = y_train_dict
        self.X_train_dict = X_train_dict
        self.exogenous_train, self.endogenous_train = self.create_train_data_from_dict(y_train_dict, X_train_dict, forecaster)
        self.forecaster = None
        self.shap_values = None
        self.explainer = None

    def create_train_data_from_dict(self, y_train_dict: dict, X_train_dict: dict, forecaster: ForecasterAutoregMultiSeries) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train = forecaster.create_train_X_y(series=y_train_dict, exog=X_train_dict)
        return X_train, pd.DataFrame(y_train)

    def fit_model(self, forecaster: ForecasterAutoregMultiSeries | Any,
                  store_last_window: Union[bool, list] = True,
                  store_in_sample_residuals: bool = True,
                  suppress_warnings: bool = False) -> ForecasterAutoregMultiSeries:
        """
        Fits the regression model.

        :param forecaster: The ForecasterAutoregMultiSeries model to fit.
        :param store_last_window: Option to store the last window.
        :param store_in_sample_residuals: Option to store in-sample residuals.
        :param suppress_warnings: Option to suppress warnings.
        :return: The fitted ForecasterAutoregMultiSeries model.
        """
        self.forecaster = forecaster
        self.forecaster.fit(
            series=self.y_train_dict,
            exog=self.X_train_dict,
            store_last_window=store_last_window,
            store_in_sample_residuals=store_in_sample_residuals,
            suppress_warnings=suppress_warnings
        )
        return self.forecaster

    def compute_shap_values(self):
        self.explainer = shap.TreeExplainer(model=self.forecaster.regressor)
        # drop the _level_skforecast column
        shap_values = self.explainer.shap_values(X=self.exogenous_train)
        self.shap_values = shap_values

        return shap_values

    def plot_summary_shap(self, max_display: int = 10, save_path: Optional[str] = None):
        """
        Plots the SHAP summary plot and saves it if a save path is provided.

        :param max_display: Maximum number of features to display in the plot.
        :param save_path: Path to save the plot. If None, the plot will be shown instead.
        """
        shap.initjs()
        plt.figure()
        shap.summary_plot(
            shap_values=self.shap_values,
            features=self.exogenous_train,
            feature_names=self.exogenous_train.columns,
            max_display=max_display
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

    def plot_bar_summary_shap(self, max_display: int = 10, save_path: Optional[str] = None):
        """
        Plots the SHAP bar summary plot and saves it if a save path is provided.

        :param max_display: Maximum number of features to display in the plot.
        :param save_path: Path to save the plot. If None, the plot will be shown instead.
        """
        shap.initjs()
        plt.figure()
        shap.summary_plot(
            shap_values=self.shap_values,
            features=self.exogenous_train,
            feature_names=self.exogenous_train.columns,
            max_display=max_display,
            plot_type='bar'
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
        else:
            plt.show()


if __name__ == '__main__':
    import pickle
    with open('../../data/intermediate_data/prediction_pipeline/y_train_light_gbm_regressor.pkl', 'rb') as f:
        y_train_dict = pickle.load(f)

    with open('../../data/intermediate_data/prediction_pipeline/X_train_light_gbm_regressor.pkl', 'rb') as f:
        X_train_dict = pickle.load(f)

    print(y_train_dict)
    print(X_train_dict)
    # Load the forecaster model
    forecaster = load_forecaster(
        file_name='../../data/intermediate_data/prediction_pipeline/light_gbm_regressor_forecaster.pkl',
        verbose=True
    )

    print(forecaster)

    # Initialize ModelExplainability object
    model_explainability = ModelExplainability(y_train_dict=y_train_dict, X_train_dict=X_train_dict, forecaster=forecaster)
    model_explainability.fit_model(forecaster=forecaster)
    shap_values = model_explainability.compute_shap_values()
    model_explainability.plot_summary_shap(save_path=None)
    model_explainability.plot_bar_summary_shap(save_path=None)

    print(shap_values)


