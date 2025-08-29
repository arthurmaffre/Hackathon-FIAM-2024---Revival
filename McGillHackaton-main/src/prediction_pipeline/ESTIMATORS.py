from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lars,
    Lasso, LassoLars, LinearRegression, LogisticRegression,
    OrthogonalMatchingPursuit, PassiveAggressiveRegressor, RANSACRegressor,
    Ridge, TheilSenRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

ESTIMATORS = {
    "ada_boost_classifier": AdaBoostClassifier,
    "ada_boost_regressor": AdaBoostRegressor,
    "ard_regressor": ARDRegression,
    "bayesian_ridge_regressor": BayesianRidge,
    "catboost_classifier": CatBoostClassifier,
    "catboost_regressor": CatBoostRegressor,
    "decision_tree_classifier": DecisionTreeClassifier,
    "decision_tree_regressor": DecisionTreeRegressor,
    "elastic_net_regressor": ElasticNet,
    "extra_trees_classifier": ExtraTreesClassifier,
    "extra_trees_regressor": ExtraTreesRegressor,
    "gaussian_process_regressor": GaussianProcessRegressor,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    "hist_gradient_boosting_classifier": HistGradientBoostingClassifier,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
    "huber_regressor": HuberRegressor,
    "k_neighbors_classifier": KNeighborsClassifier,
    "k_neighbors_regressor": KNeighborsRegressor,
    "lars_regressor": Lars,
    "lasso_lars_regressor": LassoLars,
    "lasso_regressor": Lasso,
    "light_gbm_regressor": LGBMRegressor,
    "linear_regression_regressor": LinearRegression,
    "linear_svr_regressor": LinearSVR,
    "logistic_regression_classifier": LogisticRegression,
    "mlp_classifier": MLPClassifier,
    "mlp_regressor": MLPRegressor,
    "nu_svr_regressor": NuSVR,
    "omp_regressor": OrthogonalMatchingPursuit,
    "passive_aggressive_regressor": PassiveAggressiveRegressor,
    "ransac_regressor": RANSACRegressor,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
    "ridge_regressor": Ridge,
    "svc_classifier": SVC,
    "svr_regressor": SVR,
    "theil_sen_regressor": TheilSenRegressor,
    "xgboost_classifier": XGBClassifier,
    "xgboost_regressor": XGBRegressor,
}

