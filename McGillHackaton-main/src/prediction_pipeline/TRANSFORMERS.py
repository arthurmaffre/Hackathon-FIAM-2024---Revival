from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

TRANSFORMERS = {
    "pca": PCA,
    "deseasonalize": Deseasonalizer,
    "detrend": Detrender,
    "min_max_scaler": MinMaxScaler,
    "power_transformer": PowerTransformer,
    "robust_scaler": RobustScaler,
    "standard_scaler": StandardScaler,
}