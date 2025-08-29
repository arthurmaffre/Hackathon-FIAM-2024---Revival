from sklearn.tree import DecisionTreeClassifier
from typing import Any


def decision_tree_classifier(
    *,
    criterion: str = "gini",
    splitter: str = "best",
    max_depth: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    min_weight_fraction_leaf: float = 0.0,
    max_features: Any = None,
    random_state: Any = None,
    max_leaf_nodes: int = None,
    min_impurity_decrease: float = 0.0,
    class_weight: Any = None,
    ccp_alpha: float = 0.0,
    monotonic_cst: Any = None,
) -> DecisionTreeClassifier:
    """Returns a Decision Tree Classifier with the specified parameters."""
    return DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        random_state=random_state,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        monotonic_cst=monotonic_cst,
        ccp_alpha=ccp_alpha
    )


