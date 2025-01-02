from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from typing import List, Optional, Tuple, Union
from sklearn.base import is_classifier, is_regressor
import copy
from .deep_imputer import DeepModelImputer


class Imputer:
    def __init__(self,
                 regression_imputer: object,
                 classification_imputer: object,
                 trials: int):

        self.model: Optional[object] = None
        self.__regression_model = regression_imputer
        self.__classification_model = classification_imputer
        self.trials = trials

    def set(self, categorical: bool):
        if categorical:
            self.model = self.__classification_model
        else:
            self.model = self.__regression_model

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def copy(self):
        return copy.deepcopy(self)


class Sequential:
    def __init__(self, reset: bool = False):
        self.imputers: List[Imputer] = []
        if not reset:
            self.__build_default_imputers()


    def add(self,
            regression_imputer,
            classification_imputer: object,
            trials: int = 1, index: int = -1):

        # Handle out-of-bounds indices
        if index > len(self.imputers):
            raise IndexError("Index out of range for the current imputers list.")
        if regression_imputer is not None and not is_regressor(regression_imputer) and not isinstance(regression_imputer, DeepModelImputer):
            raise ValueError('your imputer not regression type')
        if classification_imputer is not None and not is_classifier(classification_imputer):
            raise ValueError('your imputer not classification type')

        imputer = Imputer(regression_imputer, classification_imputer, trials=trials)

        # Add the imputer at the specified index or at the end
        if index != -1:
            self.imputers.insert(index, imputer)
        else:
            self.imputers.append(imputer)

    def reset(self):
        """Clear all imputers."""
        self.imputers = []

    def __build_default_imputers(self):
        """Create default imputers with pre-configured models"""
        models = [
            ('RandomForest', 'sqrt'), ('GradientBoosting', 0.95), ('GradientBoosting', 0.95),
            ('RandomForest', 0.95), ('RandomForest', 'log2'), ('GradientBoosting', 'log2'),
            ('RandomForest', 0.95), ('RandomForest', 0.95), ('GradientBoosting', 0.95)
        ]

        for model_type, max_features in models:
            regression_model, classification_model = build_model(model_type, max_features)
            self.add(regression_model, classification_model, trials=2)


def build_model(model_type: str, max_features: Union[str, float]) -> Tuple[object, object]:
    """Create a regression and classification model based on the given type and feature set."""
    if model_type == 'RandomForest':
        reg_model = RandomForestRegressor(n_estimators=100, max_depth=40, n_jobs=-1,
                                          max_features=max_features, min_samples_leaf=1, min_samples_split=2)
        clf_model = RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=-1,
                                           max_features=max_features, min_samples_leaf=1, min_samples_split=2)
    elif model_type == 'GradientBoosting':
        reg_model = GradientBoostingRegressor(max_features=max_features)
        clf_model = RandomForestClassifier(n_estimators=100, max_depth=40, n_jobs=-1,
                                           max_features=max_features, min_samples_leaf=1, min_samples_split=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return reg_model, clf_model
