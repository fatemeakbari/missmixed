import copy
from typing import List, Optional, Tuple, Union

from sklearn.base import is_regressor, is_classifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier

from .deep_imputer import DeepModelImputer

"""
Imputer: A class for handling imputation of missing values using regression or classification models.

This class provides functionality to impute missing values in datasets using either regression or
classification models, depending on the nature of the data (categorical or numerical).

Attributes:
    model (Optional[object]): The current imputation model (regression or classification).
    __regression_model (object): The regression model to use for numerical data.
    __classification_model (object): The classification model to use for categorical data.
    trials (int): The number of trials to run for imputation.

Methods:
    set_model(categorical): Sets the appropriate model (regression or classification) based on the data type.
    fit(X, Y): Fits the imputation model to the provided data.
    predict(X): Predicts the missing values using the trained model.
    copy(): Returns a deep copy of the imputer.
"""


class Imputer:
    def __init__(self,
                 regression_imputer: object,
                 classification_imputer: object,
                 trials: int):
        """
        Initializes the Imputer class.

        Args:
            regression_imputer (object): The regression model to use for numerical data.
            classification_imputer (object): The classification model to use for categorical data.
            trials (int): The number of trials to run for imputation.
        """
        self.model: Optional[object] = None
        self.__regression_model = regression_imputer
        self.__classification_model = classification_imputer
        self.trials = trials

    def set_model(self, categorical: bool):
        """
        Sets the appropriate model (regression or classification) based on the data type.

        Args:
            categorical (bool): Whether the data is categorical (True) or numerical (False).
        """
        if categorical:
            self.model = self.__classification_model
        else:
            self.model = self.__regression_model

    def fit(self, X, Y):
        """
        Fits the imputation model to the provided data.

        Args:
            X: The input features for training.
            Y: The target values for training.
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Predicts the missing values using the trained model.

        Args:
            X: The input features for prediction.

        Returns:
            The predicted values.
        """
        return self.model.predict(X)

    def copy(self):
        """
        Returns a deep copy of the imputer.

        Returns:
            Imputer: A deep copy of the current imputer.
        """
        return copy.deepcopy(self)


"""
Sequential: A class for managing a sequence of imputers.

This class provides functionality to manage a sequence of imputers, allowing for the addition,
removal, and configuration of imputers. It also supports building default imputers with
pre-configured models.

Attributes:
    imputers (List[Imputer]): A list of imputers in the sequence.

Methods:
    add(regression_imputer, classification_imputer, trials, index): Adds an imputer to the sequence.
    reset(): Clears all imputers from the sequence.
    __build_default_imputers(): Builds default imputers with pre-configured models.
    __build_model(model_type, max_features): Creates a regression and classification model based on the given type and feature set.
"""


class Sequential:
    def __init__(self, reset: bool = False, trials: int = 1):
        """
        Initializes the Sequential class.

        Args:
            reset (bool): Whether to reset the imputers list. Default is False.
        """
        self.imputers: List[Imputer] = []
        self.trials = trials
        if not reset:
            self.__build_default_imputers()

    def add(self,
            regression_imputer,
            classification_imputer: object,
            trials: int = 1, index: int = -1):
        """
        Adds an imputer to the sequence.

        Args:
            regression_imputer: The regression model to use for numerical data.
            classification_imputer (object): The classification model to use for categorical data.
            trials (int): The number of trials to run for imputation. Default is 1.
            index (int): The index at which to add the imputer. Default is -1 (end of the list).

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the provided imputer is not of the correct type.
        """
        # Handle out-of-bounds indices
        if index > len(self.imputers):
            raise IndexError("Index out of range for the current imputers list.")
        if regression_imputer is not None and not is_regressor(regression_imputer) and not isinstance(
                regression_imputer, DeepModelImputer):
            raise ValueError('your imputer not regression type')

        if classification_imputer is not None and not is_classifier(classification_imputer) and not isinstance(
                regression_imputer, DeepModelImputer):
            raise ValueError('your imputer not classification type')
        imputer = Imputer(regression_imputer, classification_imputer, trials=trials)

        # Add the imputer at the specified index or at the end
        if index != -1:
            self.imputers.insert(index, imputer)
        else:
            self.imputers.append(imputer)

    def reset(self):
        """Clears all imputers from the sequence."""
        self.imputers = []

    def __build_default_imputers(self):
        """
        Builds default imputers with pre-configured models.

        This method creates a list of default imputers using pre-configured regression and
        classification models.
        """
        models = [
            ('GradientBoosting', 'sqrt'), ('RandomForest', 'sqrt'), ('GradientBoosting', 0.95),
            ('GradientBoosting', 0.95), ('RandomForest', 0.95), ('RandomForest', 'log2'),
            ('GradientBoosting', 'log2'), ('RandomForest', 0.95), ('RandomForest', 0.95),
            ('GradientBoosting', 0.95)
        ]

        for model_type, max_features in models:
            regression_model, classification_model = self.__build_model(model_type, max_features)
            self.add(regression_model, classification_model, trials=self.trials)

    def __build_model(self, model_type: str, max_features: Union[str, float]) -> Tuple[object, object]:
        """
        Creates a regression and classification model based on the given type and feature set.

        Args:
            model_type (str): The type of model to create ('RandomForest' or 'GradientBoosting').
            max_features (Union[str, float]): The maximum number of features to consider for splitting.

        Returns:
            Tuple[object, object]: A tuple containing the regression and classification models.

        Raises:
            ValueError: If the model type is unsupported.
        """
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
