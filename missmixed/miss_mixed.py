import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Literal, Dict, Any
from missmixed.utils import train_test_split, normalize
from missmixed.architecture import Sequential
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from tqdm import tqdm

acceptable_metrics = ['r2_accuracy', 'mse']

ITERATION_BAR_FORMAT = "{l_bar}{bar}| Iteration {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


class MissMixed:
    def __init__(self,
                 raw_data: pd.DataFrame,
                 sequential: Sequential,
                 categorical_columns: List[bool],
                 metric: Literal['r2_accuracy', 'mse'] = 'r2_accuracy',
                 initial_strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
                 train_size: float = 0.9,
                 early_stopping: bool = False,
                 iter_per_stopping: int = 1,
                 tolerance_percentage: float = 0.1,
                 verbose: Literal[0, 1, 2] = 0
                 ):
        self.raw_data = raw_data
        self.sequential = sequential
        self.categorical_columns = categorical_columns
        self.train_size = train_size
        self.early_stopping = early_stopping
        self.iter_per_stopping = iter_per_stopping
        self.tolerance_percentage = tolerance_percentage
        self.verbose = verbose
        self.__clean_raw_data()
        self.imputed_df = pd.DataFrame(SimpleImputer(strategy=initial_strategy).fit_transform(self.raw_data))
        self.num_of_columns = self.imputed_df.shape[1]
        self.metric_iters = []
        self.__init_metrics(metric)


    def __clean_raw_data(self):
        non_nan_count_per_column = self.raw_data.notna().sum()
        columns_to_be_dropped = non_nan_count_per_column[non_nan_count_per_column <= 1].index

        # Drop those columns
        # df_dropped = df.drop(columns=columns_with_one_non_nan)

        if columns_to_be_dropped.size >= 1:
            self._log(0, f'Columns with all NaN values {columns_to_be_dropped} are dropped')
        self.raw_data.drop(columns=columns_to_be_dropped, inplace=True)


    def __init_metrics(self, metric):
        if metric not in acceptable_metrics:
            raise ValueError(f'Invalid metric {metric}. Only {acceptable_metrics} are acceptable.')
        self.metric_direction = 1 if metric == 'r2_accuracy' else -1
        self.non_categorical_metric = r2_score if metric == 'r2_accuracy' else mean_squared_error
        self.categorical_metric = accuracy_score if metric == 'r2_accuracy' else mean_squared_error

        self.max_metric_tests = np.full(self.imputed_df.shape[1], -np.inf * self.metric_direction)

    def __set_metric(self, categorical: bool):
        self.metric = self.categorical_metric if categorical else self.non_categorical_metric

    def fit_transform(self):
        updated_columns_count = []
        # keep number of columns that updated per iteration
        for idx, imputer in enumerate(self.__iteration_progress_bar()):
            self._log(1, f'Iteration {idx + 1}/{len(self.sequential.imputers)}')
            count = self.process_each_imputer(imputer)

            self._log(1, f'---- {count} columns updated ----')
            self._log(1, '--' * 40)
            updated_columns_count.append(count)

            if self.__check_early_stopping(updated_columns_count):
                print('Early stopping condition hits!')

    def __iteration_progress_bar(self):
        if self.verbose == 0:
            iteration_progress_bar = tqdm(
                self.sequential.imputers,
                desc="Imputing...: ",
                bar_format=(
                    ITERATION_BAR_FORMAT
                ))
        else:
            iteration_progress_bar = self.sequential.imputers
        return iteration_progress_bar

    def __check_early_stopping(self, updated_columns_count: List[int]) -> bool:
        if self.early_stopping:
            if len(updated_columns_count) >= self.iter_per_stopping:
                for updated_columns in updated_columns_count[-1 * self.iter_per_stopping:]:
                    if updated_columns / self.num_of_columns > self.tolerance_percentage:
                        return False
                return True
        return False

    def process_each_imputer(self, imputer) -> int:
        columns_updated_by_imputer = 0
        for col_index in range(self.num_of_columns):
            categorical = self.categorical_columns[col_index]
            imputer.set(categorical)
            self.__set_metric(categorical)

            if imputer.model is None:
                self._log(2, f'Imputer skipped because not found proper imputer model ')
                continue
            self._log(2, f"Imputing column {col_index + 1}/{self.num_of_columns}")

            column_update_count, score_per_iteration = self.imputation_pipeline(imputer, col_index)

            #TODO correct name
            columns_updated_by_imputer += column_update_count

            # self._log(1, f'{column_update_count} columns updated')
        self._log(1,f'Average {self.metric.__name__} train: {np.mean(score_per_iteration["train"])}, test: {np.mean(score_per_iteration["test"])}')

        return columns_updated_by_imputer

    # pipline indicate to train, evaluation and imputation
    def imputation_pipeline(self, imputer, col_index: int) -> tuple[bool, dict[str, list[Any]]]:
        change = False
        score_per_iteration = {'train': [], 'test': []}
        ds, impute_ds = self.__dataset_preparation(col_index)
        if len(ds['y_test']) >= 2:
            metric_scores, models = [], []
            # Train model and select best model based score on test data
            for _ in range(imputer.trials):
                imputer.fit(ds['x_train'], ds['y_train'])
                y_pred_train = np.maximum(imputer.predict(ds['x_train']), 0.0)
                y_pred_test = np.maximum(imputer.predict(ds['x_test']), 0.0)
                metric_scores.append(
                    {
                        'train': self.metric(ds['y_train'], y_pred_train),
                        'test': self.metric(ds['y_test'], y_pred_test)
                    }
                )
                models.append(imputer.copy())

            best_index = np.argmax([m['test'] * self.metric_direction for m in metric_scores])
            best_metric_score = metric_scores[best_index]
            score_per_iteration['train'].append(best_metric_score['train'])
            score_per_iteration['test'].append(best_metric_score['test'])

            self._log(2, f"Best {self.metric.__name__} results: {best_metric_score}")
            if self.can_impute(col_index, best_metric_score['test']):
                self.apply_best_model(models[best_index], impute_ds, col_index)
                change = True
                self._log(2, '-- Column updated --')

        return change, score_per_iteration

    def can_impute(self, col_index: int, test_score: float) -> bool:

        do = self.max_metric_tests[col_index] * self.metric_direction < test_score * self.metric_direction
        if do:
            self.max_metric_tests[col_index] = test_score

        return do

    def apply_best_model(self, model, impute_dataset, col_index: int):
        y_pred_to_impute = model.predict(impute_dataset['x'])
        self.imputed_df.loc[impute_dataset['y'].index, col_index] = y_pred_to_impute

    def __dataset_preparation(self, col_index: int):
        features_df = self.imputed_df.drop(columns=[col_index])
        target_series = self.raw_data.iloc[:, col_index]
        normalized_features = normalize(features_df.columns, features_df)
        y_non_missing = target_series.dropna()
        x_non_missing = normalized_features.loc[y_non_missing.index]
        y_missing = target_series[target_series.isnull()]
        x_missing = normalized_features.loc[y_missing.index]
        x_train, y_train, x_test, y_test = train_test_split(x_non_missing, y_non_missing, train_size=self.train_size)

        train_dataset = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }

        impute_dataset = {
            'x': x_missing,
            'y': y_missing
        }

        return train_dataset, impute_dataset

    def result(self):
        return {
            'imputed_data': self.imputed_df,
            'scores': self.max_metric_tests
        }

    def _log(self, level, *message):

        if self.verbose >= level:
            print(" ".join(map(str, message)))
