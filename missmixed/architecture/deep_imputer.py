import os
import warnings
from typing import Union, Literal

import numpy as np
import pandas as pd

from missmixed.utils.shared_data import SharedData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
warnings.filterwarnings('ignore')  # Ignore all warnings

"""
DeepModelImputer: A class for imputing missing values using deep learning models.

This class provides functionality to impute missing values in datasets using a deep learning model.
It supports both categorical and numerical data, and allows customization of the model's training
parameters such as batch size, epochs, loss function, and optimizer.

Attributes:
    model (Union["Sequential"]): The deep learning model to use for imputation.
    batch_size (int): The batch size for training the model. Default is 32.
    epochs (int): The number of epochs for training the model. Default is 1.
    callbacks (list): List of callbacks to apply during training. Default is None.
    shuffle (bool): Whether to shuffle the training data. Default is True.
    optimizer (str): The optimizer to use for training. Default is 'adam'.
    loss (Union['categorical', 'sparse_categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error']):
        The loss function to use for training. Default is None.
    device (Literal['auto', 'cpu', 'cuda']): The device to use for training ('auto', 'cpu', or 'cuda'). Default is 'auto'.
    shared (SharedData): A shared data object for storing intermediate results.

Methods:
    fit(X, y): Fits the deep learning model to the provided data.
    predict(X): Predicts the missing values using the trained model.
"""


class DeepModelImputer:
    def __init__(self, model: Union["Sequential"],
                 batch_size=32,
                 epochs=1,
                 callbacks=None,
                 shuffle=True,
                 optimizer=None,
                 verbose=False,
                 loss=Union[
                     'categorical', 'sparse_categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error'],
                 device: Literal['auto', 'cpu', 'cuda'] = 'auto'):
        """
        Initializes the DeepModelImputer class.

        Args:
            model (Union["Sequential"]): The deep learning model to use for imputation.
            batch_size (int): The batch size for training the model. Default is 32.
            epochs (int): The number of epochs for training the model. Default is 1.
            callbacks (list): List of callbacks to apply during training. Default is None.
            shuffle (bool): Whether to shuffle the training data. Default is True.
            optimizer (str): The optimizer to use for training. Default is 'adam'.
            loss (Union['categorical', 'sparse_categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error']):
                The loss function to use for training. Default is None.
            device (Literal['auto', 'cpu', 'cuda']): The device to use for training ('auto', 'cpu', or 'cuda'). Default is 'auto'.
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.shuffle = shuffle
        self.verbose = verbose
        self.optimizer = optimizer
        self.loss = loss
        self.shared = SharedData()

        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if device in ['gpu', 'auto']:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
            elif device == 'cpu':
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        except ImportError:
            raise "Not found tensor module"
        if optimizer is None:
            self.optimizer = 'adam'

    def fit(self, X, y):
        """
        Fits the deep learning model to the provided data.

        Args:
            X (np.ndarray or pd.DataFrame): The input features for training.
            y (np.ndarray or pd.Series): The target values for training.

        Returns:
            History: A History object containing training metrics.
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import utils as np_utils

        if self.shared.is_categorical():
            if self.loss is None:
                self.loss = 'crossentropy'
            num_of_classes = self.shared.num_of_class()

            if self.loss == 'crossentropy':
                if num_of_classes == 2:
                    self.loss = 'binary_crossentropy'
                    self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))
                else:
                    self.loss = 'categorical_crossentropy'
                    y = np.asarray(y).astype('float32')
                    y = np_utils.to_categorical(y, num_classes=num_of_classes)
                    self.model.add(keras.layers.Dense(units=num_of_classes, activation='softmax'))
            else:
                self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))
        else:
            if self.loss is None:
                self.loss = 'mean_squared_error'
            self.model.add(keras.layers.Dense(units=1, activation='linear'))

        model_cloned = tf.keras.models.clone_model(self.model)
        self.model = model_cloned
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        return self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                              callbacks=self.callbacks, verbose=self.verbose)

    def predict(self, X):
        """
        Predicts the missing values using the trained model.

        Args:
            X (np.ndarray or pd.DataFrame): The input features for prediction.

        Returns:
            np.ndarray: The predicted values.
        """
        result = self.model.predict(X, verbose=False)
        if self.shared.is_categorical() and self.loss == 'categorical_crossentropy':
            result = np.argmax(result, axis=1)
        return result.reshape(-1)
