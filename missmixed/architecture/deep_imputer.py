from typing import Union, Literal
import os
import warnings
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
warnings.filterwarnings('ignore')  # Ignore all warnings


class DeepModelImputer:
    def __init__(self, model: Union["Sequential"],
                 batch_size=32,
                 epochs=1,
                 callbacks=None,
                 shuffle=True,
                 optimizer=None,
                 loss=None,
                 device: Literal['auto', 'cpu', 'cuda'] = 'auto'):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.loss = loss

        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if device in ['gpu', 'auto']:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print("GPU is configured.")
                except RuntimeError as e:
                    print(e)
            elif device == 'cpu':
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        except ImportError:
            raise "Not found tensor module"
        if optimizer is None:
            self.optimizer = 'adam'
        if loss is None:
            self.loss = 'mean_squared_error'

    def fit(self, X, y):

        from keras import Model, Sequential  # Lazy import
        import tensorflow as tf
        model_cloned = tf.keras.models.clone_model(self.model)
        self.model = model_cloned
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        return self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                              callbacks=self.callbacks, verbose=False)

    def predict(self, X):
        return self.model.predict(X, verbose=False).reshape(-1)
