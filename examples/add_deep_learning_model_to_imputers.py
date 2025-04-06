import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from missmixed import MissMixed, Sequential, CategoricalListMaker, DeepModelImputer
from tensorflow import keras


# Initialize data, sequential model, and categorical columns
reference_data = pd.read_excel('LetterRecognition_MCAR_25.xlsx')
categorical_list_maker = CategoricalListMaker(reference_data)
print(reference_data.columns)
categorical_columns = categorical_list_maker.make_categorical_list(
    categorical_index=[i for i in range(reference_data.shape[1])])


model = keras.Sequential()
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dense(units=256, activation='tanh'))
model.add(keras.layers.Dense(units=512, activation='tanh'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=256, activation='tanh'))
model.add(keras.layers.Dense(units=256, activation='relu'))

early_stopping = [keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)]

regression_imputer = DeepModelImputer(model=model, epochs=10, batch_size=64, callbacks=early_stopping,
                                      optimizer='adam',
                                      loss='mean_squared_error')

classification_imputer = DeepModelImputer(model=model, epochs=10, batch_size=64,
                                      optimizer='adam',
                                      callbacks=early_stopping,
                                      loss='crossentropy')


base_model = Sequential()
base_model.add(regression_imputer=regression_imputer,
               classification_imputer=classification_imputer, trials=1, index=0)

# Create and run the MissMixed instance
miss_mixed = MissMixed(reference_data, initial_strategy='mean', sequential=base_model,
                       categorical_columns=categorical_columns, train_size=0.9, verbose=2)
miss_mixed.fit_transform()


result = miss_mixed.result()

print('Average score: ', result['avg_score'])
print(result['scores'])
print(result['imputed_data'].head())





