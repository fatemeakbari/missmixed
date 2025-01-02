import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from missmixed import MissMixed, Sequential, CategoricalListMaker, DeepModelImputer
from tensorflow import keras


# Initialize data, sequential model, and categorical columns
reference_data = pd.read_excel('Breast_Cancer_MCAR.xlsx')
categorical_list_maker = CategoricalListMaker(reference_data)
categorical_columns = categorical_list_maker.make_categorical_list()



model = keras.Sequential()
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dense(units=256, activation='tanh'))
model.add(keras.layers.Dense(units=512, activation='tanh'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=256, activation='tanh'))
model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dense(units=1))


regression_imputer = DeepModelImputer(model=model, epochs=200, batch_size=32,
                                      optimizer='adam',
                                      loss='mean_squared_error')
classification_imputer = RandomForestClassifier(n_estimators=100,
                                                max_depth=40,
                                                n_jobs=-1,
                                                max_features='sqrt',
                                                min_samples_leaf=1,
                                                min_samples_split=2)


base_model = Sequential()
base_model.add(regression_imputer=regression_imputer,
               classification_imputer=classification_imputer, trials=1, index=0)

# Create and run the MissMixed instance
miss_mixed = MissMixed(reference_data, initial_strategy='mean', sequential=base_model,
                       categorical_columns=categorical_columns, train_size=0.9, verbose=2)
miss_mixed.fit_transform()


result = miss_mixed.result()

print('Average: ', np.mean(result['scores']))
print(result['scores'])
print(result['imputed_data'].head())





