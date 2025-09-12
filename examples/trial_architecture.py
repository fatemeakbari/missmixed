import pandas as pd
import numpy as np
from missmixed import MissMixed, Sequential, CategoricalListMaker, DataFrameColumnRounder

# Initialize data, sequential model, and categorical columns
data = pd.read_excel('Breast_Cancer_MCAR_10.xlsx')
categorical_list_maker = CategoricalListMaker(data)
categorical_columns = categorical_list_maker.make_categorical_list()

base_model = Sequential(trials=2)

# Create and run the MissMixed instance
miss_mixed = MissMixed(data, initial_strategy='mean', sequential=base_model, metric='r2_accuracy',
                        categorical_columns=categorical_columns, train_size=0.9, verbose=1)
miss_mixed.fit_transform()

result = miss_mixed.result()

print('Average score: ', result['avg_score'])
print(result['scores'])

imputed_data = result['imputed_data']
imputed_data.columns = data.columns  # Specifying that the column names of the imputed df must be as they are in the original data

print(imputed_data.head())