import pandas as pd
import numpy as np
from missmixed import MissMixed, Sequential, CategoricalListMaker

# Initialize data, sequential model, and categorical columns
reference_data = pd.read_excel('LetterRecognition_MCAR_25.xlsx')
categorical_list_maker = CategoricalListMaker(reference_data)
categorical_columns = categorical_list_maker.make_categorical_list(categorical_columns=['xbox ', 'x2bar'])

base_model = Sequential()

# Create and run the MissMixed instance
miss_mixed = MissMixed(reference_data, initial_strategy='mean', sequential=base_model, metric='r2_accuracy',
                        categorical_columns=categorical_columns, train_size=0.9, verbose=2)
miss_mixed.fit_transform()

result = miss_mixed.result()

print('Average score: ', np.mean(result['scores']))
print(result['scores'])
print(result['imputed_data'].head())




