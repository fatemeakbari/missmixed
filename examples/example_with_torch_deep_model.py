import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from missmixed import MissMixed, Sequential, CategoricalListMaker, DeepModelImputer

# Initialize data, sequential model, and categorical columns
reference_data = pd.read_excel('Breast_Cancer_MCAR_10.xlsx')
categorical_list_maker = CategoricalListMaker(reference_data)
categorical_columns = categorical_list_maker.make_categorical_list()


# Define the deep learning model
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.network = nn.Sequential(
            nn.LazyLinear(256),  # Specify input features
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


in_features = 10
model = DeepModel()

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Pass optimizer and criterion to the imputer
regression_imputer = DeepModelImputer(model=model,
                                      optimizer=optimizer,
                                      loss=criterion,
                                      epochs=50,
                                      batch_size=64)
base_model = Sequential()
base_model.add(regression_imputer=regression_imputer, classification_imputer=None, trials=1, index=0)

# Create and run the MissMixed instance
miss_mixed = MissMixed(reference_data, initial_strategy='mean', sequential=base_model,
                       categorical_columns=categorical_columns, train_size=0.9, verbose=2)
miss_mixed.fit_transform()

result = miss_mixed.result()

print('Average: ', np.mean(result['scores']))
print(result['scores'])
print(result['imputed_data'].head())
