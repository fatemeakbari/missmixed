import pandas as pd

class DataFrameColumnRounder:
    def __init__(self, df:pd.DataFrame):
        """Initialize the processor with a pandas DataFrame."""
        self.df = df.copy()

    def round_columns(self, columns:list) -> pd.DataFrame:
        """
        Rounds the values of the specified columns and converts them to integers.
        
        Args:
            columns (list): List of column names to round and convert.
        
        Returns:
            pd.DataFrame: Updated DataFrame with specified columns rounded and converted to integers.
        """
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].round().astype(int)
            else:
                print(f"Column '{col}' not found in the DataFrame.")
        return self.df
    
class CategoricalListMaker:
    def __init__(self, df:pd.DataFrame):
        """Initialize the processor with a pandas DataFrame."""
        self.df = df.copy()
        
    def make_categorical_list(self, categorical_columns:list=None, non_categorical_columns:list=None, categorical_index:list=None, non_categorical_index:list=None) -> list:
        """
        Create a list of boolean values in which Trues are instances of categorical columns and Falses are of non-categorical ones.
        
        Args:
            categorical_columns (list): List of column names which are categorical.
            categorical_index (list): List of column indices which are categorical.
            non_categorical_columns (list): List of column names which are non-categorical.
            non_categorical_index (list): List of column indices which are non-categorical.
            NOTE: just one of these should be specified and others are to be remained None.
            
        Returns:
            list: a list of boolean values in which Trues are instances of categorical columns and Falses are of non-categorical ones.
        """
        
        # checking that not more than one parameter is set 
        params = [categorical_columns, non_categorical_columns, categorical_index, non_categorical_index]
        param_no = 0
        for param in params:
            if param != None:
                param_no += 1
        
        if param_no > 1:
            print(f'You cannot set more than one parameter but you did {param_no}')
            print('Please set only one of the categorical_columns, non_categorical_columns, categorical_index or non_categorical_index parameters!')
            return None
        
        categorical_list = [False for i in range(self.df.shape[1])]  #default values
        
        if (categorical_columns!=None):
            categorical_index = [i for i, s in enumerate(self.df.columns) if s in categorical_columns]
            
        elif(non_categorical_columns!=None):
            non_categorical_index = [i for i, s in enumerate(self.df.columns) if s in non_categorical_columns]
        
        if (categorical_index!=None):
            for i in categorical_index:
                categorical_list[i] = True
        
        elif (non_categorical_index!=None):
            for i in range(self.df.shape[1]):
                if i not in non_categorical_index:
                    categorical_list[i] = True
                
        return categorical_list

def train_test_split(X, y, train_size=0.9):
    train_test_split_len = int(len(X) * train_size)
    x_train = X[0: train_test_split_len]
    x_test = X[train_test_split_len:]
    y_train = y[0: train_test_split_len]
    y_test = y[train_test_split_len:]
    return x_train, y_train, x_test, y_test

