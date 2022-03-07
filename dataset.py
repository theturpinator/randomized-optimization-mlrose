from cProfile import label
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer

class Dataset:

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.kwargs = kwargs

    def get_dataset(self, name):

        match name:
            
            case 'wine-quality':
                path = 'data/WineQT.csv'
                label_name = 'quality'
                columns_to_drop_from_X = [label_name, 'Id']

                #NEED TO REMOVE ID

                # read dataset into pandas dataframe
                df = pd.read_csv(path)

                # split into attributes and labels, convert to NumPy
                X = df.drop(columns_to_drop_from_X, axis=1).to_numpy()

                df.loc[df[label_name] < 5, 'new_label'] = 0
                df.loc[(df[label_name] >= 5) & (df[label_name] <= 6), 'new_label'] = 1
                df.loc[df[label_name] > 6, 'new_label'] = 2
                
                y = df['new_label']

            case 'iris':
                X, y = load_iris(return_X_y=True, as_frame=True)

            case 'diabetes':
                X, y = load_diabetes(return_X_y=True, as_frame=True)

            case 'digits':
                X, y = load_digits(return_X_y=True, as_frame=True)

            case 'wine-region':
                X, y = load_wine(return_X_y=True, as_frame=True)

            case 'breast-cancer':
                X, y = load_breast_cancer(return_X_y=True, as_frame=True)



        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, **self.kwargs, random_state=100, stratify=y)

        return X_train, X_test, y_train, y_test


    def print_sample(self):
        print('attributes', self.X.head())
        print('labels', self.y.head())




        

        