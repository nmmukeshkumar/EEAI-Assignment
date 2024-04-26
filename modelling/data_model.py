import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        y = df[Config.CLASS_COL]  #used to extract the target variable from the dataframe

        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X, y, test_size=0.2, random_state=42)
        self.y = y
        self.embeddings = X


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_embeddings(self):
        return  self.embeddings
    # these functions are used to handle the data splitting process, and all the methods and making the interference for accessing the data