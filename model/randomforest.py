import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Config')))
from Config import *
from numpy import *
import random
num_folds = 0
seed =0

np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.bsemdl = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.mdl = MultiOutputClassifier(self.bsemdl, n_jobs=-1)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        print("Training")
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        print("Evaluating")
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        data.y_test = data.y_test.to_numpy()
        print("Printing Results")
        induvidual_accuracies = []
        array_length = len(data.y_test)
        dep_length = len(Config.CLASS_COL)
        for i in range(0,array_length):
            test_a = self.predictions[i]
            test_b = data.y_test[i]
            correct_count = 0
            print("Instance ",i+1," :")
            print("Correct labels : ",test_b)
            print("Predicted labels : ",test_a)
            for j in range(0,dep_length):
                if (test_a[j] == test_b[j]):
                    correct_count = correct_count +1
                else:
                    break
            induvidual_accuracies.append(round((correct_count/dep_length),2))
            print("Instance Accuracy : ",round((correct_count/dep_length),2))
        print("***********************************************************")
        print("Final Model Accuracy : ",round((sum(induvidual_accuracies)/array_length)*100,2)," %")
        print("***********************************************************")



    def data_transform(self) -> None:
        ...

