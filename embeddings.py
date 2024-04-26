import numpy as np
import pandas as pd
from Config import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df:pd.DataFrame):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()  # makes the fit transform function to make the tokenized value in a good format
    return X

# makes the tokenisation part by concat the two columns with the df and returns the value based on the numpy array