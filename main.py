from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    df = get_input_data()
    return  df

def preprocess_data(df):
    df =  de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    group=['Design Decision 1: Chained Multi-outputs','Design Decision 2: Hierarchical Modelling']
    i=1
    for name, group_df in grouped_df:
        print(f"Group - {group[i-1]}:")
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
        i = i+1
        print(150*'*')

