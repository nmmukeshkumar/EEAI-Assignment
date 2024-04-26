from model.randomforest import RandomForest

def model_predict(data, df, name):
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    #it makes a template and make it work with diff types of datasets and makes all the train, test, predict, and provide the predicted value
