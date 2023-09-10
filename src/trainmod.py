from surprise import SVD
from surprise import accuracy
import pickle
import joblib
import mlflow
import dagshub

def train_model(train_data,test_data):
    '''
    This function trains the SVD model on the utility matrix.
    input: utility_data (surprise Dataset object)
    output: trained_model (surprise SVD object)
    '''
    dagshub.init("anime-recommender-svd", "rachmanto.rian", mlflow=True)
    mlflow.start_run()
    model_svd=SVD(n_factors=100, random_state=42)
    model_svd.fit(train_data)

    #check accuracy

    test_pred=model_svd.test(test_data)
    test_rmse=accuracy.rmse(test_pred, verbose=True)
    print("Test RMSE of SVD model: ", test_rmse)

    #saved trained model
    with open('/Users/rianrachmanto/pypro/project/anime-recommender/model/model.pkl', 'wb') as file:
        pickle.dump(model_svd, file)

    



    return model_svd