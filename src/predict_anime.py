import pandas as pd
from load_rating import load_data_rating
from load_anime import load_anime
from unratedmovie import get_unrated_movie_id
from pred_unrated import get_pred_unrated_item
import pickle

def predict_anime():
    rating_data= pd.read_csv('/Users/rianrachmanto/pypro/project/anime-recommender/data/raw/rating.csv')
    anime= pd.read_csv('/Users/rianrachmanto/pypro/project/anime-recommender/data/raw/anime.csv')

    #replace -1 with 0 in rating_data
    rating_data['rating'] = rating_data['rating'].replace(-1,0)

    # Replace this with the desired user ID you want to make predictions for
    user_id = 200

    unrated_movie_id = get_unrated_movie_id(user_id, rating_data=rating_data)

    # Ensure that the user has rated some anime before proceeding with predictions
    if unrated_movie_id:
        # Load the trained model from the pickle file
        model_path = '/Users/rianrachmanto/pypro/project/anime-recommender/model/model.pkl'
        with open(model_path, 'rb') as file:
            trained_model = pickle.load(file)

        # Get predicted unrated anime using the trained model
        predicted_unrated_anime_df = get_pred_unrated_item(user_id=user_id, estimator=trained_model, unrated_movie_id=unrated_movie_id)

        k = 5
        top_anime_svd = predicted_unrated_anime_df.head(k).copy()

        # Create a dictionary mapping anime_id to their respective names and genres
        anime_id_to_name = anime.set_index('anime_id')['name'].to_dict()
        anime_id_to_genre = anime.set_index('anime_id')['genre'].to_dict()

        # Map the anime_id to their names and genres in the 'top_anime_svd' DataFrame
        top_anime_svd['name'] = top_anime_svd['anime_id'].map(anime_id_to_name)
        top_anime_svd['genre'] = top_anime_svd['anime_id'].map(anime_id_to_genre)

        print(top_anime_svd)
    else:
        print("User with ID {} has not rated any anime.".format(user_id))

if __name__ == "__main__":
    predict_anime()
