import streamlit as st
import pandas as pd
from load_rating import load_data_rating
from load_anime import load_anime
from unratedmovie import get_unrated_movie_id
from pred_unrated import get_pred_unrated_item
import pickle

def predict_anime(user_id):
    rating_data = pd.read_csv('rating.csv')
    anime = pd.read_csv('anime.csv')

    # Replace -1 with 0 in rating_data
    rating_data['rating'] = rating_data['rating'].replace(-1, 0)

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

        return top_anime_svd
    else:
        return "User with ID {} has not rated any anime.".format(user_id)

def main():
    # Load user IDs from the rating data
    rating_data = pd.read_csv('/Users/rianrachmanto/pypro/project/anime-recommender/data/raw/rating.csv')
    user_ids = sorted(rating_data['user_id'].unique())

    # Display the banner image
    banner_image_path = '/Users/rianrachmanto/pypro/project/anime-recommender/images/44772.jpg'
    st.image(banner_image_path, use_column_width=True)

    st.title("Anime Recommender")
    st.write("Select a user ID from the drop-down list to get anime recommendations.")

    # User ID selection using a drop-down list
    user_id = st.selectbox("Select User ID", user_ids)

    if st.button("Get Recommendations"):
        # Call the predict_anime function to get the recommendations
        recommendations = predict_anime(user_id)

        if isinstance(recommendations, pd.DataFrame):
            st.subheader("Top 5 Recommended Anime:")
            st.table(recommendations[['name', 'genre', 'predicted_rating']])
        else:
            st.write(recommendations)

if __name__ == "__main__":
    main()

