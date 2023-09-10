def get_unrated_movie_id(user_id, rating_data):
    unique_anime_id=set(rating_data['anime_id'])
    rated_movie_id=set(rating_data.loc[rating_data['user_id']==user_id]['anime_id'])
    unrated_movie_id=unique_anime_id.difference(rated_movie_id)
    return unrated_movie_id