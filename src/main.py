from load_rating import load_data_rating
from load_anime import load_anime
from anime_split_train_test import split_train_test
from datautility import util_data
from matrix_train_test_split import train_test_split
from trainmod import train_model

rating_path = '/Users/rianrachmanto/pypro/project/anime-recommender/data/raw/rating.csv'
anime_path = '/Users/rianrachmanto/pypro/project/anime-recommender/data/raw/anime.csv'

rating_data=load_data_rating(rating_path)
anime_data=load_anime(anime_path)

coo_data_train, coo_data_test=split_train_test(rating_data=rating_data, test_size=0.2, random_state=42)

print(coo_data_train.nnz)
print(coo_data_test.nnz)

utility_data=util_data(rating_data)

full_data, train_data, test_data=train_test_split(utility_data, test_size=0.2, random_state=42)

print(train_data.n_ratings, len(test_data))

model_svd=train_model(train_data,test_data)



